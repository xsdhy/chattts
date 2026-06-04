# syntax=docker/dockerfile:1

# ChatTTS 中文 TTS 服务 · CPU 形态多阶段构建。
#
# 阶段：frontend（构建前端）→ assets（下载模型权重）→ runtime（运行时镜像）。
# CPU 形态仅用于功能验证：ChatTTS 是 PyTorch 重模型，CPU 推理很慢。
# 需要可接受的速度请改用 Dockerfile.gpu（CUDA 基础镜像）。

# ---- 阶段 1：构建前端静态文件 ----
FROM node:20-slim AS frontend

WORKDIR /fe

# 先复制依赖清单，利用 Docker layer cache 加速重复构建。
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build


# ---- 阶段 2：下载模型权重 ----
FROM python:3.11-slim AS assets

WORKDIR /workspace

# 仅安装下载所需的 huggingface_hub，避免把重依赖带进该中间层。
RUN pip install --no-cache-dir "huggingface_hub>=0.23"

COPY scripts/fetch_models.sh ./scripts/fetch_models.sh
RUN MODEL_DIR=/workspace/models bash ./scripts/fetch_models.sh


# ---- 阶段 3：运行时镜像 ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_DIR=/app/models \
    MODELS_AUTO_DOWNLOAD=1 \
    DEVICE=cpu

WORKDIR /app

# 运行时系统依赖：
# - libsndfile1：soundfile 写 WAV 所需的系统库；
# - curl：容器 HEALTHCHECK 使用。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 先装 CPU 版 torch（避免默认 PyPI 拉取体积巨大的 CUDA 轮子），再装其余依赖。
RUN pip install --no-cache-dir "torch>=2.1,<3.0" \
        --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY scripts ./scripts
# 镜像内置模型：默认零网络依赖即可启动；挂载卷可覆盖该目录。
COPY --from=assets /workspace/models ./models
COPY --from=frontend /fe/dist ./static
RUN chmod +x ./scripts/*.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# 入口脚本：模型缺失（例如挂载了空目录）时自动下载一次，然后启动服务。
ENTRYPOINT ["./scripts/docker-entrypoint.sh"]
