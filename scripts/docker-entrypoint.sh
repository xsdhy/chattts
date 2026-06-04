#!/usr/bin/env bash
set -euo pipefail

# 容器入口脚本。
#
# 设计目标：既保留“开箱即用”（镜像已内置模型，零网络依赖即可启动），又支持把
# 模型目录挂载到宿主机卷。当挂载的目录为空（模型缺失）时，按 MODELS_AUTO_DOWNLOAD
# 决定是否自动下载一次到该目录并持久化，后续重启/重建容器直接复用。

MODEL_DIR="${MODEL_DIR:-/app/models}"
# 模型缺失时是否自动下载，默认开启；设为 0 可关闭（缺失时 /health 报告 degraded）。
MODELS_AUTO_DOWNLOAD="${MODELS_AUTO_DOWNLOAD:-1}"

models_present() {
  # ChatTTS custom 加载要求 asset/ 子目录中存在 Decoder / DVAE / GPT / Vocos 四个
  # 关键权重，任一以 .pt 或 .safetensors 结尾的格式都接受。只检查 "asset/ 非空" 过宽，
  # 残缺挂载会让加载阶段抛出难追溯的错误。
  local asset_dir="${MODEL_DIR}/asset"
  [[ -d "$asset_dir" ]] || return 1
  local name
  for name in Decoder DVAE GPT Vocos; do
    if [[ ! -f "$asset_dir/$name.pt" && ! -f "$asset_dir/$name.safetensors" ]]; then
      return 1
    fi
  done
  return 0
}

if models_present; then
  echo "[entrypoint] 模型已就绪：${MODEL_DIR}"
elif [[ "${MODELS_AUTO_DOWNLOAD}" == "1" ]]; then
  echo "[entrypoint] 模型缺失，开始自动下载到 ${MODEL_DIR} ..."
  MODEL_DIR="${MODEL_DIR}" bash /app/scripts/fetch_models.sh
else
  echo "[entrypoint] 警告：模型缺失且已禁用自动下载，/health 将报告 model_loaded=false。" >&2
fi

# 单进程单端口：FastAPI 同时托管 API 与前端静态文件。
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
