# ChatTTS 中文语音合成服务

基于官方 [`ChatTTS`](https://github.com/2noise/ChatTTS) 模型的中文文本转语音服务，提供 Web GUI 与 HTTP API。工程范式对标姊妹项目 [`kokoro-zh-tts`](https://github.com/xsdhy/kokoro-zh-tts)：单进程单端口、关注点分离、Docker 优先、配置可选。

> 与 kokoro 的本质差异：kokoro 是 ONNX、CPU-only、轻量；**ChatTTS 是 PyTorch + transformers 的重模型，默认依赖 GPU 才有可接受速度**。本服务在 CPU 与 GPU 两种形态上都能跑，但 CPU 下很慢，仅建议用于功能验证。

## 功能特性

- 中文文本转语音，返回完整 WAV。
- **两种生成模式**：普通生成（`POST /api/tts` 一次性返回 WAV）与 **SSE 流式输出**（`POST /api/tts/stream` 逐段返回、前端边收边播）。
- **随机音色采样**（ChatTTS 特色）：以 `seed` 为音色 id，相同 seed 复现相同音色。
- 可调采样参数：`temperature` / `top_p` / `top_k`，以及文本 `refine` 开关。
- 语速调节（0.5x–2.0x，numpy 线性插值变速，无需 ffmpeg）。
- 长文本分段：按中文标点 / 软断点 / 字数阈值切分后拼接，段间插入静音。
- GPU/CPU 自适应：`DEVICE=auto` 自动探测 CUDA，无 GPU 回退 CPU。
- 并发保护：信号量限制同时推理数，排队超时返回 503（默认并发 1，保护显存）。
- 单进程单端口：FastAPI 同时托管 API 与前端静态文件，含 SPA fallback。
- Docker 开箱即用：多阶段构建（前端 → 模型 → 运行时），内置 `HEALTHCHECK`。

## 目录结构

```text
.
├── Dockerfile                 # CPU 形态多阶段构建
├── Dockerfile.gpu             # GPU 形态（CUDA 基础镜像）
├── docker-compose.yml         # 含 GPU 示例与模型挂载
├── README.md                  # 主文档：用法 + 架构 + API + FAQ
├── docs/                       # 需求文档（SSE 流式接口需求等）
├── conftest.py                # pytest 路径夹具（backend.app.* 导入）
├── pytest.ini                 # pytest 配置
├── backend/
│   ├── requirements.txt       # 运行时依赖
│   ├── requirements-dev.txt   # 测试依赖（不入镜像）
│   └── app/
│       ├── __init__.py
│       ├── main.py            # FastAPI 入口：路由 + 静态托管 + lifespan
│       ├── config.py          # Settings + 环境变量解析 + 设备探测
│       ├── schemas.py         # Pydantic 请求/响应模型
│       ├── tts_engine.py      # 核心引擎：ChatTTS 封装、切分、变速、逐段产出、WAV 编码
│       ├── speakers.py        # 音色：随机采样、seed 缓存、命名清单
│       └── concurrency.py     # 推理信号量限流（acquire/release/slot，排队超时 503）
├── frontend/                  # React + TypeScript + Vite 单页应用
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/{main.tsx,styles.css,sse.ts,streamingPlayer.ts,wav.ts}
├── tests/                     # 后端 pytest（含 SSE 流式接口测试）
├── scripts/
│   ├── fetch_models.sh        # 下载模型权重（huggingface_hub）
│   ├── docker-entrypoint.sh   # 启动前检查/下载模型，再起 uvicorn
│   └── smoke_tts.py           # 最小合成冒烟测试
└── models/                    # 模型目录（构建期下载，可挂载覆盖）
```

## 技术栈

| 层 | 选型 |
|---|---|
| 后端 | Python 3.11、FastAPI、Uvicorn、Pydantic v2 |
| 推理 | 官方 `ChatTTS` 包、`torch`、`transformers`（间接依赖） |
| 音频 | `soundfile` + `numpy`（WAV 编码、变速） |
| 前端 | React 18、TypeScript 5（strict）、Vite 7、lucide-react |
| 容器 | `node:20-slim` → `python:3.11-slim`（CPU）/ `pytorch/pytorch` CUDA（GPU） |

## 设计原则

| 原则 | 说明 |
|---|---|
| 开箱即用 | 镜像内置模型；首次启动若本地无模型则自动下载，下载后可挂载持久化 |
| 内置一切 | 前端构建产物、模型、依赖都在镜像里，模型就绪时运行期零网络依赖 |
| 单进程单端口 | FastAPI 既挂 API 又托管静态前端，带 SPA fallback 路由 |
| 配置可选 | 全部环境变量有默认值；模型目录按 `MODEL_DIR` → `repo/models` → `/app/models` 探测 |
| GPU 可选 | 自动探测 CUDA：有 GPU 用 GPU，无则回退 CPU（ChatTTS 是 PyTorch 重模型，GPU 优先） |

> 非目标：不做用户系统/鉴权/计费、不做客户端流式输入（文本仍一次性提交）、不做训练/微调、
> 不兼容旧 `/tts` GET 接口、不做分布式部署。
> 注：服务端→客户端的流式输出已通过 `POST /api/tts/stream`（SSE）支持。

## 架构设计

### 模块职责

| 模块 | 职责 |
|---|---|
| `main.py` | 创建 FastAPI app；`lifespan` 启动时异步加载模型；注册路由；挂载静态前端；SPA fallback；校验错误转 400 |
| `config.py` | 冻结的 `Settings` dataclass；从环境变量解析；模型目录探测链；`DEVICE=auto` 设备探测 |
| `schemas.py` | `TTSRequest`、`SpeakerResponse`、`HealthResponse` 等 Pydantic 模型 |
| `tts_engine.py` | `ChatTTSEngine`：懒加载（线程安全双检锁）、文本归一化/切分、分段合成 + 段间静音、变速、WAV 编码；自定义异常层级 |
| `speakers.py` | 随机音色采样、`seed → embedding` 缓存（可复现）、预置命名音色清单 |
| `concurrency.py` | `InferenceGate`：`asyncio.Semaphore` 限制并发，`slot()` 上下文管理器，超时抛 `QueueTimeoutError` |

### 启动流程（lifespan）

1. 读取 `Settings`，探测设备（CUDA/CPU）。
2. 在线程池中加载 ChatTTS 模型（本地有权重用 `source="custom"`，否则按需从 HuggingFace 下载）。
3. 把 `chat.sample_random_speaker` 绑定给音色注册表。
4. 预生成默认音色 embedding，保证无音色请求的输出在同进程内稳定。
5. 加载失败不退出进程：`/health` 报告 `degraded`，`/api/tts` 返回 503。

### 推理管线

```text
请求 text
  → 文本归一化（合并空白等）
  → 长文本切分（中文标点 / 软断点 / 字数阈值，合并过短段）
  → 解析音色 seed → speaker embedding（带缓存，可复现）
  → 逐段 infer（spk_emb + 采样参数 top_P/top_K/temperature，可选 refine）
  → 段间插入静音 → 拼接 → numpy float32
  → 变速（numpy 线性插值，无 ffmpeg/scipy）
  → soundfile 编码 24kHz mono WAV bytes → 返回 audio/wav
```

### 异常层级与映射

```text
TTSEngineError (base)
├── ModelAssetError      # 模型缺失/加载失败  → 503
└── SynthesisError       # 合成阶段失败       → 400
QueueTimeoutError        # 并发排队超时       → 503
ValueError               # 参数/音色非法      → 400
```

### 音色管理（ChatTTS 特色）

- 调用 `chat.sample_random_speaker()` 采样 speaker embedding；采样前 `torch.manual_seed(seed)`
  固定全局随机数状态，使**相同 seed → 相同 embedding**，从而音色可复现。
- 以整数 `seed` 作为音色 id，进程内缓存 `seed → embedding`，避免重复计算。
- 启动时固定 `DEFAULT_SPEAKER` 作为默认音色；`POST /api/speakers/random` 每次返回新 seed，
  前端可试听后固定。

## 快速运行（Docker）

```bash
# CPU（很慢，仅验证）
docker run -p 8000:8000 xsdhy/chattts:latest

# GPU + 持久化模型（推荐）
docker run -p 8000:8000 --gpus all -v ./models:/app/models xsdhy/chattts:gpu
```

启动后打开 <http://localhost:8000> 即可使用。首次启动若本地无模型，会自动从
HuggingFace 下载到 `MODEL_DIR`；挂载 `./models` 可持久化，避免重复下载。

### docker compose

```bash
docker compose up -d
```

GPU 形态：在 `docker-compose.yml` 中取消注释 `deploy.resources.reservations.devices`，
并把 `build.dockerfile` 改为 `Dockerfile.gpu`（或使用 GPU 版镜像）。

### 本地构建镜像

```bash
docker build -t xsdhy/chattts:latest .              # CPU
docker build -f Dockerfile.gpu -t xsdhy/chattts:gpu .   # GPU
```

## 本地开发

本地开发统一使用项目根目录下的虚拟环境 `.venv`（与运行镜像一致用 Python 3.11），
所有命令都从仓库根目录执行、以 `.venv/bin/...` 调用，避免污染系统 Python。

### 1. 准备 Python 虚拟环境

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r backend/requirements.txt
```

> macOS 上 `pip install torch` 会自动安装 CPU 版；NVIDIA GPU 环境请按
> [PyTorch 官方指引](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版 torch。

### 2. 下载模型

```bash
MODEL_DIR=./models .venv/bin/python -m pip install "huggingface_hub>=0.23"  # requirements 已含，可跳过
MODEL_DIR=./models bash scripts/fetch_models.sh
```

模型默认从 HuggingFace 仓库 `2Noise/ChatTTS` 整仓下载到 `models/`（保留 `asset/`、
`config/` 结构供 `source="custom"` 加载）。也可不下载，首次启动时按
`MODELS_AUTO_DOWNLOAD=1` 自动下载。

### 3. 启动后端（开发端口 8001，与 `vite.config.ts` 代理一致）

```bash
PORT=8001 .venv/bin/uvicorn backend.app.main:app --reload --port 8001
```

### 4. 前端（HMR，自动代理 `/api`、`/health` 到 8001）

```bash
cd frontend
npm install
npm run dev
```

构建前端供后端托管：`npm run build` 后产物在 `frontend/dist`，后端会自动优先托管。

### 5. 冒烟测试（直接走引擎，不经 HTTP）

```bash
.venv/bin/python scripts/smoke_tts.py --text "你好，世界" --output audio.wav
```

### 6.（可选）后端自动化测试

```bash
.venv/bin/python -m pip install -r backend/requirements-dev.txt
.venv/bin/pytest
```

## API 文档

所有业务接口前缀 `/api`。交互式文档见 `/docs`（FastAPI 自动生成）。

| Method | Path | 说明 | 返回 |
|---|---|---|---|
| GET | `/health` | 健康检查（容器探活） | `HealthResponse` JSON |
| GET | `/api/speakers` | 列出预置音色 | `SpeakerResponse[]` |
| POST | `/api/speakers/random` | 采样一个新随机音色 | `SpeakerResponse` |
| POST | `/api/tts` | 文本合成为 WAV（普通生成） | 二进制 `audio/wav` |
| POST | `/api/tts/stream` | 文本合成（SSE 流式输出） | `text/event-stream` |
| GET | `/` | 前端页面 | HTML |

### `POST /api/tts` 请求体

```jsonc
{
  "text": "你好，欢迎使用 ChatTTS。",  // 必填，非空，<= MAX_TEXT_LEN
  "speaker": "42",                      // 音色 id（seed 字符串），缺省用默认音色
  "speed": 1.0,                          // 0.5 ~ 2.0
  "refine": false,                       // 是否启用文本 refine
  "temperature": 0.3,                    // 采样温度
  "top_p": 0.7,
  "top_k": 20,
  "format": "wav"
}
```

成功返回 `audio/wav` 二进制；失败返回 JSON `{"detail": "..."}`。

示例：

```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，欢迎使用 ChatTTS。","speaker":"42","speed":1.0}' \
  --output out.wav
```

### `POST /api/tts/stream` 流式输出（SSE）

请求体与 `POST /api/tts` 完全一致，但响应为 `text/event-stream`：客户端一次性提交完整文本，
服务端逐段返回事件，音频片段以 **Base64 编码** 放入 `audio` 事件。事件类型：

| 事件 | 说明 |
|---|---|
| `start` | 开始处理：`{request_id, format, sample_rate}` |
| `progress` | 分段进度（在该段合成前发送）：`{current, total, message}` |
| `audio` | 音频片段：`{index, audio(base64 WAV), mime, sample_rate, final}` |
| `done` | 全部完成：`{chunks, format}` |
| `error` | 流内错误：`{detail, code}` |

约定时序为 **`progress` 在前、`audio` 在后**；每个 `audio` 是可独立解码的 WAV，**非末尾分片**
尾部已并入段间静音、**末尾分片**不追加静音，因此前端按 `index` 顺序拼接得到的完整音频与普通
接口输出在采样率与时长上一致。可前置的校验（参数、音色非法、模型未就绪、排队超时）在流开始前以
标准 HTTP 状态码返回（400 / 503）；流开始后的错误通过 `error` 事件通知。

```bash
curl -N http://localhost:8000/api/tts/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"text":"你好，欢迎使用流式语音合成。","speaker":"42","speed":1.0}'
```

> 注意：SSE 是文本协议，Base64 编码会带来约 **+33%** 体积膨胀。部分反向代理（如 nginx）默认会
> 缓冲响应，需关闭缓冲才能逐事件下发——本服务已在响应头设置 `X-Accel-Buffering: no` 与
> `Cache-Control: no-cache`；若前置代理仍缓冲，请在代理侧关闭对应缓冲（如 nginx
> `proxy_buffering off;`）。

### 约定

- `/api/tts` 与 `/api/tts/stream` 共享同一并发门控（`inference_gate`）；排队超过 `QUEUE_TIMEOUT` 返回 503 `{"detail":"服务繁忙，请稍后重试"}`。流式接口整条流持有一个槽位直到结束。
- 模型未加载时 `/health` 返回 `degraded`，`/api/tts`、`/api/tts/stream` 返回 503。
- 校验错误统一返回 **400**（覆盖 FastAPI 默认 422）。
- 随机音色可复现：相同 `seed` 生成相同 embedding，便于固定喜欢的音色。

## 环境变量

全部可选，均有合理默认值，不配也能跑。

| 变量 | 默认 | 说明 |
|---|---|---|
| `PORT` | 8000 | HTTP 端口 |
| `MODEL_DIR` | 自动探测 | 模型目录（`MODEL_DIR` → `repo/models` → `/app/models`） |
| `MODELS_AUTO_DOWNLOAD` | 1 | 缺失时自动下载 |
| `MODEL_SOURCE` | huggingface | `huggingface` / `local` |
| `DEVICE` | auto | `auto` / `cuda` / `cpu` |
| `DEFAULT_SPEAKER` | 2 | 默认音色 seed |
| `DEFAULT_SPEED` | 1.0 | 默认语速 |
| `MAX_TEXT_LEN` | 2000 | 单请求最大字数 |
| `MAX_SEGMENT_CHARS` | 120 | 普通接口 `POST /api/tts` 的长文本切分阈值 |
| `STREAM_MAX_SEGMENT_CHARS` | 50 | 流式接口 `POST /api/tts/stream` 的切分阈值；调小→分片更多更快出首声、调大→更接近普通生成 |
| `SILENCE_MS_BETWEEN_SEGMENTS` | 120 | 段间静音毫秒 |
| `MAX_CONCURRENCY` | 1 | 最大并发推理数（显存约束） |
| `QUEUE_TIMEOUT` | 60.0 | 排队超时秒数 |
| `SAMPLE_RATE` | 24000 | 输出采样率 |

## GPU 说明

- ChatTTS 单次推理显存占用较大，`MAX_CONCURRENCY` 默认设为 1，避免 OOM。
- GPU 形态需宿主机安装 NVIDIA 驱动与 `nvidia-container-toolkit`，运行时加 `--gpus all`。
- `DEVICE=auto` 会在容器内探测 CUDA：有 GPU 用 GPU，否则回退 CPU。

## FAQ

**Q：CPU 下为什么这么慢？**
A：ChatTTS 是 PyTorch 重模型，CPU 推理本身就慢，属预期现象。生产请用 GPU 形态。

**Q：模型下载到哪里？能否离线？**
A：默认下载到 `MODEL_DIR`（容器内 `/app/models`）。挂载该目录即可持久化；镜像
构建阶段也会内置一份模型，模型就绪时运行期零网络依赖。

**Q：如何固定一个喜欢的音色？**
A：用「🎲 随机音色」试听，记下返回的 `seed`，之后在请求里传 `speaker` 为该 seed
即可复现同一音色。

**Q：合成结果每次不一样？**
A：采样存在随机性。固定 `seed`（音色）+ 文本 + 采样参数可获得稳定结果；调低
`temperature` 也会降低随机性。

## 与 kokoro-zh-tts 的差异

| 维度 | kokoro-zh-tts | 本项目（ChatTTS） |
|---|---|---|
| 推理后端 | ONNX Runtime，CPU-only | PyTorch + transformers，GPU 优先 |
| 引擎依赖 | `kokoro-onnx` 包 | 官方 `ChatTTS` 包 |
| 音色 | 固定 100+ 预置音色 | 随机采样 + seed 可复现 + 默认音色 |
| 额外能力 | — | 文本 refine、采样参数（temperature/top_p/top_k）可调 |
| 默认并发 | CPU 核数 | 1（显存约束） |
| 端口 | 8080 | 8000 |
| 其余架构 | 单进程单端口、React+Vite、多阶段 Docker、SPA fallback、健康检查、并发门控 | **完全一致** |

## License

模型版权归 [2noise/ChatTTS](https://github.com/2noise/ChatTTS) 所有，请遵循其许可协议。
本仓库的服务层代码见 `LICENSE`。
