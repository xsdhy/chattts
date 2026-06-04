# 非流式输入 SSE 流式输出 TTS 接口需求文档

> 本文档对标姊妹项目 `kokoro-zh-tts` 的同名需求，并适配 ChatTTS 的差异：音色用整数
> `seed`（音色 id 即 seed 的十进制字符串）而非固定音色清单；请求体额外支持
> `refine` / `temperature` / `top_p` / `top_k` 高级采样参数；`MAX_CONCURRENCY` 默认 1
> （单次推理显存占用大）。

## 0. 实现进度跟踪

| # | 任务 | 对应章节 | 状态 |
|---:|---|---|:---:|
| 1 | `concurrency.py` 增加显式 `acquire`/`release` 槽位 API | 6.2 | [x] |
| 2 | `tts_engine.py` 增加逐段产出能力（`plan_segments`/`synthesize_segment`/`synthesize_segments`） | 6.3 | [x] |
| 3 | `schemas.py` 增加音色前置校验函数 `resolve_speaker` | 5、6.1 | [x] |
| 4 | `main.py` 新增 `POST /api/tts/stream` SSE 路由 | 6.4、6.5、6.6 | [x] |
| 5 | 新增后端 pytest 自动化测试 | 9.1.10 | [x] |
| 6 | 前端实现普通/流式模式切换与边收边播 | 7 | [x] |
| 7 | 更新 README 与部署文档 | 11 | [x] |

## 1. 背景

当前服务提供 `POST /api/tts` 接口，前端提交完整文本后需等待后端生成完整 WAV 再一次性
返回。对于较长文本，首包等待时间长，前端只能展示“生成中”，无法及时反馈进度，也无法在
部分音频生成完成后逐步播放。

本需求新增一个“非流式输入、SSE 流式输出”的接口：客户端仍一次性提交完整 JSON，服务端以
`text/event-stream` 持续返回合成进度与音频片段。音频片段使用 Base64 编码，便于通过 SSE
文本通道传输。

## 2. 目标

1. 新增后端接口 `POST /api/tts/stream`，请求体沿用现有 TTS 参数。
2. 响应使用 SSE，逐步输出 `start` / `progress` / `audio` / `done` / `error` 事件。
3. 音频数据通过 Base64 编码放入 SSE `data` 字段。
4. 普通接口 `POST /api/tts` 与流式接口 `POST /api/tts/stream` 并列长期支持。
5. 前端支持普通/流式模式切换；流式模式能接收音频片段、展示进度、边收边播，完成后下载完整音频。

## 3. 非目标

1. 不实现客户端流式输入，文本仍一次性提交。
2. 不实现服务端增量文本合成。
3. 不新增音频格式，仍只支持 WAV。
4. 不替换现有普通接口。

## 4. 接口设计

### 4.1 请求

```http
POST /api/tts/stream
Content-Type: application/json
Accept: text/event-stream
```

请求体与 `POST /api/tts` 一致：

```json
{
  "text": "你好，欢迎使用本地语音合成服务。",
  "speaker": "2",
  "speed": 1.0,
  "refine": false,
  "temperature": 0.3,
  "top_p": 0.7,
  "top_k": 20,
  "format": "wav"
}
```

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---:|---|---|
| `text` | string | 是 | 无 | 待合成文本（≤ `MAX_TEXT_LEN`） |
| `speaker` | string | 否 | `DEFAULT_SPEAKER` | 音色 id（整数 seed 的字符串形式） |
| `speed` | number | 否 | `1.0` | 语速 `0.5`~`2.0` |
| `refine` | bool | 否 | `false` | 是否启用 ChatTTS 文本 refine |
| `temperature` | number | 否 | `0.3` | 采样温度 |
| `top_p` | number | 否 | `0.7` | nucleus 采样阈值 |
| `top_k` | int | 否 | `20` | top-k 候选数 |
| `format` | string | 否 | `wav` | 当前仅支持 `wav` |

### 4.2 成功响应

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream; charset=utf-8
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
```

### 4.3 事件类型

#### `start`

```text
event: start
data: {"request_id":"...","format":"wav","sample_rate":24000}
```

| 字段 | 说明 |
|---|---|
| `request_id` | 单次请求 id，用于日志关联 |
| `format` | 输出格式，当前 `wav` |
| `sample_rate` | 取自配置 `settings.sample_rate`（当前 24000） |

> `start.sample_rate` 来自配置；每个 `audio` 事件的 `sample_rate` 取自该分片实际合成结果
> （`SegmentSynthesis.sample_rate`）。正常情况下二者一致，但实现不得把配置值误当作分片实际值。

#### `progress`

约定时序：**在某一分段开始合成之前**发送对应 `progress`，`current` 为“即将/正在合成的分段
序号”（从 1 开始）；该分段合成完成后再发对应 `audio`。即每段固定 `progress` 在前、`audio` 在后。

```text
event: progress
data: {"current":1,"total":5,"message":"正在合成第 1/5 段"}
```

#### `audio`

`audio` 字段为 Base64 编码的独立可解码 WAV 片段。

```text
event: audio
data: {"index":0,"audio":"UklGR...","mime":"audio/wav","sample_rate":24000,"final":false}
```

| 字段 | 说明 |
|---|---|
| `index` | 片段序号，从 0 开始 |
| `audio` | Base64 编码的 WAV |
| `mime` | `audio/wav` |
| `sample_rate` | 当前片段采样率 |
| `final` | 是否最后一个片段 |

**段间静音一致性（重要）**：`engine.synthesize` 会在相邻分段间插入一段短静音
（`SILENCE_MS_BETWEEN_SEGMENTS`）。为保证“流式下载得到的完整 WAV”与“普通接口输出”一致：

- 每个**非末尾**分片（`final=false`）在其音频末尾包含该分段之后的段间静音；
- **末尾**分片（`final=true`）不追加尾部静音。

实现上：普通接口与流式接口都复用 `plan_segments` + `synthesize_segment`，由后者统一处理
变速与尾部静音，因此两条路径逐段拼接结果逐样本一致。

#### `done`

```text
event: done
data: {"chunks":5,"format":"wav"}
```

#### `error`

已返回 200 并开始 SSE 后无法再改 HTTP 状态码，过程中错误通过 `error` 事件通知。

```text
event: error
data: {"detail":"...","code":"SYNTHESIS_ERROR"}
```

## 5. 错误处理

关键原则：**尽量在 SSE 流开始前完成所有可前置校验**，用标准 HTTP 状态码反馈；一旦发出第一个
SSE 事件就只能用 `error` 事件通知。

| 状态码 | 场景 | 错误码 |
|---:|---|---|
| 400 | 文本为空/过长、语速越界、音色标识非法、格式不支持 | `VALIDATION_ERROR` |
| 503 | 等待推理槽位超时 | `QUEUE_TIMEOUT` |
| 503 | 模型资产缺失或不可读 | `MODEL_ASSET_ERROR` |

前置校验顺序（实现遵守）：

1. **请求体结构校验**：由 `TTSRequest` 完成，失败经 `validation_exception_handler` 转 400。
2. **音色显式校验**：流式接口在进入 SSE 前调用 `schemas.resolve_speaker`（包装
   `speakers.parse_speaker`）；非法 → 400，不进入 SSE。
3. **模型资产/就绪校验**：进入 SSE 前 `await asyncio.to_thread(engine.load)`；缺资产 → 503。
4. **推理槽位获取**：进入 SSE 前 `await inference_gate.acquire()`；超时 → 503。

SSE 开始后的错误（逐段合成失败等）发送 `event: error` 后结束连接。错误码：
`VALIDATION_ERROR` / `QUEUE_TIMEOUT` / `MODEL_ASSET_ERROR` / `SYNTHESIS_ERROR` / `INTERNAL_ERROR`。

## 6. 后端实现要求

### 6.1 改动文件

| 文件 | 改动 |
|---|---|
| `backend/app/main.py` | 新增 `POST /api/tts/stream` 路由，返回 `StreamingResponse` |
| `backend/app/schemas.py` | 新增 `resolve_speaker` 前置校验函数 |
| `backend/app/tts_engine.py` | 新增逐段产出能力 `plan_segments`/`synthesize_segment`/`synthesize_segments`，并重构 `synthesize` 复用之 |
| `backend/app/concurrency.py` | 新增显式 `acquire`/`release` 槽位 API |

### 6.2 并发控制

复用现有 `inference_gate`，不新建第二套信号量。为 `InferenceGate` 增加 `acquire(timeout)` /
`release()`，保留 `slot()` 供普通接口使用。**采用方案 A**：整条流持有同一槽位直到流结束，在
生成器 `finally` 释放，覆盖“正常 done / 发生 error / 客户端断开”三种路径。

### 6.3 引擎逐段产出

- `plan_segments(text, *, speaker, speed, max_chars=None)`：纯校验 + 切分，不触发推理，返回
  `(seed, speed, segments)`。流式路由传入更小的 `STREAM_MAX_SEGMENT_CHARS`（默认 50，普通接口为
  `MAX_SEGMENT_CHARS`=120），使分片更多更小、首片更快到达、全程更流畅——代价是推理次数增多、总
  合成时间略升、衔接处可能略碎。`max_chars` 同时是短段合并的目标长度，调小亦减弱短段合并力度。
- `synthesize_segment(segment, index, total, *, seed, speed, refine, temperature, top_p, top_k)`：
  单段推理 + 变速 + 段间静音（非末尾并入尾部静音），返回 `SegmentSynthesis`。
- `synthesize_segments(...)`：逐段产出生成器，作为可复用封装。
- 路由层把每段推理放到 `asyncio.to_thread`，两段之间让出控制权及时 flush 事件。

### 6.4 路由与流式输出

`StreamingResponse` 返回 `text/event-stream`；注入 `fastapi.Request`（请求体参数名 `payload`）用于
断开检测；SSE 标准行格式（`event:` + 单行 `data:` + 空行分隔）；音频 Base64 写入 `audio.audio`；
响应头 `X-Accel-Buffering: no`、`Cache-Control: no-cache`。

### 6.5 断开与资源释放

每段循环前后 `await request.is_disconnected()` 检测断开；断开/error/done 三路径都在生成器
`finally` 释放槽位；已在 `to_thread` 中运行的当前段无法中断，但保证“该段返回后不再发后续事件”。

### 6.6 日志

记录 `request_id`、音色、文本长度、分段数、成功/失败/断开状态与耗时（`chattts.tts` logger）。

## 7. 前端实现要求

### 7.1 交互

普通/流式模式切换；流式调用 `/api/tts/stream`；按钮文案显示“连接中/生成 N/M/播放中/整理音频”；
生成中禁用文本、音色、语速、高级参数与模式切换；首个可播放片段到达即起播、后续顺序衔接；
支持取消（中止 fetch、停止播放、清理片段）；完成后复用播放器与下载按钮，保留完整音频 Blob。

### 7.2 技术实现

浏览器原生 `EventSource` 仅支持 GET，故用 `fetch` POST + `ReadableStream` 解析 SSE
（`frontend/src/sse.ts`）。`audio` 事件：`atob` → `Uint8Array` → 解析 WAV 分片。

**解码路径区分（重要）**：

- **播放用**：Web Audio API（`frontend/src/streamingPlayer.ts`），按 `index` 顺序定时调度
  `AudioBufferSourceNode` 近无缝衔接（`AudioContext` 会重采样到设备采样率，仅用于即时播放）。
- **下载用**：不要用 `decodeAudioData` 的结果再编码（会被重采样改写采样率）。用最小 WAV
  解析/封装工具（`frontend/src/wav.ts`）直接读取每个分片的原始 PCM 与采样率，按 `index` 顺序
  拼接后用同一采样率重新封装为完整 WAV。

### 7.3 异常处理

非 200 沿用 `formatApiError`；读取流失败展示“流式生成中断，请重试”；`error` 事件展示 `detail`；
用户取消不展示错误并恢复可编辑；新一轮生成前 revoke 旧 `audioUrl`。

## 8. 兼容性与限制

普通/流式接口长期共存；不影响单进程单端口约束；SSE + Base64 带来约 +33% 体积；部分反向代理
默认缓冲响应，部署需关闭缓冲（`X-Accel-Buffering: no`）；流式需边收边播，以首片尽快起播为目标。

## 9. 验收标准

### 9.1 后端

接受与 `/api/tts` 相同请求体；`Content-Type: text/event-stream` 且含 `X-Accel-Buffering: no` /
`Cache-Control: no-cache`；至少含 `start`/`audio`/`done`；长文本多 `progress`/`audio` 且
progress 在前；每个 `audio.audio` 为合法 Base64 且解码为可独立播放 WAV；参数错误（含音色非法）
流前 400；排队超时/资产缺失流前 503；客户端断开后不续发事件且不泄漏槽位；分片按 index 拼接与
`/api/tts` 输出在采样率与时长上一致；新增 pytest 覆盖事件顺序、音色非法 400、排队超时 503、断开
释放槽位、段间静音一致性。

### 9.2 前端

普通/流式切换；流式显示进度；首片到达即起播；后续顺序衔接；完成后下载完整 WAV 且采样率与后端
一致；`error` 事件展示并恢复；取消后中止请求、停止播放、恢复可操作。

### 9.3 回归

`POST /api/tts` 仍正常；Web GUI 普通生成不受影响；`npm run build` 通过；`/health`、`/api/speakers`
正常，原有 pytest 通过。

## 10. 示例

```bash
curl -N \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{"text":"你好，欢迎使用流式语音合成。","speaker":"2","speed":1.0}' \
  http://localhost:8000/api/tts/stream
```

## 11. 文档更新要求

更新 `README.md` 功能特性（补充 SSE 流式）、API 文档（新增 `POST /api/tts/stream`）、前端使用说明
（普通/流式差异），以及部署说明（关闭代理缓冲）。
