"""FastAPI 应用入口。

一个进程同时提供 REST API 与前端静态文件托管，满足“单进程、单端口”的硬约束：
- 启动时在 ``lifespan`` 内异步加载 ChatTTS 模型；
- 注册健康检查、音色、随机音色、合成等路由；
- 挂载前端构建产物，并提供 SPA fallback；
- 注册全局异常处理：参数错误 → 400，模型 / 排队类 → 503，避免路由层重复 try/except。
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .concurrency import QueueTimeoutError, inference_gate
from .config import settings
from .schemas import (
    HealthResponse,
    PreprocessChangeItem,
    PreprocessResponse,
    PreprocessSegmentItem,
    SpeakerResponse,
    TTSRequest,
)
from .speakers import parse_speaker, speaker_registry
from .text_preprocess import PreprocessOptions
from .tts_engine import ModelAssetError, SynthesisError, encode_wav, engine

logger = logging.getLogger("chattts.service")
# 流式接口日志：记录 request_id、音色、文本长度、分段数、成功/断开状态与耗时。
stream_logger = logging.getLogger("chattts.tts")
logging.basicConfig(level=logging.INFO)

PACKAGE_DIR = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_DIR.parent
PROJECT_ROOT = APP_ROOT.parent
STATIC_CANDIDATES = (
    # 本地源码布局：repo/frontend/dist
    PROJECT_ROOT / "frontend" / "dist",
    # Docker 运行时布局：/app/static
    APP_ROOT / "static",
    PROJECT_ROOT / "static",
)

# SPA fallback 不能误把 FastAPI 内置文档路径吞进 index.html。
_FASTAPI_DOCS_PATHS = frozenset({"docs", "redoc", "openapi.json"})


def _static_dir() -> Path | None:
    """查找前端构建产物目录；不存在时 API 仍可独立运行。"""

    for candidate in STATIC_CANDIDATES:
        if (candidate / "index.html").exists():
            return candidate
    return None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """应用生命周期：启动时预加载模型，失败也保留健康检查可见性。"""

    logger.info(
        "启动 ChatTTS 服务：device=%s model_dir=%s max_concurrency=%d queue_timeout=%.1f",
        settings.device,
        settings.model_dir,
        settings.max_concurrency,
        settings.queue_timeout,
    )
    try:
        # 模型加载是阻塞且耗时的，放到线程池避免卡住事件循环。
        await asyncio.to_thread(engine.load)
    except ModelAssetError as exc:
        # 缺模型时不让进程退出：/health 报告 degraded，/api/tts 返回 503。
        logger.warning("模型未加载：%s", exc)
    yield


app = FastAPI(title="ChatTTS 中文语音合成服务", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------- #
# 全局异常处理：路由内只写业务流程，错误层级集中映射到 HTTP 状态码。
# ---------------------------------------------------------------------- #
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """把 FastAPI/Pydantic 默认 422 转为需求文档约定的 400。"""

    return JSONResponse(content={"detail": exc.errors()}, status_code=400)


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    """业务侧 ``ValueError``（音色非法、文本为空、语速越界等）统一映射为 400。"""

    return JSONResponse(content={"detail": str(exc)}, status_code=400)


@app.exception_handler(SynthesisError)
async def synthesis_error_handler(_: Request, exc: SynthesisError) -> JSONResponse:
    """合成阶段失败（已包装底层错误）→ 400。"""

    return JSONResponse(content={"detail": str(exc)}, status_code=400)


@app.exception_handler(ModelAssetError)
async def model_asset_error_handler(
    _: Request, exc: ModelAssetError
) -> JSONResponse:
    """模型资产缺失 / 加载失败 → 503。"""

    return JSONResponse(content={"detail": str(exc)}, status_code=503)


@app.exception_handler(QueueTimeoutError)
async def queue_timeout_error_handler(
    _: Request, exc: QueueTimeoutError
) -> JSONResponse:
    """推理排队超时 → 503。"""

    return JSONResponse(content={"detail": str(exc)}, status_code=503)


# 挂载前端静态资源目录（assets）。
static_dir = _static_dir()
if static_dir is not None:
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


async def ensure_engine_ready() -> None:
    """路由依赖：在进入业务逻辑前确保模型已加载。

    模型未加载时调用 ``engine.load``（在线程池中执行，避免阻塞事件循环）。
    失败时抛 ``ModelAssetError``，由全局 handler 映射为 503。
    """

    if engine.is_loaded:
        return
    await asyncio.to_thread(engine.load)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """健康检查，供容器编排探活。"""

    return HealthResponse(
        status="ok" if engine.is_loaded else "degraded",
        model_loaded=engine.is_loaded,
        device=engine.device,
        default_speaker=str(settings.default_speaker),
        max_concurrency=settings.max_concurrency,
        queue_timeout=settings.queue_timeout,
    )


@app.get("/api/speakers", response_model=list[SpeakerResponse])
async def list_speakers() -> list[SpeakerResponse]:
    """列出预置可用音色。"""

    return [
        SpeakerResponse(id=item.id, seed=item.seed, display_name=item.display_name)
        for item in speaker_registry.list_speakers()
    ]


@app.post(
    "/api/speakers/random",
    response_model=SpeakerResponse,
    dependencies=[Depends(ensure_engine_ready)],
)
async def random_speaker() -> SpeakerResponse:
    """采样一个新的随机音色，返回其 seed/id（可复现）。"""

    seed, _ = await asyncio.to_thread(speaker_registry.sample_new)
    return SpeakerResponse(id=str(seed), seed=seed, display_name=f"音色 #{seed}")


@app.post(
    "/api/tts",
    responses={
        200: {"content": {"audio/wav": {}}},
        400: {"description": "请求参数非法"},
        503: {"description": "服务繁忙或模型未就绪"},
    },
    dependencies=[Depends(ensure_engine_ready)],
)
async def tts(request: TTSRequest) -> Response:
    """文本转语音，返回 WAV 音频字节流。"""

    # 并发门控：排队超时直接 503，保护显存不被过多并发请求压垮。
    async with inference_gate.slot():
        wav_bytes = await asyncio.to_thread(
            engine.synthesize_wav_bytes,
            request.text,
            speaker=request.speaker,
            speed=request.speed,
            refine=request.refine,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            options=_build_preprocess_options(request),
        )
    return Response(content=wav_bytes, media_type="audio/wav")


def _build_preprocess_options(payload: TTSRequest) -> PreprocessOptions:
    """从请求体构造预处理选项（普通 / 流式 / 预览三处复用，保证语义一致）。"""

    return PreprocessOptions(
        enabled=payload.preprocess,
        profile=payload.preprocess_profile,
        prosody=payload.prosody,
        allow_control_tokens=payload.allow_control_tokens,
        refine=payload.refine,
    )


@app.post(
    "/api/tts/preprocess",
    response_model=PreprocessResponse,
    responses={400: {"description": "请求参数非法"}},
)
async def tts_preprocess(payload: TTSRequest) -> PreprocessResponse:
    """文本预处理预览（落实 6.9 / D5）。

    纯文本处理，**不经过 ``inference_gate`` 推理门控**、不触发模型推理，因此不占用显存
    槽位。返回归一化文本、预览分片、变更记录与 D1 路径信息，供前端「一键整理」展示。
    提交 TTS 时仍发送用户原文，由服务端再次完整预处理（D5）。
    """

    options = _build_preprocess_options(payload)
    # 预览默认用流式双阈值，可被请求字段覆盖（见 6.9）。
    first_chars = payload.first_segment_chars or settings.stream_first_segment_chars
    rest_chars = payload.max_segment_chars or settings.stream_segment_chars

    # plan_segments 为纯函数（不触发推理）：复用它保证预览分片与真实合成分片完全一致。
    plan = await asyncio.to_thread(
        engine.plan_segments,
        payload.text,
        speaker=payload.speaker,
        speed=payload.speed,
        options=options,
        first_chars=first_chars,
        rest_chars=rest_chars,
    )
    result = plan.result
    return PreprocessResponse(
        original_text=payload.text,
        normalized_text=result.text,
        refine_prompt=result.refine_prompt,
        prosody=options.prosody,
        profile=options.profile,
        refine_mode=result.refine_mode,
        segments=[
            PreprocessSegmentItem(index=index, text=segment)
            for index, segment in enumerate(plan.segments)
        ],
        changes=[
            PreprocessChangeItem(type=change.type, **{"from": change.source}, to=change.target)
            for change in result.changes
        ],
    )


def _sse_event(event: str, data: dict) -> bytes:
    """把事件名与数据序列化为一条标准 SSE 记录。

    SSE 线格式要求：``event:`` 行 + 单行 ``data:`` JSON 行，并以一个空行结尾表示事件
    结束。``data`` 内的 JSON 用 ``ensure_ascii=False`` 保留中文，且必须是单行
    （``json.dumps`` 默认不含换行），避免多行 data 被拆成多个事件。
    """

    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


@app.post(
    "/api/tts/stream",
    responses={
        200: {"content": {"text/event-stream": {}}},
        400: {"description": "请求参数非法（含音色不存在）"},
        503: {"description": "排队超时或模型资产缺失"},
    },
)
async def tts_stream(payload: TTSRequest, request: Request) -> StreamingResponse:
    """文本转语音（SSE 流式输出）。

    客户端一次性提交完整 JSON，服务端以 ``text/event-stream`` 逐步返回
    ``start`` / ``progress`` / ``audio`` / ``done`` / ``error`` 事件，音频片段以
    Base64 编码放入 ``audio`` 事件的 ``data`` 字段。

    关键时序（见需求 5、6.2）：**所有可前置的校验都在返回 200、进入 SSE 之前完成**，
    这样才能用标准 HTTP 状态码反馈；一旦发出第一个 SSE 事件就只能用 ``error`` 事件通知
    前端。因此这里在创建 ``StreamingResponse`` 之前依次完成：
    1. 音色显式校验（非法 → 400）；
    2. 参数校验 + 文本切分（``plan_segments``，不触发推理，失败 → 400）；
    3. 模型资产/就绪校验（``engine.load`` 失败 → 503 ``MODEL_ASSET_ERROR``）；
    4. 获取推理槽位（超时 → 503 ``QUEUE_TIMEOUT``）。

    校验类异常（ValueError / ModelAssetError / QueueTimeoutError）直接抛出，由全局
    exception handler 映射到对应 HTTP 状态码。
    """

    # 单次请求 id，用于把日志与一次流式会话关联起来。
    request_id = uuid.uuid4().hex[:8]

    # 1+2. 音色显式校验与参数校验/预处理/切分：失败统一抛 ValueError → 全局 handler 400。
    #    流式接口用双阈值（首片小、后续片大）切分：首片更快到达、后续更连贯（见 6.8）。
    #    客户端可通过 first_segment_chars / max_segment_chars 覆盖默认值（schema 限定 10~500）。
    parse_speaker(payload.speaker)
    first_chars = payload.first_segment_chars or settings.stream_first_segment_chars
    rest_chars = payload.max_segment_chars or settings.stream_segment_chars
    plan = engine.plan_segments(
        payload.text,
        speaker=payload.speaker,
        speed=payload.speed,
        options=_build_preprocess_options(payload),
        first_chars=first_chars,
        rest_chars=rest_chars,
    )
    seed = plan.seed
    selected_speed = plan.speed
    segments = plan.segments
    refine_prompt = plan.refine_prompt

    # 3. 模型资产/就绪校验：缺资产时 ModelAssetError → 全局 handler 503。
    await asyncio.to_thread(engine.load)

    # 4. 进入 SSE 前获取推理槽位：超时 QueueTimeoutError → 全局 handler 503。
    #    采用“整条流持有同一个槽位直到流结束”的方案 A（见 6.2）。
    await inference_gate.acquire()
    # 注意：从 acquire 成功到 StreamingResponse 真正被消费之间，必须保证不再抛异常；
    #     若新增同步初始化，请放进生成器的 try/finally 内，否则会泄漏槽位。

    total = len(segments)
    text_len = len(payload.text)

    async def event_stream() -> AsyncIterator[bytes]:
        """SSE 事件生成器：在 handler 返回后由服务器消费。

        槽位已在进入此生成器前获取，必须在 ``finally`` 中释放，覆盖“正常 done、发生
        error、客户端断开”三种路径，避免槽位泄漏。
        """

        sent = 0
        disconnected = False
        status = "error"
        started_at = time.monotonic()
        try:
            # start 事件：sample_rate 取自配置（见 4.3 说明），与分片实际采样率区分。
            yield _sse_event(
                "start",
                {
                    "request_id": request_id,
                    "format": "wav",
                    "sample_rate": settings.sample_rate,
                },
            )

            for index, segment in enumerate(segments):
                # 每段开始前检测客户端是否断开：断开后停止后续合成与事件发送。
                if await request.is_disconnected():
                    disconnected = True
                    break

                # 约定时序：progress 在前、audio 在后。current 为“即将合成的分段序号”。
                yield _sse_event(
                    "progress",
                    {
                        "current": index + 1,
                        "total": total,
                        "message": f"正在合成第 {index + 1}/{total} 段",
                    },
                )

                # 单段同步推理放到线程池，避免阻塞事件循环；两段之间自然让出控制权。
                try:
                    segment_result = await asyncio.to_thread(
                        engine.synthesize_segment,
                        segment,
                        index,
                        total,
                        seed=seed,
                        speed=selected_speed,
                        refine=payload.refine,
                        temperature=payload.temperature,
                        top_p=payload.top_p,
                        top_k=payload.top_k,
                        refine_prompt=refine_prompt,
                    )
                except SynthesisError as exc:
                    # 已返回 200，无法再改 HTTP 状态码，只能用 error 事件通知前端。
                    yield _sse_event(
                        "error", {"detail": str(exc), "code": "SYNTHESIS_ERROR"}
                    )
                    return
                except Exception:  # noqa: BLE001 - 兜底未预期错误
                    yield _sse_event(
                        "error",
                        {"detail": "服务端内部错误", "code": "INTERNAL_ERROR"},
                    )
                    return

                # 合成完成后再检测一次断开：已在 to_thread 运行的当前段无法中断，但可以
                # 做到“该段返回后不再发送后续事件”。
                if await request.is_disconnected():
                    disconnected = True
                    break

                # 每段编码为可独立解码的 WAV，再 Base64 写入 audio 事件。
                wav_bytes = encode_wav(
                    segment_result.samples, segment_result.sample_rate
                )
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
                yield _sse_event(
                    "audio",
                    {
                        "index": segment_result.index,
                        "audio": audio_b64,
                        "mime": "audio/wav",
                        "sample_rate": segment_result.sample_rate,
                        "final": segment_result.is_final,
                    },
                )
                sent += 1

            if not disconnected:
                # 所有分片发送完成。
                status = "ok"
                yield _sse_event("done", {"chunks": sent, "format": "wav"})
        finally:
            # 三种路径（done / error / 断开）都在此释放槽位，确保不泄漏。
            inference_gate.release()
            if disconnected:
                status = "disconnected"
            elapsed = time.monotonic() - started_at
            stream_logger.info(
                "tts_stream request_id=%s speaker=%s text_len=%d segments=%d sent=%d status=%s elapsed=%.3fs",
                request_id,
                seed,
                text_len,
                total,
                sent,
                status,
                elapsed,
            )

    # 关闭代理缓冲（X-Accel-Buffering）并禁用缓存，保证 SSE 能逐事件下发。
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream; charset=utf-8",
        headers=headers,
    )


def _index_or_hint(message: str) -> FileResponse | dict[str, str]:
    """前端已构建则返回 index.html，否则给出提示。"""

    if static_dir is None:
        return {"message": message}
    return FileResponse(static_dir / "index.html")


@app.get("/", response_model=None)
async def index() -> FileResponse | dict[str, str]:
    """返回 Web GUI；前端未构建时返回提示。"""

    return _index_or_hint("前端尚未构建，请先运行 npm run build。API 可继续使用。")


@app.get("/{path:path}", response_model=None)
async def spa_fallback(path: str) -> FileResponse | dict[str, str]:
    """单页应用兜底路由，刷新非根路径时仍返回 index.html。

    显式排除 ``api/`` / ``health`` / FastAPI 内置文档路径，避免被 catch-all 吞掉。
    """

    if path.startswith("api/") or path == "health" or path in _FASTAPI_DOCS_PATHS:
        raise HTTPException(status_code=404, detail="Not Found")
    return _index_or_hint("前端尚未构建，请先运行 npm run build。")
