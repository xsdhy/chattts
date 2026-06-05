"""``POST /api/tts/stream`` 流式接口的自动化测试。

覆盖需求 9.1.10 要求的关键场景：
- SSE 事件解析与顺序（start → progress/audio 成对 → done）；
- 音色非法在流开始前返回 400；
- 排队超时在流开始前返回 503（QUEUE_TIMEOUT）；
- 模型资产缺失在流开始前返回 503（MODEL_ASSET_ERROR）；
- 客户端断开后不继续发送事件且释放推理槽位；
- 段间静音一致性：流式分片按 index 拼接与普通 ``synthesize`` 完整输出逐样本一致。

测试不依赖真实模型：通过 monkeypatch 把 ``engine.load`` 置空、把 ``synthesize_segment``
替换为产出静音样本的假实现，因此 ``encode_wav`` / Base64 仍走真实路径，能验证“音频为
合法、可独立解码的 WAV”。
"""

from __future__ import annotations

import base64
import json

import httpx
import numpy as np
import pytest

from backend.app.concurrency import QueueTimeoutError, inference_gate
from backend.app.config import settings
from backend.app.main import app
from backend.app.text_preprocess import PreprocessResult
from backend.app.tts_engine import (
    ModelAssetError,
    SegmentPlan,
    SegmentSynthesis,
    engine,
)


def _fake_plan(seed: int, speed: float, segments, refine_prompt=None) -> SegmentPlan:
    """构造一个假的 ``SegmentPlan``，供 monkeypatch ``plan_segments`` 使用。"""

    segs = list(segments)
    return SegmentPlan(
        seed=seed,
        speed=speed,
        segments=segs,
        refine_prompt=refine_prompt,
        result=PreprocessResult(text="".join(segs)),
    )


def _parse_sse(text: str) -> list[tuple[str | None, dict | None]]:
    """把 SSE 文本解析为 ``[(event, data_dict), ...]`` 列表。

    事件之间以空行（``\\n\\n``）分隔，每个事件包含 ``event:`` 与单行 ``data:``。
    """

    events: list[tuple[str | None, dict | None]] = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        event_name: str | None = None
        data_raw: str | None = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_raw = line[len("data:") :].strip()
        events.append(
            (event_name, json.loads(data_raw) if data_raw is not None else None)
        )
    return events


def _fake_synthesize_segment(segment, index, total, *, seed, speed, **kwargs):
    """假的单段合成：返回一小段静音，采样率与配置一致。"""

    samples = np.zeros(2400, dtype=np.float32)  # 0.1s @ 24kHz
    return SegmentSynthesis(
        index=index,
        total=total,
        samples=samples,
        sample_rate=24000,
        is_final=index == total - 1,
    )


def _client() -> httpx.AsyncClient:
    """构造一个直连 ASGI 应用的异步客户端（不经过真实网络/生命周期）。"""

    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch):
    """默认把引擎打桩，避免任何测试触发真实模型加载。"""

    monkeypatch.setattr(engine, "load", lambda: None)
    monkeypatch.setattr(engine, "synthesize_segment", _fake_synthesize_segment)


async def test_stream_success_event_order(monkeypatch):
    """成功路径：事件顺序为 start → (progress, audio) × N → done，且槽位被释放。"""

    # 固定 3 段，便于断言顺序与数量，不依赖真实切分逻辑。
    monkeypatch.setattr(
        engine,
        "plan_segments",
        lambda text, **kwargs: _fake_plan(2, 1.0, ["一", "二", "三"]),
    )

    slots_before = inference_gate.available

    async with _client() as client:
        resp = await client.post(
            "/api/tts/stream",
            json={"text": "你好世界", "speaker": "2", "speed": 1.0, "format": "wav"},
        )

    assert resp.status_code == 200
    # 响应头：SSE 内容类型 + 关闭代理缓冲 + 禁用缓存。
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-accel-buffering"] == "no"
    assert resp.headers["cache-control"] == "no-cache"

    events = _parse_sse(resp.text)
    names = [name for name, _ in events]

    # 第一个事件必须是 start，最后一个必须是 done。
    assert names[0] == "start"
    assert names[-1] == "done"
    assert events[0][1]["format"] == "wav"
    assert events[0][1]["sample_rate"] == 24000
    assert "request_id" in events[0][1]

    # 中间为 3 组 (progress, audio)，且每组 progress 在前、audio 在后。
    middle = names[1:-1]
    assert middle == ["progress", "audio", "progress", "audio", "progress", "audio"]

    audio_events = [data for name, data in events if name == "audio"]
    assert [a["index"] for a in audio_events] == [0, 1, 2]
    assert [a["final"] for a in audio_events] == [False, False, True]

    # 每个 audio 的 data 都是合法 Base64，解码后是可独立解码的 WAV（RIFF/WAVE 头）。
    for a in audio_events:
        decoded = base64.b64decode(a["audio"])
        assert decoded[:4] == b"RIFF"
        assert decoded[8:12] == b"WAVE"
        assert a["mime"] == "audio/wav"
        assert a["sample_rate"] == 24000

    # done.chunks 与发出的 audio 事件数量一致。
    assert events[-1][1]["chunks"] == len(audio_events) == 3

    # 正常结束后推理槽位被释放，回到初始可用数。
    assert inference_gate.available == slots_before


async def test_unknown_speaker_returns_400():
    """音色非法：在流开始前返回 400，而不是 SSE error 事件。"""

    async with _client() as client:
        resp = await client.post(
            "/api/tts/stream",
            json={"text": "你好", "speaker": "abc", "speed": 1.0, "format": "wav"},
        )

    assert resp.status_code == 400
    assert not resp.headers["content-type"].startswith("text/event-stream")
    assert "音色" in resp.json()["detail"]


async def test_queue_timeout_returns_503(monkeypatch):
    """排队超时：在流开始前返回 503（QUEUE_TIMEOUT）。"""

    monkeypatch.setattr(
        engine,
        "plan_segments",
        lambda text, **kwargs: _fake_plan(2, 1.0, ["一"]),
    )

    async def _raise_timeout():
        raise QueueTimeoutError("服务繁忙，请稍后重试")

    monkeypatch.setattr(inference_gate, "acquire", _raise_timeout)

    async with _client() as client:
        resp = await client.post(
            "/api/tts/stream",
            json={"text": "你好", "speaker": "2", "speed": 1.0, "format": "wav"},
        )

    assert resp.status_code == 503
    assert "繁忙" in resp.json()["detail"]


async def test_model_asset_error_returns_503(monkeypatch):
    """模型资产缺失：在流开始前返回 503（MODEL_ASSET_ERROR）。"""

    monkeypatch.setattr(
        engine,
        "plan_segments",
        lambda text, **kwargs: _fake_plan(2, 1.0, ["一"]),
    )

    def _raise_asset():
        raise ModelAssetError("模型资产缺失")

    monkeypatch.setattr(engine, "load", _raise_asset)

    async with _client() as client:
        resp = await client.post(
            "/api/tts/stream",
            json={"text": "你好", "speaker": "2", "speed": 1.0, "format": "wav"},
        )

    assert resp.status_code == 503
    assert "模型资产" in resp.json()["detail"]


async def test_disconnect_releases_slot(monkeypatch):
    """客户端断开：不再发送后续事件（无 done），且推理槽位被释放。"""

    monkeypatch.setattr(
        engine,
        "plan_segments",
        lambda text, **kwargs: _fake_plan(2, 1.0, ["一", "二", "三"]),
    )

    # 模拟客户端在进入循环时即处于断开状态：第一段开始前检测到断开并退出。
    async def _always_disconnected(self) -> bool:
        return True

    monkeypatch.setattr(
        "starlette.requests.Request.is_disconnected", _always_disconnected
    )

    slots_before = inference_gate.available

    async with _client() as client:
        resp = await client.post(
            "/api/tts/stream",
            json={"text": "你好", "speaker": "2", "speed": 1.0, "format": "wav"},
        )

    events = _parse_sse(resp.text)
    names = [name for name, _ in events]

    # 收到 start 后立即因断开退出：没有 audio、没有 done。
    assert names[0] == "start"
    assert "audio" not in names
    assert "done" not in names

    # 断开路径同样在 finally 释放了槽位，未泄漏。
    assert inference_gate.available == slots_before


def test_stream_uses_smaller_segments_than_normal():
    """流式分片更小（提升体感速度）：

    同一段中等长度文本，流式用双阈值（首片 ``STREAM_FIRST_SEGMENT_CHARS``、后续片
    ``STREAM_SEGMENT_CHARS``）切分应得到不少于普通 ``MAX_SEGMENT_CHARS`` 的分段数，
    且首片不超过首片硬上限——首片更快到达。``plan_segments`` 为纯函数，不触发推理。
    """

    assert settings.stream_first_segment_chars < settings.max_segment_chars

    # 一段会被普通阈值合并成 1 段、但会被流式首片阈值拆开的中等文本。
    text = "你好，欢迎使用流式语音合成服务，" * 4 + "希望第一声尽快出来。"

    normal = engine.plan_segments(text).segments
    stream = engine.plan_segments(
        text,
        first_chars=settings.stream_first_segment_chars,
        rest_chars=settings.stream_segment_chars,
    ).segments

    assert len(stream) >= len(normal)
    # 首片不超过首片硬上限（双阈值「首片求快」语义）。
    assert len(stream[0]) <= settings.stream_first_hard_cap


async def test_stream_respects_request_max_segment_chars(monkeypatch):
    """请求体的 ``max_segment_chars`` 应覆盖默认 ``STREAM_SEGMENT_CHARS``（后续分片目标上限）。

    构造一段长文本，分别用 ``max_segment_chars=20`` 和 ``max_segment_chars=100`` 请求，
    断言更小的上限产出更多 audio 事件。
    """

    text = "你好，欢迎使用流式语音合成服务，" * 5 + "希望第一声尽快出来。"

    async with _client() as client:
        small = await client.post(
            "/api/tts/stream",
            json={
                "text": text,
                "speaker": "2",
                "speed": 1.0,
                "format": "wav",
                "max_segment_chars": 20,
            },
        )
        big = await client.post(
            "/api/tts/stream",
            json={
                "text": text,
                "speaker": "2",
                "speed": 1.0,
                "format": "wav",
                "max_segment_chars": 100,
            },
        )

    assert small.status_code == 200
    assert big.status_code == 200
    small_audio = [n for n, _ in _parse_sse(small.text) if n == "audio"]
    big_audio = [n for n, _ in _parse_sse(big.text) if n == "audio"]
    assert len(small_audio) > len(big_audio)


async def test_stream_rejects_out_of_range_max_segment_chars():
    """``max_segment_chars`` 超出 schema 限定的 10~500 时返回 400。"""

    async with _client() as client:
        resp = await client.post(
            "/api/tts/stream",
            json={
                "text": "你好",
                "speaker": "2",
                "speed": 1.0,
                "format": "wav",
                "max_segment_chars": 5,
            },
        )

    assert resp.status_code == 400


async def test_streamed_segments_match_full_synthesis(monkeypatch):
    """段间静音一致性（需求 9.9）：

    把各分片按 index 顺序拼接得到的 PCM，应与普通 ``synthesize`` 的完整输出逐样本一致。
    这里 monkeypatch 最底层的 ``_infer_segment`` 返回确定性样本，验证“非末尾分片尾部
    并入段间静音、末尾分片不追加静音”的拼接语义。
    """

    # 恢复 autouse 夹具对 synthesize_segment 的实例级打桩为真实类方法，
    # 这样才能验证真实的“尾部静音拼接”逻辑。
    monkeypatch.delattr(engine, "synthesize_segment", raising=False)
    monkeypatch.setattr(engine, "load", lambda: None)

    segments = ["甲", "乙", "丙"]
    monkeypatch.setattr(
        engine,
        "plan_segments",
        lambda text, **kwargs: _fake_plan(2, 1.0, list(segments)),
    )

    # 跳过对真实 ChatTTS 的依赖：embedding 与采样参数构造都打桩。
    monkeypatch.setattr(
        "backend.app.tts_engine.speaker_registry.embedding_for_seed",
        lambda seed: "dummy-emb",
    )
    monkeypatch.setattr(
        engine,
        "_build_infer_params",
        lambda *, spk_emb, temperature, top_p, top_k: None,
    )

    # 每段返回一段确定性、可区分的样本（值 = 段序号+1），便于核对拼接顺序。
    def _fake_infer(self, segment, *, refine, infer_params, refine_params):
        value = float(segments.index(segment) + 1)
        return np.full(100, value, dtype=np.float32)

    monkeypatch.setattr(
        "backend.app.tts_engine.ChatTTSEngine._infer_segment", _fake_infer
    )

    # 完整合成（普通接口路径）。
    full = engine.synthesize("整段文本", speaker="2", speed=1.0)

    # 逐段拼接（流式接口路径：plan_segments + synthesize_segment，与路由一致）。
    plan = engine.plan_segments(
        "整段文本",
        speaker="2",
        speed=1.0,
        first_chars=settings.stream_first_segment_chars,
        rest_chars=settings.stream_segment_chars,
    )
    planned = plan.segments
    streamed = [
        engine.synthesize_segment(
            seg,
            idx,
            len(planned),
            seed=plan.seed,
            speed=plan.speed,
        )
        for idx, seg in enumerate(planned)
    ]
    concatenated = np.concatenate([seg.samples for seg in streamed])

    assert full.sample_rate == 24000
    assert np.array_equal(full.samples, concatenated)
    # 末尾分片不应以静音收尾（最后一个样本来自语音，而非 0 静音）。
    assert streamed[-1].is_final is True
    assert streamed[-1].samples[-1] != 0.0
