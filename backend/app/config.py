"""运行时配置。

所有配置集中在这里，从环境变量解析，且**全部带合理默认值**，满足需求文档
“配置可选不可必、零配置即可跑”的硬约束。其它模块（路由 / 引擎 / 并发 / 音色）
统一从这里读取，避免把模型路径、默认参数散落到各处。

与 kokoro 的关键差异：ChatTTS 是 PyTorch 重模型，默认依赖 GPU 才有可接受速度，
因此这里额外提供 ``DEVICE`` 自适应与 ``MAX_CONCURRENCY`` 默认 1（显存约束）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("chattts.config")


def _int_env(name: str, default: int) -> int:
    """读取正整数环境变量；非法值发出 warning 并回退默认值。"""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "环境变量 %s=%r 不是合法整数，已回退默认值 %d。", name, raw, default
        )
        return default
    if value <= 0:
        logger.warning(
            "环境变量 %s=%r 必须为正整数，已回退默认值 %d。", name, raw, default
        )
        return default
    return value


def _float_env(name: str, default: float) -> float:
    """读取正浮点环境变量；非法或非正值发出 warning 并回退默认值。"""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "环境变量 %s=%r 不是合法浮点数，已回退默认值 %s。", name, raw, default
        )
        return default
    if value <= 0:
        logger.warning(
            "环境变量 %s=%r 必须为正浮点数，已回退默认值 %s。", name, raw, default
        )
        return default
    return value


def _resolve_stream_segment_chars(default: int = 100) -> int:
    """解析流式「后续分片目标上限」``STREAM_SEGMENT_CHARS``（见 7.6）。

    向后兼容：若未设置新变量、但设置了旧的 ``STREAM_MAX_SEGMENT_CHARS``，则把旧值作为
    ``STREAM_SEGMENT_CHARS`` 读取并告警（旧变量语义已由「硬切上限」改为「后续分片目标
    上限参考」，见 6.1）。
    """

    if os.getenv("STREAM_SEGMENT_CHARS") is not None:
        return _int_env("STREAM_SEGMENT_CHARS", default)
    if os.getenv("STREAM_MAX_SEGMENT_CHARS") is not None:
        logger.warning(
            "STREAM_MAX_SEGMENT_CHARS 已弃用：将作为 STREAM_SEGMENT_CHARS（后续分片目标"
            "上限）读取。请改用 STREAM_SEGMENT_CHARS / STREAM_FIRST_SEGMENT_CHARS。"
        )
        return _int_env("STREAM_MAX_SEGMENT_CHARS", default)
    return default


def _default_model_dir() -> Path:
    """推导默认模型目录（探测链：MODEL_DIR → repo/models → /app/models）。

    本地开发时项目根目录下的 ``models/`` 最方便；Docker 运行时资产固化在
    ``/app/models``。两者都可通过 ``MODEL_DIR`` 覆盖。返回第一个已存在的候选；
    都不存在时返回首选项（让后续下载逻辑去创建它）。
    """

    current_file = Path(__file__).resolve()
    candidates = (
        # 本地源码布局：repo/backend/app/config.py -> repo/models
        current_file.parents[2] / "models",
        # Docker 运行时布局：/app/app/config.py -> /app/models
        current_file.parents[1] / "models",
        Path("/app/models"),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_device(raw: str | None = None) -> str:
    """解析推理设备。

    ``DEVICE=auto``（默认）时自动探测 CUDA：有 GPU 用 ``cuda``，否则回退 ``cpu``。
    显式设为 ``cuda`` / ``cpu`` 时直接采用。

    探测会触发 ``import torch``，开销较大；因此 ``Settings`` 不在 import 期间
    立即调用，而是用 ``default_factory`` 推迟到实例化（lifespan 中）才解析。
    """

    raw = (raw if raw is not None else os.getenv("DEVICE") or "auto").strip().lower()
    if raw in ("cuda", "cpu"):
        return raw
    # auto：尝试探测 CUDA。torch 未安装或无 GPU 时一律回退 CPU。
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - 探测失败时安全回退 CPU
        logger.warning("CUDA 设备探测失败，回退 CPU。", exc_info=True)
        return "cpu"


@dataclass(frozen=True)
class Settings:
    """服务配置（不可变）。

    所有字段都带默认值；冻结 dataclass 避免运行期被意外篡改。
    """

    # ---- HTTP / 模型来源 ----
    port: int = _int_env("PORT", 8000)
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(_default_model_dir())))
    models_auto_download: bool = os.getenv("MODELS_AUTO_DOWNLOAD", "1") != "0"
    model_source: str = os.getenv("MODEL_SOURCE", "huggingface")
    device: str = _resolve_device()

    # ---- 音色 / 合成默认参数 ----
    # 默认音色 seed：固定一个整数，保证无音色请求的输出在同进程内稳定可复现。
    default_speaker: int = _int_env("DEFAULT_SPEAKER", 2)
    default_speed: float = _float_env("DEFAULT_SPEED", 1.0)

    # ---- 文本切分 / 长度限制 ----
    max_text_len: int = _int_env("MAX_TEXT_LEN", 2000)
    # 普通接口 POST /api/tts 的分片目标上限（不启用双阈值，见 6.8）。
    max_segment_chars: int = _int_env("MAX_SEGMENT_CHARS", 120)
    # ---- 流式双阈值（见 6.8 / 7.6）----
    # 首片目标上限：小，求快，让用户尽快出声。
    stream_first_segment_chars: int = _int_env("STREAM_FIRST_SEGMENT_CHARS", 50)
    # 后续分片目标上限：大，求连贯，减少推理次数与段间割裂。
    stream_segment_chars: int = _resolve_stream_segment_chars(100)
    # 首片无法在自然断点结束时的硬上限。
    stream_first_hard_cap: int = _int_env("STREAM_FIRST_HARD_CAP", 90)
    silence_ms_between_segments: int = _int_env("SILENCE_MS_BETWEEN_SEGMENTS", 120)

    # ---- 并发 / 输出 ----
    # ChatTTS 单次推理显存占用较大，默认并发设为 1（区别于 kokoro 按 CPU 核数）。
    max_concurrency: int = _int_env("MAX_CONCURRENCY", 1)
    queue_timeout: float = _float_env("QUEUE_TIMEOUT", 60.0)
    sample_rate: int = _int_env("SAMPLE_RATE", 24000)


settings = Settings()
