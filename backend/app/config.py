"""运行时配置。

所有配置集中在这里，从环境变量解析，且**全部带合理默认值**，满足需求文档
“配置可选不可必、零配置即可跑”的硬约束。其它模块（路由 / 引擎 / 并发 / 音色）
统一从这里读取，避免把模型路径、默认参数散落到各处。

与 kokoro 的关键差异：ChatTTS 是 PyTorch 重模型，默认依赖 GPU 才有可接受速度，
因此这里额外提供 ``DEVICE`` 自适应与 ``MAX_CONCURRENCY`` 默认 1（显存约束）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    """读取正整数环境变量；非法值回退默认值，保证零配置也能启动。"""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _float_env(name: str, default: float) -> float:
    """读取正浮点环境变量；非法或非正值回退默认值。"""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


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


def _resolve_device() -> str:
    """解析推理设备。

    ``DEVICE=auto``（默认）时自动探测 CUDA：有 GPU 用 ``cuda``，否则回退 ``cpu``。
    显式设为 ``cuda`` / ``cpu`` 时直接采用。探测在导入期完成，开销很小。
    """

    raw = (os.getenv("DEVICE") or "auto").strip().lower()
    if raw in ("cuda", "cpu"):
        return raw
    # auto：尝试探测 CUDA。torch 未安装或无 GPU 时一律回退 CPU。
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 - 探测失败时安全回退 CPU
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
    max_segment_chars: int = _int_env("MAX_SEGMENT_CHARS", 120)
    # 流式接口专用、更小的分段上限：分片更多 → 首片更快到达、全程更流畅（代价是推理
    # 次数增多、总合成时间略升、衔接处可能略碎）。它同时是短段合并的目标长度，因此调小
    # 也会减弱短段合并力度。普通接口 POST /api/tts 仍用 MAX_SEGMENT_CHARS。
    stream_max_segment_chars: int = _int_env("STREAM_MAX_SEGMENT_CHARS", 50)
    silence_ms_between_segments: int = _int_env("SILENCE_MS_BETWEEN_SEGMENTS", 120)

    # ---- 并发 / 输出 ----
    # ChatTTS 单次推理显存占用较大，默认并发设为 1（区别于 kokoro 按 CPU 核数）。
    max_concurrency: int = _int_env("MAX_CONCURRENCY", 1)
    queue_timeout: float = _float_env("QUEUE_TIMEOUT", 60.0)
    sample_rate: int = _int_env("SAMPLE_RATE", 24000)


settings = Settings()
