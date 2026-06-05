"""API 请求与响应模型。

所有面向 HTTP 的参数校验集中在这里，路由函数只负责串联业务流程。校验失败由
FastAPI 抛出，main.py 的异常处理器统一转成 400（覆盖默认 422）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import settings


class TTSRequest(BaseModel):
    """``POST /api/tts`` / ``POST /api/tts/stream`` / ``POST /api/tts/preprocess`` 请求体。

    预处理相关字段（``preprocess``/``preprocess_profile``/``prosody``/
    ``allow_control_tokens``）对普通与流式接口**都生效**；分片相关字段
    （``first_segment_chars``/``max_segment_chars``）**仅流式生效**（见 6.1）。
    """

    text: str = Field(..., description="要合成的文本，必填、非空")
    speaker: str | None = Field(
        default=None, description="音色 id（即 seed 的字符串形式），缺省用默认音色"
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速，范围 0.5-2.0")
    refine: bool = Field(default=False, description="是否启用 refine_text 文本规整")
    temperature: float = Field(
        default=0.3, gt=0.0, le=2.0, description="采样温度，越大越随机"
    )
    top_p: float = Field(default=0.7, gt=0.0, le=1.0, description="nucleus 采样阈值")
    top_k: int = Field(default=20, ge=1, le=100, description="top-k 采样候选数")
    format: Literal["wav"] = Field(default="wav", description="当前仅支持 WAV")

    # ---- 文本预处理参数（两接口生效，见 6.1）----
    preprocess: bool = Field(default=True, description="是否启用文本预处理")
    preprocess_profile: Literal["plain", "balanced", "expressive"] = Field(
        default="balanced", description="增强强度：plain / balanced / expressive"
    )
    prosody: Literal["flat", "natural", "dialogue", "narration"] = Field(
        default="natural", description="朗读风格：flat / natural / dialogue / narration"
    )
    allow_control_tokens: bool = Field(
        default=False, description="是否允许用户原文中的 ChatTTS 控制 token 生效"
    )

    # ---- 流式分片参数（仅 /api/tts/stream 生效）----
    first_segment_chars: int | None = Field(
        default=None,
        ge=10,
        le=500,
        description="流式首片目标上限（10~500）；缺省用 STREAM_FIRST_SEGMENT_CHARS",
    )
    # 语义变更（见 6.1）：由「硬切上限」改为「后续分片目标上限参考」，分片器允许为保护
    # 语义单元略微超出。缺省走配置 STREAM_SEGMENT_CHARS。
    max_segment_chars: int | None = Field(
        default=None,
        ge=10,
        le=500,
        description="流式后续分片目标上限（10~500），仅对 /api/tts/stream 生效",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        """校验文本非空且不超过最大长度。不修剪原始值（由引擎层归一化）。"""

        stripped = value.strip()
        if not stripped:
            raise ValueError("文本不能为空")
        if len(stripped) > settings.max_text_len:
            raise ValueError(f"文本长度不能超过 {settings.max_text_len} 个字符")
        return value


class PreprocessSegmentItem(BaseModel):
    """预览分片项（按 ``index`` 顺序）。"""

    index: int = Field(..., description="分片序号，从 0 开始")
    text: str = Field(..., description="该分片的可朗读文本")


class PreprocessChangeItem(BaseModel):
    """单条变更记录。序列化字段名统一为 ``type``/``from``/``to``（见 6.9）。"""

    # 允许以字段名（type_/from_/to）构造，又以别名（from/to）序列化输出。
    model_config = ConfigDict(populate_by_name=True)

    type: str = Field(..., description="变更类型：number / date / unit / acronym ...")
    from_: str = Field(..., alias="from", description="原文片段")
    to: str = Field(..., alias="to", description="转换后片段")


class PreprocessResponse(BaseModel):
    """``POST /api/tts/preprocess`` 响应（见 6.9）。"""

    original_text: str = Field(..., description="用户原始文本")
    normalized_text: str = Field(..., description="预处理后的可朗读文本")
    refine_prompt: str | None = Field(
        default=None, description="D1 决定的最终 refine prompt；token_injection 路径下为 null"
    )
    prosody: str = Field(..., description="本次使用的朗读风格")
    profile: str = Field(..., description="本次使用的增强强度")
    refine_mode: Literal["token_injection", "refine_prompt"] = Field(
        ..., description="D1 路径：token_injection 或 refine_prompt"
    )
    segments: list[PreprocessSegmentItem] = Field(
        default_factory=list, description="预览分片列表"
    )
    changes: list[PreprocessChangeItem] = Field(
        default_factory=list, description="变更记录列表"
    )


class SpeakerResponse(BaseModel):
    """音色展示项 / 随机采样结果。"""

    id: str = Field(..., description="音色标识（seed 的字符串形式）")
    seed: int | None = Field(default=None, description="随机音色种子，可复现")
    display_name: str = Field(..., description="人类可读的音色名")


class HealthResponse(BaseModel):
    """健康检查响应。"""

    status: str = Field(..., description='"ok" 或 "degraded"')
    model_loaded: bool = Field(..., description="模型是否已加载完成")
    device: str = Field(..., description='推理设备："cuda" 或 "cpu"')
    default_speaker: str = Field(..., description="默认音色 id")
    max_concurrency: int = Field(..., description="最大并发推理数")
    queue_timeout: float = Field(..., description="排队超时秒数")


class ErrorResponse(BaseModel):
    """统一错误响应，方便前端展示。"""

    detail: str
