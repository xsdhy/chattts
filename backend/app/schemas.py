"""API 请求与响应模型。

所有面向 HTTP 的参数校验集中在这里，路由函数只负责串联业务流程。校验失败由
FastAPI 抛出，main.py 的异常处理器统一转成 400（覆盖默认 422）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import settings


class TTSRequest(BaseModel):
    """``POST /api/tts`` 请求体。"""

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
