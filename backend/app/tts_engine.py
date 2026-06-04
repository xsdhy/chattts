"""ChatTTS 核心引擎封装。

本模块把官方 ``ChatTTS`` 包封装成服务可用的引擎，负责：

- 懒加载（线程安全双检锁）一次性构建 ``ChatTTS.Chat`` 实例，并探测 GPU/CPU；
- 文本归一化、长文本切分（按中文标点 / 软断点 / 字数阈值，合并过短段）；
- 逐段调用 ``chat.infer``（带 speaker embedding + 采样参数），段间插入静音；
- numpy 线性插值变速（不引入 ffmpeg/scipy）；
- soundfile 编码 24kHz 单声道 WAV 字节。

不 vendoring 模型源码：``ChatTTS`` 作为 PyPI 依赖在运行时延迟导入。
"""

from __future__ import annotations

import io
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import soundfile as sf

from .config import Settings, settings
from .speakers import parse_speaker, speaker_registry


# ---- 异常层级（见需求 5.4）----
class TTSEngineError(RuntimeError):
    """TTS 引擎错误基类。"""


class ModelAssetError(TTSEngineError):
    """模型资产缺失或加载失败。"""


class SynthesisError(TTSEngineError):
    """合成阶段失败。"""


@dataclass(frozen=True)
class SynthesisResult:
    """合成结果：float32 单声道样本 + 采样率。"""

    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class SegmentSynthesis:
    """单个分段的逐段合成结果，供流式接口逐段产出。

    与 ``SynthesisResult`` 的区别在于它携带分段在整体中的位置信息
    （``index``/``total``/``is_final``），且 ``samples`` 已按“段间静音一致性”要求
    处理过：非末尾分段尾部已并入段间静音，末尾分段不追加静音。
    """

    index: int
    total: int
    samples: np.ndarray
    sample_rate: int
    is_final: bool


class ChatTTSEngine:
    """ChatTTS 引擎封装。

    ``ChatTTS.Chat`` 初始化很重（加载 GPT / DVAE / Vocos 等多个子模型），因此用
    懒加载 + 锁保证一个进程内只构造一次。FastAPI 在 ``lifespan`` 里显式调用
    ``load()`` 预热；首请求也可触发加载。
    """

    # 句末标点（硬断点）与软断点（逗号 / 顿号 / 空白）。
    _sentence_break_re = re.compile(r"([。！？!?；;：:\n]+)")
    _soft_break_re = re.compile(r"([，,、\s]+)")

    def __init__(self, config: Settings = settings) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._loaded = False
        self._chat = None  # ChatTTS.Chat 实例
        self._device = config.device

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载完成。"""

        return self._loaded

    @property
    def device(self) -> str:
        """当前推理设备（"cuda" / "cpu"）。"""

        return self._device

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """加载 ChatTTS 模型（双检锁，避免并发重复加载）。

        加载完成后：
        1. 把 ``chat.sample_random_speaker`` 绑定给音色注册表；
        2. 预生成默认音色 embedding，保证无音色请求输出稳定。

        Raises:
            ModelAssetError: 模型资产缺失或底层加载失败。
        """

        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            try:
                import torch

                import ChatTTS
            except Exception as exc:  # noqa: BLE001 - 依赖缺失给出明确提示
                raise ModelAssetError(
                    "未能导入 ChatTTS / torch，请先安装依赖：pip install -r backend/requirements.txt"
                ) from exc

            device = torch.device(self._device)
            load_kwargs = self._resolve_load_kwargs(device)

            started = time.monotonic()
            chat = ChatTTS.Chat()
            try:
                ok = chat.load(**load_kwargs)
            except Exception as exc:  # noqa: BLE001 - 包装底层错误
                raise ModelAssetError(f"ChatTTS 模型加载失败: {exc}") from exc
            if ok is False:
                raise ModelAssetError(
                    f"ChatTTS 模型加载失败（source={load_kwargs.get('source')}，"
                    f"model_dir={self.config.model_dir}）"
                )

            self._chat = chat
            self._loaded = True
            elapsed = time.monotonic() - started

            # 绑定采样器并预热默认音色，保证后续无音色请求复用稳定 embedding。
            speaker_registry.bind(chat.sample_random_speaker)
            speaker_registry.embedding_for_seed(self.config.default_speaker)

            print(
                f"[engine] ChatTTS 已加载：device={self._device} "
                f"source={load_kwargs.get('source')} 耗时={elapsed:.1f}s "
                f"default_speaker={self.config.default_speaker}",
                flush=True,
            )

    def _resolve_load_kwargs(self, device) -> dict:
        """决定 ``chat.load`` 的参数。

        优先使用本地模型目录（``source="custom"``）以实现运行时零网络依赖；本地
        缺失时按 ``MODEL_SOURCE`` 回退到 HuggingFace 自动下载（需允许联网）。

        Raises:
            ModelAssetError: 本地缺模型且不允许联网下载。
        """

        common = dict(device=device, compile=False)
        if _model_dir_ready(self.config.model_dir):
            return dict(
                source="custom", custom_path=str(self.config.model_dir), **common
            )

        # 本地不可用：是否允许从 HuggingFace 下载。
        if self.config.model_source == "huggingface" and self.config.models_auto_download:
            return dict(source="huggingface", **common)

        raise ModelAssetError(
            f"模型资产缺失：{self.config.model_dir} 下未找到 ChatTTS 权重，"
            "且未开启自动下载。请运行 scripts/fetch_models.sh 或设置 "
            "MODELS_AUTO_DOWNLOAD=1。"
        )

    # ------------------------------------------------------------------ #
    # 合成
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        text: str,
        *,
        speaker: str | None = None,
        speed: float | None = None,
        refine: bool = False,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20,
    ) -> SynthesisResult:
        """把文本合成为完整音频样本。

        Args:
            text: 待合成文本。
            speaker: 音色 id（seed 字符串），缺省用默认音色。
            speed: 语速，缺省用配置默认值。
            refine: 是否启用 ChatTTS 文本 refine。
            temperature/top_p/top_k: 采样参数。

        Returns:
            ``SynthesisResult``（float32 单声道样本 + 采样率）。

        实现上复用与流式接口相同的 ``plan_segments`` + ``synthesize_segment``，因此
        普通接口与流式接口的逐段合成、变速、段间静音语义完全一致——这正是“流式下载
        得到的完整 WAV 与普通接口输出在听感上一致”的前提（见需求 9.3/9.9）。

        Raises:
            ValueError: 文本为空、参数非法或音色非法。
            SynthesisError: 推理阶段失败。
            ModelAssetError: 模型不可用。
        """

        seed, selected_speed, segments = self.plan_segments(
            text, speaker=speaker, speed=speed
        )
        self.load()

        total = len(segments)
        sample_rate = self.config.sample_rate
        pieces: list[np.ndarray] = []
        started = time.monotonic()
        for index, segment in enumerate(segments):
            result = self.synthesize_segment(
                segment,
                index,
                total,
                seed=seed,
                speed=selected_speed,
                refine=refine,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            sample_rate = result.sample_rate
            # ``synthesize_segment`` 已把“非末尾分段的段间静音”并入分段尾部，
            # 因此这里顺序拼接即可，无需再额外插入 ``_silence``。
            pieces.append(result.samples)

        if not pieces:
            raise SynthesisError("未生成任何音频片段")

        merged = np.concatenate(pieces)
        elapsed = time.monotonic() - started
        print(
            f"[engine] 合成完成：chars={len(text or '')} segments={total} "
            f"speaker={seed} speed={selected_speed} refine={refine} "
            f"耗时={elapsed:.2f}s",
            flush=True,
        )
        return SynthesisResult(samples=merged, sample_rate=sample_rate)

    def plan_segments(
        self,
        text: str,
        *,
        speaker: str | None = None,
        speed: float | None = None,
        max_chars: int | None = None,
    ) -> tuple[int, float, list[str]]:
        """流式前置：做参数校验与文本切分，但**不触发模型推理**。

        把“可在流开始前完成的校验”集中在这里，供流式路由在返回 200 之前调用，从而把
        文本为空、语速越界、音色非法等问题用标准 HTTP 400 反馈，而不是变成 SSE
        ``error`` 事件（见需求 5）。

        Args:
            max_chars: 分段上限（同时是短段合并的目标长度）。缺省用
                ``MAX_SEGMENT_CHARS``（普通接口）；流式路由传入更小的
                ``STREAM_MAX_SEGMENT_CHARS`` 以获得更多、更小的分片，让首片更快到达、
                全程更流畅。

        Returns:
            ``(seed, selected_speed, segments)``：解析后的音色 seed、语速与切分后的
            文本分段列表。

        Raises:
            ValueError: 文本为空、语速越界或音色标识非法。
        """

        normalized = self._normalize_text(text)
        if not normalized:
            raise ValueError("文本不能为空")

        # parse_speaker 对非整数 / 越界 seed 抛 ValueError，由路由层映射为 400。
        seed = parse_speaker(speaker)
        selected_speed = self._validate_speed(
            speed if speed is not None else self.config.default_speed
        )
        segments = split_text(
            normalized, max_chars=max_chars or self.config.max_segment_chars
        )
        if not segments:
            raise ValueError("文本不能为空")

        return seed, selected_speed, segments

    def synthesize_segment(
        self,
        segment: str,
        index: int,
        total: int,
        *,
        seed: int,
        speed: float,
        refine: bool = False,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20,
    ) -> SegmentSynthesis:
        """合成单个分段，并按“段间静音一致性”要求处理尾部静音。

        这是流式接口的最小同步单元：路由层会用 ``asyncio.to_thread`` 调用本方法，避免
        阻塞事件循环。``seed``/``speed`` 应来自 ``plan_segments`` 的解析结果。

        段间静音策略（见需求 4.3）：
        - **非末尾**分段（``is_final=False``）在其音频尾部并入一段段间静音；
        - **末尾**分段（``is_final=True``）不追加尾部静音。

        这样前端把各分片解码为 PCM 后顺序拼接，结果与普通接口逐段拼接一致。
        """

        self.load()

        is_final = index == total - 1
        spk_emb = speaker_registry.embedding_for_seed(seed)
        infer_params = self._build_infer_params(
            spk_emb=spk_emb,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        refine_params = self._build_refine_params() if refine else None

        piece = self._infer_segment(
            segment,
            refine=refine,
            infer_params=infer_params,
            refine_params=refine_params,
        )

        sample_rate = self.config.sample_rate
        piece = _change_speed(piece, speed)
        if not is_final:
            piece = np.concatenate(
                [piece, _silence(sample_rate, self.config.silence_ms_between_segments)]
            )

        return SegmentSynthesis(
            index=index,
            total=total,
            samples=piece,
            sample_rate=sample_rate,
            is_final=is_final,
        )

    def synthesize_segments(
        self,
        text: str,
        *,
        speaker: str | None = None,
        speed: float | None = None,
        refine: bool = False,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20,
    ) -> Iterator[SegmentSynthesis]:
        """逐段产出生成器：每段产出一个 ``SegmentSynthesis``。

        复用 ``plan_segments``（校验+切分）与 ``synthesize_segment``（单段合成+段间
        静音）。这是一个**同步生成器**，每次 ``next()`` 只合成一个分段，因此路由层可以
        把每段推理放到线程池执行，并在两段之间让出控制权及时 flush SSE 事件。

        路由层目前直接使用 ``plan_segments`` + ``synthesize_segment`` 以便在每段前后
        插入进度事件与断开检测；本生成器作为等价的可复用封装一并提供。它与流式路由
        保持一致，使用更小的 ``STREAM_MAX_SEGMENT_CHARS`` 切分。
        """

        seed, selected_speed, segments = self.plan_segments(
            text,
            speaker=speaker,
            speed=speed,
            max_chars=self.config.stream_max_segment_chars,
        )
        self.load()

        total = len(segments)
        for index, segment in enumerate(segments):
            yield self.synthesize_segment(
                segment,
                index,
                total,
                seed=seed,
                speed=selected_speed,
                refine=refine,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

    def synthesize_wav_bytes(
        self,
        text: str,
        *,
        speaker: str | None = None,
        speed: float | None = None,
        refine: bool = False,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20,
    ) -> bytes:
        """合成并编码为 WAV 字节，供 API 直接返回。"""

        result = self.synthesize(
            text,
            speaker=speaker,
            speed=speed,
            refine=refine,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return encode_wav(result.samples, result.sample_rate)

    # ------------------------------------------------------------------ #
    # 内部：单段推理与参数构造
    # ------------------------------------------------------------------ #
    def _infer_segment(
        self,
        segment: str,
        *,
        refine: bool,
        infer_params,
        refine_params,
    ) -> np.ndarray:
        """对单个文本分段调用 ChatTTS 推理，返回 float32 单声道样本。"""

        if self._chat is None:
            raise TTSEngineError("模型尚未加载")

        try:
            wavs = self._chat.infer(
                [segment],
                skip_refine_text=not refine,
                params_refine_text=refine_params,
                params_infer_code=infer_params,
            )
        except Exception as exc:  # noqa: BLE001 - 包装为业务异常
            raise SynthesisError(
                f"文本分段合成失败: {segment[:30]}"
            ) from exc

        if not wavs:
            raise SynthesisError(f"文本分段未产生音频: {segment[:30]}")
        return _to_mono_float32(wavs[0])

    def _build_infer_params(
        self, *, spk_emb: str, temperature: float, top_p: float, top_k: int
    ):
        """构造 ``ChatTTS.Chat.InferCodeParams``（注意字段名 top_P / top_K）。"""

        import ChatTTS

        return ChatTTS.Chat.InferCodeParams(
            spk_emb=spk_emb,
            temperature=temperature,
            top_P=top_p,
            top_K=top_k,
        )

    def _build_refine_params(self):
        """构造默认的 ``RefineTextParams``（口语化 + 适度停顿）。"""

        import ChatTTS

        return ChatTTS.Chat.RefineTextParams(prompt="[oral_2][laugh_0][break_4]")

    # ------------------------------------------------------------------ #
    # 内部：文本与校验
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_text(text: str) -> str:
        """最小文本归一化：合并连续空白，去首尾空白，保留标点语义。"""

        return re.sub(r"\s+", " ", text or "").strip()

    @staticmethod
    def _validate_speed(speed: float) -> float:
        """校验语速范围 0.5~2.0。"""

        value = float(speed)
        if value < 0.5 or value > 2.0:
            raise ValueError("speed 必须在 0.5 到 2.0 之间")
        return value


# ---------------------------------------------------------------------- #
# 模块级工具函数
# ---------------------------------------------------------------------- #
def _model_dir_ready(model_dir: Path) -> bool:
    """判断本地模型目录是否可用于 ``source="custom"`` 加载。

    ChatTTS 自定义路径要求目录下存在 ``asset/`` 子目录且非空（GPT/DVAE/Vocos 等
    权重都在其中）。这里只做轻量存在性探测，真正合法性交给 ``chat.load``。
    """

    asset_dir = model_dir / "asset"
    if not asset_dir.is_dir():
        return False
    return any(asset_dir.iterdir())


def split_text(text: str, *, max_chars: int = 120) -> list[str]:
    """按中文标点和长度阈值切分文本。

    策略分三层：
    1. 优先按句末标点切句（保留标点，朗读停顿更自然）；
    2. 单句过长时再按逗号 / 顿号 / 空白等软断点切；
    3. 仍过长则按固定长度硬切，确保单段不超过阈值。
    最后合并过短片段，减少推理次数。
    """

    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []

    sentence_chunks = _split_keep_delimiter(
        normalized, ChatTTSEngine._sentence_break_re
    )
    segments: list[str] = []
    for chunk in sentence_chunks:
        if len(chunk) <= max_chars:
            segments.append(chunk)
            continue
        for soft_chunk in _split_keep_delimiter(chunk, ChatTTSEngine._soft_break_re):
            if len(soft_chunk) <= max_chars:
                segments.append(soft_chunk)
            else:
                segments.extend(_hard_wrap(soft_chunk, max_chars=max_chars))

    return _merge_short_segments(segments, max_chars=max_chars)


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """把 float32 音频样本编码为 16bit PCM WAV 字节。"""

    buffer = io.BytesIO()
    sf.write(
        buffer,
        _to_mono_float32(samples),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return buffer.getvalue()


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    """写出 WAV 文件，供冒烟测试脚本使用。"""

    sf.write(
        str(path),
        _to_mono_float32(samples),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )


def _split_keep_delimiter(text: str, pattern: re.Pattern[str]) -> list[str]:
    """按正则切分并把分隔符拼回前一片段，避免丢失停顿信息。"""

    parts = pattern.split(text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part:
            continue
        current += part
        if pattern.fullmatch(part):
            chunks.append(current.strip())
            current = ""
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _hard_wrap(text: str, *, max_chars: int) -> list[str]:
    """没有合适标点时按固定长度切分。"""

    return [
        text[index : index + max_chars].strip()
        for index in range(0, len(text), max_chars)
        if text[index : index + max_chars].strip()
    ]


def _merge_short_segments(segments: Iterable[str], *, max_chars: int) -> list[str]:
    """合并过短片段，减少推理次数，同时保持长度上限。"""

    merged: list[str] = []
    current = ""
    for segment in (item.strip() for item in segments if item.strip()):
        candidate = f"{current}{segment}" if current else segment
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = segment
    if current:
        merged.append(current)
    return merged


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    """统一音频数组为 float32 单声道一维数组。

    ChatTTS 输出可能是 ``(T,)`` 或 ``(1, T)``；这里先 squeeze，再对仍是二维的情况
    按“较短轴为声道”启发式取均值，避免拼接时 dtype/shape 不一致。
    """

    array = np.asarray(samples, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 2:
        # 行数远小于列数 → 形如 (channels, T)，对声道维求均值；反之对列求均值。
        if array.shape[0] <= array.shape[1]:
            array = array.mean(axis=0)
        else:
            array = array.mean(axis=1)
    elif array.ndim == 0:
        array = array.reshape(1)
    return np.ascontiguousarray(array.astype(np.float32))


def _silence(sample_rate: int, duration_ms: int) -> np.ndarray:
    """生成段间短静音，让长文本拼接听起来更自然。"""

    length = max(0, int(sample_rate * duration_ms / 1000))
    return np.zeros(length, dtype=np.float32)


def _change_speed(samples: np.ndarray, speed: float) -> np.ndarray:
    """用线性插值做后处理变速（不引入 ffmpeg/scipy）。

    ``speed > 1`` 缩短音频、``speed < 1`` 拉长音频。线性插值音质不如专业 DSP，
    但对语速滑块足够稳定，且零额外依赖。
    """

    if abs(speed - 1.0) < 1e-6:
        return samples

    source = _to_mono_float32(samples)
    target_length = max(1, int(round(len(source) / speed)))
    if len(source) <= 1 or target_length == len(source):
        return source

    source_positions = np.arange(len(source), dtype=np.float32)
    target_positions = np.linspace(
        0, len(source) - 1, num=target_length, dtype=np.float32
    )
    return np.interp(target_positions, source_positions, source).astype(np.float32)


engine = ChatTTSEngine()
