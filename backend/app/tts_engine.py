"""ChatTTS 核心引擎封装。

本模块把官方 ``ChatTTS`` 包封装成服务可用的引擎，负责：

- 懒加载（线程安全双检锁）一次性构建 ``ChatTTS.Chat`` 实例，并探测 GPU/CPU；
- 文本归一化、长文本切分（按中文标点 / 软断点 / 字数阈值，合并过短段）；
- 逐段调用 ``chat.infer``（带 speaker embedding + 采样参数），段间插入静音;
- numpy 线性插值变速（不引入 ffmpeg/scipy）；
- soundfile 编码 24kHz 单声道 WAV 字节。

不 vendoring 模型源码：``ChatTTS`` 作为 PyPI 依赖在运行时延迟导入。
"""

from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import Settings, settings
from .speakers import parse_speaker, speaker_registry
from .text_preprocess import (
    AtomicSpan,
    PreprocessOptions,
    PreprocessResult,
    preprocess_text,
)

logger = logging.getLogger("chattts.engine")


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
class SegmentPlan:
    """``plan_segments`` 的结果：把「可在推理前完成」的产物打包在一起。

    既供普通 / 流式合成路径消费（``seed``/``speed``/``segments``/``refine_prompt``），
    也供预览接口回显（``result`` 携带 ``normalized_text``/``changes``/``refine_mode``）。
    """

    seed: int
    speed: float
    segments: list[str]
    refine_prompt: str | None
    result: PreprocessResult


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

            logger.info(
                "ChatTTS 已加载：device=%s source=%s 耗时=%.1fs default_speaker=%d",
                self._device,
                load_kwargs.get("source"),
                elapsed,
                self.config.default_speaker,
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
        options: PreprocessOptions | None = None,
    ) -> SynthesisResult:
        """把文本合成为完整音频样本。

        Args:
            text: 待合成文本。
            speaker: 音色 id（seed 字符串），缺省用默认音色。
            speed: 语速，缺省用配置默认值。
            refine: 是否启用 ChatTTS 文本 refine。
            temperature/top_p/top_k: 采样参数。
            options: 文本预处理选项；缺省按 ``refine`` 构造默认 ``balanced/natural``。

        Returns:
            ``SynthesisResult``（float32 单声道样本 + 采样率）。

        实现上复用与流式接口相同的 ``plan_segments`` + ``synthesize_segment``，因此
        普通接口与流式接口的逐段合成、变速、段间静音语义完全一致——这正是“流式下载
        得到的完整 WAV 与普通接口输出在听感上一致”的前提（见需求 9.3/9.9）。
        普通接口不启用双阈值（首片/后续片用同一较大目标长度，见 6.8）。

        Raises:
            ValueError: 文本为空、参数非法或音色非法。
            SynthesisError: 推理阶段失败。
            ModelAssetError: 模型不可用。
        """

        plan = self.plan_segments(
            text,
            speaker=speaker,
            speed=speed,
            options=options if options is not None else PreprocessOptions(refine=refine),
        )
        self.load()

        total = len(plan.segments)
        sample_rate = self.config.sample_rate
        pieces: list[np.ndarray] = []
        started = time.monotonic()
        for index, segment in enumerate(plan.segments):
            result = self.synthesize_segment(
                segment,
                index,
                total,
                seed=plan.seed,
                speed=plan.speed,
                refine=refine,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                refine_prompt=plan.refine_prompt,
            )
            sample_rate = result.sample_rate
            # ``synthesize_segment`` 已把“非末尾分段的段间静音”并入分段尾部，
            # 因此这里顺序拼接即可，无需再额外插入 ``_silence``。
            pieces.append(result.samples)

        if not pieces:
            raise SynthesisError("未生成任何音频片段")

        merged = np.concatenate(pieces)
        elapsed = time.monotonic() - started
        logger.info(
            "合成完成：chars=%d segments=%d speaker=%d speed=%s refine=%s refine_mode=%s 耗时=%.2fs",
            len(text or ""),
            total,
            plan.seed,
            plan.speed,
            refine,
            plan.result.refine_mode,
            elapsed,
        )
        return SynthesisResult(samples=merged, sample_rate=sample_rate)

    def plan_segments(
        self,
        text: str,
        *,
        speaker: str | None = None,
        speed: float | None = None,
        options: PreprocessOptions | None = None,
        first_chars: int | None = None,
        rest_chars: int | None = None,
    ) -> SegmentPlan:
        """流式前置：做参数校验、文本预处理与分片，但**不触发模型推理**。

        把“可在流开始前完成的校验”集中在这里，供流式路由在返回 200 之前调用，从而把
        文本为空、语速越界、音色非法等问题用标准 HTTP 400 反馈，而不是变成 SSE
        ``error`` 事件（见需求 5）。

        管线（落实 5.2 / 7.5）：``preprocess_text(text, options)`` → ``segment_text(...)``。
        预处理与原子保护在普通 / 流式接口完全一致，仅长度阈值不同（D3）。

        Args:
            options: 预处理选项；缺省 ``PreprocessOptions()``（balanced/natural/不 refine）。
            first_chars: 首片目标上限。缺省回退到 ``rest_chars``（普通接口即不启用双阈值）。
            rest_chars: 后续分片目标上限。缺省用 ``MAX_SEGMENT_CHARS``（普通接口）。
                流式路由分别传入 ``STREAM_FIRST_SEGMENT_CHARS`` / ``STREAM_SEGMENT_CHARS``。

        Returns:
            ``SegmentPlan``：音色 seed、语速、分段列表、refine_prompt 与完整预处理结果。

        Raises:
            ValueError: 文本为空、语速越界或音色标识非法。
        """

        options = options if options is not None else PreprocessOptions()
        result = preprocess_text(text, options)
        if not result.text.strip():
            raise ValueError("文本不能为空")

        # parse_speaker 对非整数 / 越界 seed 抛 ValueError，由路由层映射为 400。
        seed = parse_speaker(speaker)
        selected_speed = self._validate_speed(
            speed if speed is not None else self.config.default_speed
        )

        rest = rest_chars or self.config.max_segment_chars
        first = first_chars or rest
        # 首片长度微调（仅双阈值流式生效，见 6.8 第 5 条）：
        # - expressive profile：首片略短、出声更快、保留更多停顿；
        # - flat prosody：首片略长、减少段间割裂。
        if first < rest:
            if options.profile == "expressive":
                first = max(10, int(first * 0.8))
            elif options.prosody == "flat":
                first = min(rest, int(first * 1.2))
        # 仅当首片目标显著小于后续片（即流式双阈值）时才启用「首片硬上限」；
        # 普通接口 first==rest，硬上限等于 rest，相当于不特殊处理首片。
        first_cap = self.config.stream_first_hard_cap if first < rest else rest

        segments = segment_text(
            result.text,
            result.atomic_spans,
            first_chars=first,
            rest_chars=rest,
            first_hard_cap=first_cap,
        )
        if not segments:
            raise ValueError("文本不能为空")

        # 可观测性（6.10）：只记录长度、分段数、命中的规则类型计数、refine_mode，
        # **不记录完整文本**，兼顾隐私与按真实样本迭代规则的需要。
        if logger.isEnabledFor(logging.INFO):
            hit_counts: dict[str, int] = {}
            for change in result.changes:
                hit_counts[change.type] = hit_counts.get(change.type, 0) + 1
            logger.info(
                "preprocess: chars_in=%d chars_out=%d segments=%d refine_mode=%s rules=%s",
                len(text or ""),
                len(result.text),
                len(segments),
                result.refine_mode,
                hit_counts,
            )

        return SegmentPlan(
            seed=seed,
            speed=selected_speed,
            segments=segments,
            refine_prompt=result.refine_prompt,
            result=result,
        )

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
        refine_prompt: str | None = None,
    ) -> SegmentSynthesis:
        """合成单个分段，并按“段间静音一致性”要求处理尾部静音。

        这是流式接口的最小同步单元：路由层会用 ``asyncio.to_thread`` 调用本方法，避免
        阻塞事件循环。``seed``/``speed`` 应来自 ``plan_segments`` 的解析结果。

        ``refine_prompt``（落实 D1）：仅当 ``refine=True`` 时用它构造 ``RefineTextParams``
        交给 ChatTTS 由 refine_text 全权负责停顿；缺省回退到历史默认
        ``[oral_2][laugh_0][break_4]``。``refine=False`` 路径下不注入任何手工停顿 token，
        停顿完全交由文本自身标点与 ChatTTS 决定，此处不走 refine。

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
        refine_params = (
            self._build_refine_params(refine_prompt) if refine else None
        )

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
        options: PreprocessOptions | None = None,
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
            options=options,
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

    def _build_refine_params(self, prompt: str | None = None):
        """构造 ``RefineTextParams``。

        ``prompt`` 来自 ``prosody`` 配置化映射（见 6.7）；缺省保留历史默认
        ``[oral_2][laugh_0][break_4]``，保证未传 prosody 时行为不变。
        """

        import ChatTTS

        return ChatTTS.Chat.RefineTextParams(
            prompt=prompt or "[oral_2][laugh_0][break_4]"
        )

    # ------------------------------------------------------------------ #
    # 内部：文本与校验
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_speed(speed: float) -> float:
        """校验语速范围 0.5~2.0。

        Schema 已通过 Pydantic 约束做过一次校验；这里再做一次是因为 ``synthesize``
        也被脚本/测试直接调用，绕过了 schema。属于有意冗余的纵深防御。
        """

        value = float(speed)
        if value < 0.5 or value > 2.0:
            raise ValueError("speed 必须在 0.5 到 2.0 之间")
        return value


# ---------------------------------------------------------------------- #
# 模块级工具函数
# ---------------------------------------------------------------------- #
# ChatTTS custom 加载所需的关键资产（任一格式存在即可）。
_REQUIRED_ASSETS: tuple[str, ...] = ("Decoder", "DVAE", "GPT", "Vocos")


def _model_dir_ready(model_dir: Path) -> bool:
    """判断本地模型目录是否可用于 ``source="custom"`` 加载。

    ChatTTS 自定义路径要求 ``asset/`` 子目录下包含 GPT / DVAE / Decoder / Vocos
    等关键权重文件（任一以 ``.pt`` 或 ``.safetensors`` 结尾的格式都接受）。
    只检查“asset 目录非空”过于宽松，残缺挂载会让加载阶段抛出难追溯的错误。
    """

    asset_dir = model_dir / "asset"
    if not asset_dir.is_dir():
        return False
    for name in _REQUIRED_ASSETS:
        if not (
            (asset_dir / f"{name}.pt").exists()
            or (asset_dir / f"{name}.safetensors").exists()
        ):
            return False
    return True


# 句末硬断点（强边界）与软断点（弱边界）字符集。同时认全角与半角，因为预处理不强制
# 把英文标点转中文。注意：**不再**在分片入口二次压缩空白（D2/6.8）——结构标记此时已
# token 化，二次 ``re.sub(r"\s+"," ")`` 会破坏它们。
_HARD_BREAK_CHARS = frozenset("。！？!?；;：:")
_SOFT_BREAK_CHARS = frozenset("，,、 \t")


def segment_text(
    text: str,
    atomic_spans: list[AtomicSpan] | None = None,
    *,
    first_chars: int,
    rest_chars: int,
    first_hard_cap: int | None = None,
) -> list[str]:
    """token / 原子-aware 的分片器（落实 D3 / 6.8）。

    输入为**预处理后的可朗读文本**及其原子片段标注。核心约束：任何层级（含受约束硬切）
    都不得切开一个 ``AtomicSpan``（ChatTTS token、日期、时间、金额、百分比、单位、英文
    缩写、URL 占位符、长编号等）。

    切分优先级（6.8）：句末硬断点 → 软断点（逗号 / 顿号 / 空格）→ 受约束硬切（在不破坏
    原子片段前提下尽量靠近目标上限）。``[lbreak]`` 结构标记作为强制分片边界，让段落 /
    列表项优先成片。

    双阈值（6.8）：首片用 ``first_chars``（求快），后续片用 ``rest_chars``（求连贯）；首片
    若无法在自然断点内结束，可延长到 ``first_hard_cap``。普通接口令 ``first==rest`` 即退化为
    单阈值。

    Args:
        text: 预处理后的可朗读文本。
        atomic_spans: 原子片段区间列表。
        first_chars: 首片目标上限。
        rest_chars: 后续分片目标上限。
        first_hard_cap: 首片硬上限；缺省取 ``rest_chars``。

    Returns:
        分段后的文本列表（每段已 strip，丢弃空段）。
    """

    if not text or not text.strip():
        return []
    spans = atomic_spans or []
    first_hard_cap = first_hard_cap or rest_chars
    length = len(text)

    # 位于原子片段「内部」的切点（严格在 (start, end) 之间）一律禁止切分。
    forbidden: set[int] = set()
    for span in spans:
        forbidden.update(range(span.start + 1, span.end))
    # [lbreak] 结构标记的结束位置：作为强制分片边界。
    lbreak_ends = {
        span.end for span in spans if text[span.start:span.end] == "[lbreak]"
    }

    # 1) 先在所有「自然断点」处切出最细原子片段（hard / soft / lbreak 边界）。
    #    禁止切点（原子片段内部）即便恰为断点字符也跳过——这正是「不切开缩写里的空格」。
    cut_points = [0]
    for i, ch in enumerate(text):
        cut = i + 1
        if cut in forbidden:
            continue
        if ch in _HARD_BREAK_CHARS or ch in _SOFT_BREAK_CHARS or cut in lbreak_ends:
            cut_points.append(cut)
    if cut_points[-1] != length:
        cut_points.append(length)

    # 原子片段三元组：(start, end, 是否 lbreak 强制边界)；空白原子并入前段、此处先跳过。
    atoms: list[list] = []
    for start, end in zip(cut_points, cut_points[1:]):
        if not text[start:end].strip():
            continue
        atoms.append([start, end, end in lbreak_ends])

    # 2) 仍超过 rest_chars 的原子（无自然断点的长串）→ 受约束硬切（不破坏原子片段）。
    ranges: list[list] = []
    for start, end, is_lbreak in atoms:
        wrapped = _constrained_wrap(start, end, rest_chars, forbidden)
        for sub_index, (sub_start, sub_end) in enumerate(wrapped):
            # 仅把 lbreak 边界标记保留在该原子的最后一个子片段上。
            ranges.append([sub_start, sub_end, is_lbreak and sub_index == len(wrapped) - 1])

    if not ranges:
        return []

    # 双阈值下：若首个 range 自身就超过首片硬上限，对它单独按硬上限再细切。
    if first_chars < rest_chars and (ranges[0][1] - ranges[0][0]) > first_hard_cap:
        head = ranges.pop(0)
        rewrapped = _constrained_wrap(head[0], head[1], first_hard_cap, forbidden)
        injected = [[s, e, False] for s, e in rewrapped]
        if injected:
            injected[-1][2] = head[2]
        ranges = injected + ranges

    # 3) 贪婪打包：首片目标 first_chars、后续片目标 rest_chars；[lbreak] 强制成段边界。
    segments: list[str] = []
    cur_start: int | None = None
    cur_end = 0

    def _flush() -> None:
        nonlocal cur_start, cur_end
        if cur_start is not None:
            piece = text[cur_start:cur_end].strip()
            if piece:
                segments.append(piece)
        cur_start = None

    for start, end, is_lbreak in ranges:
        target = first_chars if not segments else rest_chars
        if cur_start is None:
            cur_start, cur_end = start, end
        elif (end - cur_start) <= target:
            cur_end = end
        else:
            _flush()
            cur_start, cur_end = start, end
        if is_lbreak:
            _flush()
    _flush()
    return segments


def _constrained_wrap(
    start: int, end: int, limit: int, forbidden: set[int]
) -> list[tuple[int, int]]:
    """受约束硬切：把 ``[start, end)`` 切成若干 ≤ ``limit`` 的子区间，且不切开原子片段。

    优先在 ``start + limit`` 处切；该处若落在原子片段内部（禁止切点），先向左回退寻找
    合法切点，再不行则向右延伸到第一个合法切点（此时子片段会略超 ``limit``，这是为
    保护语义单元而允许的——见 6.1「允许为保护语义单元略微超出」）。
    """

    if end - start <= limit:
        return [(start, end)]
    pieces: list[tuple[int, int]] = []
    pos = start
    while end - pos > limit:
        cut = pos + limit
        # 向左回退到合法切点。
        back = cut
        while back > pos and back in forbidden:
            back -= 1
        if back > pos:
            cut = back
        else:
            # 左侧全被原子片段占据：向右延伸到第一个合法切点。
            cut = pos + limit
            while cut < end and cut in forbidden:
                cut += 1
            if cut >= end:
                break
        pieces.append((pos, cut))
        pos = cut
    if pos < end:
        pieces.append((pos, end))
    return pieces


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


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    """统一音频数组为 float32 单声道一维数组。

    ChatTTS ``infer`` 返回一维 PCM 或 ``(1, T)`` 形状；``np.squeeze`` 会去掉单维。
    squeeze 后仍 ndim==2 的情况理论上不应出现（约定为 ``(channels, T)``），但为
    保守起见仍取声道维均值。零维标量被还原为长度 1 的数组。
    """

    array = np.asarray(samples, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 2:
        # 显式约定：ChatTTS 输出多通道时形状为 (channels, T)。
        array = array.mean(axis=0)
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
    if source.size == 0:
        # 空音频不做插值，避免 np.interp 抛错。
        return source
    target_length = max(1, int(round(len(source) / speed)))
    if len(source) <= 1 or target_length == len(source):
        return source

    source_positions = np.arange(len(source), dtype=np.float32)
    target_positions = np.linspace(
        0, len(source) - 1, num=target_length, dtype=np.float32
    )
    return np.interp(target_positions, source_positions, source).astype(np.float32)


engine = ChatTTSEngine()
