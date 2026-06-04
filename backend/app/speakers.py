"""音色（speaker）管理。

ChatTTS 的特色能力是**随机音色采样**：``chat.sample_random_speaker()`` 从模型的
说话人分布中采样一个 embedding。本模块在其之上提供需求文档要求的三件事：

1. **可复现**：以整数 ``seed`` 作为音色 id，相同 seed → 相同 embedding。实现方式是
   采样前用 ``torch.manual_seed(seed)`` 固定全局随机数状态。
2. **缓存**：进程内缓存 ``seed -> embedding``，避免重复计算（线程安全）。
3. **命名清单**：预置一组固定 seed 作为下拉可选音色，供前端展示与试听。

音色 id 与 seed 一一对应，因此 ``id`` 直接采用 seed 的十进制字符串形式。
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass
from typing import Callable

from .config import settings

# 采样器类型：无参、返回一个 speaker embedding 字符串（ChatTTS 的编码格式）。
SpeakerSampler = Callable[[], str]

# 预置命名音色的 seed 列表。ChatTTS 没有官方固定音色，这里挑选一组稳定 seed
# 作为下拉默认选项，方便用户直接试听挑选；用户也可通过“随机音色”探索更多。
_CURATED_SEEDS: tuple[int, ...] = (2, 7, 21, 42, 111, 333, 1024, 2048)

# 随机 seed 的取值上限（32 位正整数范围），保证可被 torch.manual_seed 接受。
_MAX_SEED = 2**31 - 1


@dataclass(frozen=True)
class SpeakerListEntry:
    """供 `/api/speakers` 返回的最小音色项。"""

    id: str
    seed: int
    display_name: str


def _display_name(seed: int) -> str:
    """根据 seed 生成人类可读的展示名。"""

    return f"音色 #{seed}"


class SpeakerRegistry:
    """音色注册表：负责 seed→embedding 的采样、缓存与清单管理。

    引擎在加载完成后通过 ``bind()`` 注入真正的采样器，避免本模块直接依赖 ChatTTS，
    保持与 ``tts_engine`` 的单向依赖关系（engine -> speakers）。
    """

    def __init__(self, config=settings) -> None:
        self.config = config
        self._lock = threading.Lock()
        # seed -> embedding 缓存；embedding 计算较贵，命中后直接复用。
        self._cache: dict[int, str] = {}
        self._sampler: SpeakerSampler | None = None

    def bind(self, sampler: SpeakerSampler) -> None:
        """绑定底层随机音色采样器（由引擎在模型加载后调用）。"""

        self._sampler = sampler

    @property
    def is_ready(self) -> bool:
        """采样器是否已绑定（即模型是否已加载）。"""

        return self._sampler is not None

    def embedding_for_seed(self, seed: int) -> str:
        """返回给定 seed 对应的 speaker embedding（带缓存，保证可复现）。

        采样前用 ``torch.manual_seed(seed)`` 固定全局 RNG，使 ``sample_random_speaker()``
        的输出只由 seed 决定；**采样完成后立即把之前的 RNG 状态还原**，避免污染同进程
        内后续的推理随机性（GPT 采样等也依赖全局 RNG，曾经造成“相同请求第一次/第二次
        结果飘”或“合成结果与音色 seed 强相关”等不易复现的现象）。

        Raises:
            RuntimeError: 采样器尚未绑定（模型未加载）。
        """

        if self._sampler is None:
            raise RuntimeError("音色采样器尚未就绪（模型未加载）")

        # 双检锁：命中缓存直接返回；否则在锁内采样并写缓存，避免并发重复计算。
        cached = self._cache.get(seed)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._cache.get(seed)
            if cached is not None:
                return cached

            import torch

            # 备份全局 RNG（CPU + 所有 CUDA 设备），采样后还原。
            cpu_state = torch.random.get_rng_state()
            cuda_states = (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            )
            try:
                torch.manual_seed(seed)
                embedding = self._sampler()
            finally:
                torch.random.set_rng_state(cpu_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)

            self._cache[seed] = embedding
            return embedding

    def sample_new(self) -> tuple[int, str]:
        """采样一个全新的随机音色，返回 ``(seed, embedding)``。

        seed 本身随机生成（供用户记录后复现），embedding 通过 ``embedding_for_seed``
        计算并入缓存。
        """

        seed = random.randint(1, _MAX_SEED)
        embedding = self.embedding_for_seed(seed)
        return seed, embedding

    def default_seed(self) -> int:
        """默认音色 seed（来自配置）。"""

        return self.config.default_speaker

    def list_speakers(self) -> list["SpeakerListEntry"]:
        """返回预置命名音色清单（含默认音色，去重并保持顺序）。

        仅返回元数据，不需要模型已加载——因此前端挂载时即可填充下拉框。
        """

        seeds: list[int] = []
        for seed in (self.default_seed(), *_CURATED_SEEDS):
            if seed not in seeds:
                seeds.append(seed)

        return [
            SpeakerListEntry(
                id=str(seed),
                seed=seed,
                display_name=_display_name(seed)
                + ("（默认）" if seed == self.default_seed() else ""),
            )
            for seed in seeds
        ]


def parse_speaker(speaker: str | None) -> int:
    """把请求里的音色标识解析为整数 seed。

    - ``None`` / 空串：回退到配置默认音色 seed；
    - 纯数字字符串：解析为 seed；
    - 其它：抛出 ``ValueError``，由路由层映射为 400。

    Raises:
        ValueError: 音色标识不是合法整数 seed。
    """

    if speaker is None or not str(speaker).strip():
        return settings.default_speaker

    text = str(speaker).strip()
    try:
        seed = int(text)
    except ValueError as exc:
        raise ValueError(f"非法音色标识: {speaker}（应为整数 seed）") from exc

    if seed < 0 or seed > _MAX_SEED:
        raise ValueError(f"音色 seed 超出范围 0~{_MAX_SEED}: {seed}")
    return seed


# 全局单例：与 ``engine`` 配合使用。
speaker_registry = SpeakerRegistry()
