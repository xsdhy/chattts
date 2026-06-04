"""推理并发控制。

ChatTTS 是 PyTorch 重模型，单次推理显存/算力占用大，过多请求同时进入会导致
显存 OOM 或延迟急剧上升。本模块用信号量限制同时推理数，并用等待超时作为过载
保护（排队超过阈值快速失败返回 503）。
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from .config import settings


class QueueTimeoutError(RuntimeError):
    """等待推理槽位超时。

    路由层捕获后映射为 HTTP 503 ``服务繁忙，请稍后重试``。
    """


@dataclass
class InferenceGate:
    """推理闸门。

    ``asyncio.Semaphore`` 只保护“进入推理区”的请求数量；真正的同步推理由路由层
    交给线程池（``asyncio.to_thread``）执行，避免阻塞 FastAPI 事件循环。
    """

    max_concurrency: int = settings.max_concurrency
    queue_timeout: float = settings.queue_timeout

    def __post_init__(self) -> None:
        # 信号量在事件循环就绪后才被真正使用，这里提前构造即可。
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    async def acquire(self) -> None:
        """显式获取一个推理槽位；等待超时抛出 ``QueueTimeoutError``。

        相比 ``slot()`` 上下文管理器，这里把“取”和“放”拆成两个独立动作，用于
        流式接口这种“在路由里取、在生成器 ``finally`` 里放”的跨作用域场景：

        - 流式路由必须在进入 SSE（返回 200）之前就拿到槽位，这样等待超时还能用
          标准 HTTP 503 反馈；
        - 真正的流式输出发生在 handler 返回之后、由服务器消费生成器时，因此释放
          动作必须延后到生成器结束（见 ``main.py`` 的 ``finally``）。

        若改用 ``async with slot()`` 包住 ``return StreamingResponse(...)``，handler
        一返回槽位就被释放，整段流式输出将不受限流保护，所以必须用本方法 + 显式
        ``release()``。
        """

        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.queue_timeout
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise QueueTimeoutError("服务繁忙，请稍后重试") from exc

    def release(self) -> None:
        """释放一个此前通过 ``acquire()`` 获取的推理槽位。

        必须与 ``acquire()`` 成对调用，且保证“正常结束、发生 error、客户端断开”
        三种路径下都会被调用，避免槽位泄漏导致并发数被逐渐耗尽。
        """

        self._semaphore.release()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """等待并占用一个推理槽位；排队超时抛出 ``QueueTimeoutError``。

        作用域结束（正常返回或异常）都会自动释放槽位，避免泄漏导致并发数被
        逐渐耗尽。供 ``POST /api/tts`` 包裹同步推理调用使用；内部复用
        ``acquire``/``release``，与流式接口共享同一套信号量与超时逻辑。
        """

        await self.acquire()
        try:
            yield
        finally:
            self.release()


inference_gate = InferenceGate()
