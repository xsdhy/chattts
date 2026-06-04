/**
 * 流式音频播放器：基于 Web Audio API 的按序、近无缝衔接播放。
 *
 * 设计（见需求 7.2 流式播放要求）：
 * - 用单个 `AudioContext` + 维护一个调度游标 `nextStartTime`，把每个分片包成
 *   `AudioBufferSourceNode` 并用 `start(when)` 排到上一段结束的时间点，从而尽量
 *   消除分片切换处的停顿。
 * - 分片必须按 `index` 顺序播放。后端保证按 index 顺序发送，这里再用一个待播缓冲
 *   按 `nextExpectedIndex` 做一次保险排序，避免任何时序导致乱序。
 * - 首个分片到达即起播，不等待 `done`。
 * - 看门狗：若 pending 中堆积了分片但 `nextExpectedIndex` 长时间未推进，触发 stuck
 *   回调，让上层主动中止流并提示用户，避免“播放永远卡住”的静默失败。
 */
export class StreamingAudioPlayer {
  private context: AudioContext | null = null;
  /** 下一段音频应当开始播放的 AudioContext 时间（秒）。 */
  private nextStartTime = 0;
  /** 期望播放的下一个分片序号。 */
  private nextExpectedIndex = 0;
  /** 暂存“提前到达但还轮不到播放”的分片。 */
  private readonly pending = new Map<number, AudioBuffer>();
  /** 已排程但尚未播放完的源节点，用于取消时统一停止。 */
  private readonly scheduledSources = new Set<AudioBufferSourceNode>();

  private stuckHandler: (() => void) | null = null;
  private stuckTimeoutMs = 30_000;
  private watchdog: ReturnType<typeof setTimeout> | null = null;

  /**
   * 注册“分片缺失超时”回调：当 `pending` 中有分片但 `nextExpectedIndex` 长时间
   * 未推进时触发。上层通常会用它去 abort 流并提示用户。
   */
  setStuckHandler(handler: () => void, timeoutMs = 30_000): void {
    this.stuckHandler = handler;
    this.stuckTimeoutMs = timeoutMs;
  }

  /** 延迟创建 AudioContext：必须在用户手势（点击生成）后创建/恢复才能发声。 */
  private ensureContext(): AudioContext {
    if (!this.context) {
      const Ctor: typeof AudioContext =
        window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.context = new Ctor();
      this.nextStartTime = this.context.currentTime;
    }
    // 某些浏览器初始为 suspended，需要显式恢复。
    if (this.context.state === "suspended") {
      void this.context.resume();
    }
    return this.context;
  }

  /**
   * 接收一个分片的 PCM 数据并按 index 顺序排程播放。
   *
   * @param index 分片序号（从 0 开始）
   * @param samples 该分片的 Float32 PCM（[-1, 1]）
   * @param sampleRate 该分片的原始采样率
   */
  enqueue(index: number, samples: Float32Array, sampleRate: number): void {
    if (samples.length === 0) {
      // AudioBuffer 长度必须 >= 1；空分片直接丢弃，并把 nextExpectedIndex 推进。
      if (index === this.nextExpectedIndex) {
        this.nextExpectedIndex += 1;
        this.flushPending();
      } else {
        // 乱序到达的空分片：占位推进 pending 序列。
        this.pending.set(index, this.makeEmptyBuffer(sampleRate));
        this.flushPending();
      }
      return;
    }
    const context = this.ensureContext();

    // 用分片原始采样率创建 AudioBuffer；播放时由 AudioContext 自行重采样到输出设备，
    // 仅用于即时播放，不影响“下载用”的原始采样率（下载走 wav.ts 单独重组）。
    const buffer = context.createBuffer(1, samples.length, sampleRate);
    // 用 set() 写入声道数据，避免 copyToChannel 对 Float32Array 泛型类型的严格约束。
    buffer.getChannelData(0).set(samples);

    this.pending.set(index, buffer);
    this.flushPending();
  }

  /** 创建 1-sample 静音 buffer 用作空分片占位（AudioBuffer length 必须 >= 1）。 */
  private makeEmptyBuffer(sampleRate: number): AudioBuffer {
    const context = this.ensureContext();
    return context.createBuffer(1, 1, sampleRate);
  }

  /** 把缓冲中“按序就绪”的分片依次排程播放。 */
  private flushPending(): void {
    const context = this.context;
    if (!context) {
      return;
    }

    while (this.pending.has(this.nextExpectedIndex)) {
      const buffer = this.pending.get(this.nextExpectedIndex)!;
      this.pending.delete(this.nextExpectedIndex);
      this.nextExpectedIndex += 1;

      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);

      // 若游标已落后于当前时间（首段或出现空档），从当前时间起播，避免负延迟。
      const startAt = Math.max(this.nextStartTime, context.currentTime);
      source.start(startAt);
      this.nextStartTime = startAt + buffer.duration;

      this.scheduledSources.add(source);
      source.onended = () => {
        this.scheduledSources.delete(source);
      };
    }

    // pending 仍非空意味着某个分片迟到 / 丢失，启动看门狗；否则关闭。
    if (this.pending.size > 0) {
      this.armWatchdog();
    } else {
      this.clearWatchdog();
    }
  }

  private armWatchdog(): void {
    this.clearWatchdog();
    if (!this.stuckHandler) {
      return;
    }
    this.watchdog = setTimeout(() => {
      const handler = this.stuckHandler;
      this.watchdog = null;
      handler?.();
    }, this.stuckTimeoutMs);
  }

  private clearWatchdog(): void {
    if (this.watchdog !== null) {
      clearTimeout(this.watchdog);
      this.watchdog = null;
    }
  }

  /** 停止播放并释放资源（用于取消或开始新一轮生成前清理）。 */
  stop(): void {
    this.clearWatchdog();
    for (const source of this.scheduledSources) {
      try {
        source.stop();
      } catch {
        // 已结束的源再次 stop 会抛错，忽略即可。
      }
      source.disconnect();
    }
    this.scheduledSources.clear();
    this.pending.clear();
    this.nextExpectedIndex = 0;

    if (this.context) {
      void this.context.close();
      this.context = null;
    }
    this.nextStartTime = 0;
  }
}
