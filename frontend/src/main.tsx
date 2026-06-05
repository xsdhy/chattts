import React from "react";
import ReactDOM from "react-dom/client";
import {
  ChevronDown,
  ChevronUp,
  Dices,
  Download,
  Loader2,
  Mic,
  Play,
  Sparkles,
  Wand2,
  X,
} from "lucide-react";
import "./styles.css";
import { parseSseStream } from "./sse";
import { StreamingAudioPlayer } from "./streamingPlayer";
import { base64ToUint8Array, decodeWav, encodeWav, pcm16ToFloat32 } from "./wav";

/** 单个音色项，对应后端 SpeakerResponse。 */
type Speaker = {
  id: string;
  seed: number | null;
  display_name: string;
};

/** 生成模式：普通（一次性返回 WAV）/ 流式（SSE 边收边播）。 */
type Mode = "normal" | "stream";

/** `/api/tts/preprocess` 预览响应，对应后端 PreprocessResponse。 */
type PreprocessPreview = {
  original_text: string;
  normalized_text: string;
  refine_prompt: string | null;
  prosody: string;
  profile: string;
  refine_mode: "token_injection" | "refine_prompt";
  segments: { index: number; text: string }[];
  changes: { type: string; from: string; to: string }[];
};

/** 健康检查响应，对应后端 HealthResponse。 */
type Health = {
  status: string;
  model_loaded: boolean;
  device: string;
  default_speaker: string;
  max_concurrency: number;
  queue_timeout: number;
};

/**
 * 单个分片在 UI 中的状态：
 * - pending  : 后端 progress 事件已声明，但音频还未到达
 * - received : 音频已到达且已入队播放器，但还轮不到播
 * - playing  : 正在通过扬声器播放
 * - played   : 已播放完毕
 */
type ChunkState = "pending" | "received" | "playing" | "played";

type ChunkInfo = {
  index: number;
  state: ChunkState;
  /** 该分片实际音频时长（毫秒），用于按比例渲染时间线宽度。 */
  durationMs: number;
};

/**
 * 流式生成的整体阶段，用于驱动顶部步进器：
 * connecting → synthesizing → playing → finalizing → done
 * 出错时直接落到 "error"。
 */
type StreamStage =
  | "idle"
  | "connecting"
  | "synthesizing"
  | "playing"
  | "finalizing"
  | "done"
  | "error";

// 与后端 MAX_TEXT_LEN 默认值保持一致；超过会被后端拒绝。
const MAX_TEXT_LEN = 2000;
const DEFAULT_TEXT = "你好，欢迎使用 ChatTTS 中文语音合成服务。";

// 流式分片字符上限的可调范围；与后端 schema 约束（10~500）保持一致。
const STREAM_CHUNK_MIN = 10;
const STREAM_CHUNK_MAX = 200;
const STREAM_CHUNK_DEFAULT = 50;

function App() {
  // ---- 文本与音色 ----
  const [text, setText] = React.useState(DEFAULT_TEXT);
  const [speakers, setSpeakers] = React.useState<Speaker[]>([]);
  const [speaker, setSpeaker] = React.useState("");
  const [health, setHealth] = React.useState<Health | null>(null);

  // ---- 基础参数 ----
  const [speed, setSpeed] = React.useState(1);
  const [mode, setMode] = React.useState<Mode>("normal");
  // 流式分片字符上限：从 localStorage 读取，缺省 50。
  const [streamChunkChars, setStreamChunkChars] = React.useState<number>(() => {
    try {
      const raw = localStorage.getItem("chattts.streamChunkChars");
      if (!raw) return STREAM_CHUNK_DEFAULT;
      const v = Number(raw);
      if (!Number.isFinite(v)) return STREAM_CHUNK_DEFAULT;
      return Math.max(STREAM_CHUNK_MIN, Math.min(STREAM_CHUNK_MAX, Math.round(v)));
    } catch {
      return STREAM_CHUNK_DEFAULT;
    }
  });

  // ---- 高级参数（折叠区）----
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [refine, setRefine] = React.useState(false);
  const [temperature, setTemperature] = React.useState(0.3);
  const [topP, setTopP] = React.useState(0.7);
  const [topK, setTopK] = React.useState(20);

  // ---- 文本预处理参数（对应后端 6.1，普通/流式都生效）----
  const [preprocess, setPreprocess] = React.useState(true);
  const [preprocessProfile, setPreprocessProfile] =
    React.useState<"plain" | "balanced" | "expressive">("balanced");
  const [prosody, setProsody] = React.useState<
    "flat" | "natural" | "dialogue" | "narration"
  >("natural");
  const [allowControlTokens, setAllowControlTokens] = React.useState(false);

  // ---- 「一键整理」预览状态（落实 D5：仅展示，不改写输入框）----
  const [isPreviewing, setIsPreviewing] = React.useState(false);
  const [preview, setPreview] = React.useState<PreprocessPreview | null>(null);

  // ---- 交互状态 ----
  const [isSampling, setIsSampling] = React.useState(false);
  const [isGenerating, setIsGenerating] = React.useState(false);
  // 流式生成时的按钮状态文案：连接中 / 生成 2/5 / 播放中 / 整理音频。
  const [statusLabel, setStatusLabel] = React.useState("");
  const [error, setError] = React.useState("");
  const [audioUrl, setAudioUrl] = React.useState("");
  const [audioBlob, setAudioBlob] = React.useState<Blob | null>(null);

  // ---- 流式可视化状态 ----
  const [stage, setStage] = React.useState<StreamStage>("idle");
  const [chunks, setChunks] = React.useState<ChunkInfo[]>([]);
  /** TTFB：从点击生成到收到第一段音频的毫秒。仅在流式模式下有意义。 */
  const [ttfbMs, setTtfbMs] = React.useState<number | null>(null);
  /** 当前正在播放的分片 index（驱动“正在播”的高亮效果）。 */
  const [playingIndex, setPlayingIndex] = React.useState<number | null>(null);
  /** 已缓冲秒数（来自 player.getBufferedAhead），由 rAF 拉取。 */
  const [bufferedAhead, setBufferedAhead] = React.useState(0);

  // 流式相关的可变引用：当前请求的中止控制器与播放器实例。
  const abortRef = React.useRef<AbortController | null>(null);
  const playerRef = React.useRef<StreamingAudioPlayer | null>(null);

  // streamChunkChars 持久化到 localStorage（隐私模式失败时静默忽略）。
  React.useEffect(() => {
    try {
      localStorage.setItem("chattts.streamChunkChars", String(streamChunkChars));
    } catch {
      // ignore
    }
  }, [streamChunkChars]);

  // 挂载时拉取音色列表与健康状态。
  React.useEffect(() => {
    let ignore = false;

    async function bootstrap() {
      try {
        const [speakersRes, healthRes] = await Promise.all([
          fetch("/api/speakers"),
          fetch("/health"),
        ]);
        if (speakersRes.ok) {
          const data = (await speakersRes.json()) as Speaker[];
          if (!ignore) {
            setSpeakers(data);
            setSpeaker(data[0]?.id ?? "");
          }
        }
        if (healthRes.ok) {
          const data = (await healthRes.json()) as Health;
          if (!ignore) {
            setHealth(data);
          }
        }
      } catch (err) {
        if (!ignore) {
          setError(err instanceof Error ? err.message : "初始化失败");
        }
      }
    }

    bootstrap();
    return () => {
      ignore = true;
    };
  }, []);

  // 组件卸载时统一释放对象 URL、中止进行中的流式请求并停止播放。
  // 这里访问的是 audioUrl 的最新值（闭包通过 ref 读取），因此 effect 依赖空数组即可。
  const audioUrlRef = React.useRef("");
  audioUrlRef.current = audioUrl;
  React.useEffect(() => {
    return () => {
      abortRef.current?.abort();
      playerRef.current?.stop();
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, []);

  /** 当播放器存在时，每帧采样 bufferedAhead，给 UI 显示缓冲健康度。 */
  React.useEffect(() => {
    if (!isGenerating && stage !== "playing" && stage !== "finalizing") {
      return;
    }
    // 250ms 拉一次足够展示缓冲变化，避免每帧渲染抖动。
    const id = setInterval(() => {
      const player = playerRef.current;
      if (player) {
        setBufferedAhead(player.getBufferedAhead());
      }
    }, 250);
    return () => clearInterval(id);
  }, [isGenerating, stage]);

  const charCount = text.trim().length;
  const canGenerate = charCount > 0 && Boolean(speaker) && !isGenerating;

  /** 采样一个新的随机音色，并把它加入下拉框、设为当前选中。 */
  async function sampleRandomSpeaker() {
    setError("");
    setIsSampling(true);
    try {
      const response = await fetch("/api/speakers/random", { method: "POST" });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(formatApiError(payload, response.status));
      }
      const newSpeaker = (await response.json()) as Speaker;
      setSpeakers((prev) =>
        prev.some((item) => item.id === newSpeaker.id) ? prev : [newSpeaker, ...prev],
      );
      setSpeaker(newSpeaker.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "随机音色失败");
    } finally {
      setIsSampling(false);
    }
  }

  /** 当前请求体：两种模式共用同一组参数；流式模式追加分片大小参数。 */
  function requestBody(includeChunkSize: boolean) {
    const base: Record<string, unknown> = {
      text,
      speaker,
      speed,
      refine,
      temperature,
      top_p: topP,
      top_k: topK,
      format: "wav",
      // 文本预处理参数：普通与流式都带上（后端两接口都生效，见 6.1）。
      preprocess,
      preprocess_profile: preprocessProfile,
      prosody,
      allow_control_tokens: allowControlTokens,
    };
    if (includeChunkSize) {
      // 流式后续分片目标上限（沿用现有 streamChunkChars）；首片走服务端默认。
      base.max_segment_chars = streamChunkChars;
    }
    return base;
  }

  /**
   * 「一键整理」：调用 `POST /api/tts/preprocess` 预览处理结果（落实 D5）。
   * 仅用于展示，**不改写输入框**；提交 TTS 时仍发送原文，由服务端再次完整预处理。
   * 与 TTS 请求互不阻塞；生成进行中禁用，避免状态错乱。
   */
  async function previewPreprocess() {
    if (charCount === 0 || isGenerating || isPreviewing) {
      return;
    }
    setIsPreviewing(true);
    setError("");
    try {
      const response = await fetch("/api/tts/preprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // 预览也带上流式分片字段，分片预览与流式实际切分一致。
        body: JSON.stringify(requestBody(true)),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(formatApiError(payload, response.status));
      }
      setPreview((await response.json()) as PreprocessPreview);
    } catch (err) {
      setError(err instanceof Error ? err.message : "预览失败");
    } finally {
      setIsPreviewing(false);
    }
  }

  /** 设置新的音频 URL，并 revoke 旧的，避免内存泄漏。 */
  function swapAudioUrl(nextUrl: string) {
    setAudioUrl((previousUrl) => {
      if (previousUrl) {
        URL.revokeObjectURL(previousUrl);
      }
      return nextUrl;
    });
  }

  /** 开始新一轮生成前的统一清理：中止旧请求、停止旧播放、清空错误与可视化状态。 */
  function prepareNewRun() {
    abortRef.current?.abort();
    abortRef.current = null;
    playerRef.current?.stop();
    playerRef.current = null;
    setError("");
    setChunks([]);
    setTtfbMs(null);
    setPlayingIndex(null);
    setBufferedAhead(0);
    setStage("idle");
  }

  /** 把分片更新为指定状态（不存在时新建条目）。 */
  function patchChunk(index: number, patch: Partial<ChunkInfo>) {
    setChunks((prev) => {
      const next = [...prev];
      const i = next.findIndex((c) => c.index === index);
      if (i === -1) {
        next.push({ index, state: "pending", durationMs: 0, ...patch });
      } else {
        next[i] = { ...next[i], ...patch };
      }
      next.sort((a, b) => a.index - b.index);
      return next;
    });
  }

  /** 普通生成：调用 `POST /api/tts`，一次性拿到完整 WAV。 */
  async function generateNormal() {
    prepareNewRun();
    setIsGenerating(true);

    try {
      const response = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody(false)),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(formatApiError(payload, response.status));
      }

      const blob = await response.blob();
      setAudioBlob(blob);
      swapAudioUrl(URL.createObjectURL(blob));
    } catch (err) {
      setError(err instanceof Error ? err.message : "生成失败");
    } finally {
      setIsGenerating(false);
    }
  }

  /**
   * 流式生成：调用 `POST /api/tts/stream`，按 SSE 边接收边播放，完成后基于已解码的
   * PCM 分片重组完整 WAV 供下载。
   */
  async function generateStream() {
    prepareNewRun();
    setIsGenerating(true);
    setStage("connecting");
    setStatusLabel("连接中");

    const t0 = performance.now();
    const controller = new AbortController();
    abortRef.current = controller;
    const player = new StreamingAudioPlayer();
    // 把“正在播放第几段”反馈到 UI，驱动时间线高亮 + 当前段标签。
    player.setCallbacks({
      onChunkPlayStart: (index) => {
        setPlayingIndex(index);
        patchChunk(index, { state: "playing" });
        // 首段开始播放才正式进入 playing 阶段（之前可能仍在 synthesizing）。
        setStage((prev) => (prev === "playing" || prev === "finalizing" ? prev : "playing"));
      },
      onChunkPlayEnd: (index) => {
        patchChunk(index, { state: "played" });
        setPlayingIndex((curr) => (curr === index ? null : curr));
      },
    });
    // 看门狗：30s 内 pending 仍未推进时，主动中止流并提示用户。
    player.setStuckHandler(() => {
      controller.abort();
      setError("流式分片接收超时，请重试");
    }, 30_000);
    playerRef.current = player;

    // 暂存各分片的原始 PCM（按 index），用于完成后重组下载用的完整 WAV。
    const pcmChunks: { index: number; pcm: Int16Array }[] = [];
    // 以 start / 首个分片的采样率为准，作为最终下载 WAV 的采样率。
    let streamSampleRate = 0;
    let firstAudioSeen = false;

    try {
      const response = await fetch("/api/tts/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(requestBody(true)),
        signal: controller.signal,
      });

      if (!response.ok) {
        // 流尚未开始（非 200），沿用普通错误展示逻辑。
        const payload = await response.json().catch(() => null);
        throw new Error(formatApiError(payload, response.status));
      }
      if (!response.body) {
        throw new Error("流式生成中断，请重试");
      }

      // 收到 200 响应即视为合成开始。
      setStage("synthesizing");
      setStatusLabel("准备合成");

      for await (const evt of parseSseStream(response.body, controller.signal)) {
        if (evt.event === "start") {
          const data = JSON.parse(evt.data) as { sample_rate?: number };
          if (data.sample_rate) {
            streamSampleRate = data.sample_rate;
          }
        } else if (evt.event === "progress") {
          const data = JSON.parse(evt.data) as { current: number; total: number };
          // 后端 current 从 1 开始；提前为所有 index 初始化占位，确保时间线长度立刻完整。
          if (data.total > 0) {
            setChunks((prev) => {
              if (prev.length >= data.total) {
                return prev;
              }
              const next = [...prev];
              for (let i = next.length; i < data.total; i += 1) {
                next.push({ index: i, state: "pending", durationMs: 0 });
              }
              return next;
            });
          }
          setStatusLabel(`合成 ${data.current}/${data.total}`);
        } else if (evt.event === "audio") {
          const data = JSON.parse(evt.data) as {
            index: number;
            audio: string;
            sample_rate: number;
          };
          // Base64 -> 字节 -> 解析独立 WAV 分片为原始 PCM 与采样率。
          const decoded = decodeWav(base64ToUint8Array(data.audio));
          if (!streamSampleRate) {
            streamSampleRate = decoded.sampleRate;
          } else if (decoded.sampleRate !== streamSampleRate) {
            // 各分片应当来自同一模型、同一采样率；不一致会让下载的 WAV 音高错乱。
            // 直接中止流并提示用户，而不是“以首个为准”静默写错。
            throw new Error(
              `分片采样率不一致（${decoded.sampleRate} != ${streamSampleRate}），流被中止`,
            );
          }
          // 暂存原始 PCM（下载用），并把解码后的 Float32 入队播放（播放用）。
          pcmChunks.push({ index: data.index, pcm: decoded.pcm });
          player.enqueue(data.index, pcm16ToFloat32(decoded.pcm), decoded.sampleRate);

          const durationMs = (decoded.pcm.length / decoded.sampleRate) * 1000;
          patchChunk(data.index, { state: "received", durationMs });

          if (!firstAudioSeen) {
            firstAudioSeen = true;
            setTtfbMs(performance.now() - t0);
          }
          setStatusLabel("播放中");
        } else if (evt.event === "done") {
          setStage("finalizing");
          setStatusLabel("整理音频");
          // 按 index 排序后用统一采样率重组完整 WAV（不经过 AudioContext 重采样）。
          pcmChunks.sort((a, b) => a.index - b.index);
          const blob = encodeWav(
            pcmChunks.map((chunk) => chunk.pcm),
            streamSampleRate || 24000,
          );
          setAudioBlob(blob);
          swapAudioUrl(URL.createObjectURL(blob));
        } else if (evt.event === "error") {
          const data = JSON.parse(evt.data) as { detail?: string };
          throw new Error(data.detail || "流式生成失败");
        }
      }
      setStage("done");
    } catch (err) {
      // 用户主动取消：不展示错误，仅回到可编辑状态。
      if (controller.signal.aborted) {
        player.stop();
        setStage("idle");
      } else {
        player.stop();
        setStage("error");
        setError(err instanceof Error ? err.message : "流式生成中断，请重试");
      }
    } finally {
      setIsGenerating(false);
      setStatusLabel("");
      abortRef.current = null;
      // 成功路径不在此 stop()：已排程的音频会继续播放至结束。
    }
  }

  /** 取消进行中的流式请求：中止 fetch、停止播放、回到可操作状态。 */
  function cancelStream() {
    abortRef.current?.abort();
    abortRef.current = null;
    playerRef.current?.stop();
    playerRef.current = null;
    setIsGenerating(false);
    setStatusLabel("");
    setError("");
    setStage("idle");
    setPlayingIndex(null);
  }


  /** 点击生成：按当前模式分派到普通或流式生成。 */
  function handleGenerate() {
    if (!canGenerate) {
      return;
    }
    if (mode === "stream") {
      void generateStream();
    } else {
      void generateNormal();
    }
  }

  /** 下载当前合成的 WAV。 */
  function downloadAudio() {
    if (!audioBlob) {
      return;
    }
    const link = document.createElement("a");
    const url = URL.createObjectURL(audioBlob);
    link.href = url;
    link.download = `chattts-${speaker || "voice"}.wav`;
    link.click();
    // 立刻 revoke 在 Safari 上可能让下载流尚未建立就失败，延后 1s 释放。
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // 顶栏状态药丸文案：优先展示设备 + 模型状态。
  const statusText = health
    ? health.model_loaded
      ? `已就绪 · ${health.device.toUpperCase()}`
      : "模型加载中"
    : "连接中";
  const statusClass = health ? (health.model_loaded ? "ok" : "degraded") : "";

  // 生成过程中禁用文本/音色/语速/高级参数/模式切换，避免状态错乱（见需求 7.1.4）。
  const controlsDisabled = isGenerating;
  const primaryLabel = isGenerating
    ? mode === "stream"
      ? statusLabel || "生成中"
      : "生成中"
    : "生成";

  const totalChunks = chunks.length;
  const receivedCount = chunks.filter(
    (c) => c.state === "received" || c.state === "playing" || c.state === "played",
  ).length;
  const playedCount = chunks.filter((c) => c.state === "played").length;
  // 流式面板可见条件：流式模式下，正在跑或者有可视化历史
  const showStreamPanel = mode === "stream" && (isGenerating || chunks.length > 0);

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div className="brand">
            <Mic aria-hidden="true" size={26} />
            <div>
              <h1>ChatTTS 中文语音合成</h1>
              <p>随机音色 · seed 可复现 · 采样参数可调</p>
            </div>
          </div>
          <div className={`status-pill ${statusClass}`}>{statusText}</div>
        </header>

        <div className="editor-grid">
          <section className="input-panel" aria-label="文本输入">
            <div className="panel-heading">
              <h2>文本</h2>
              <span>
                {charCount} / {MAX_TEXT_LEN}
              </span>
            </div>
            <textarea
              value={text}
              maxLength={MAX_TEXT_LEN}
              disabled={controlsDisabled}
              onChange={(event) => setText(event.target.value)}
              placeholder="输入要合成的文本，建议尽量简短，过长文本在 CPU 上会很慢"
            />

            <div className="preview-actions">
              <button
                className="secondary-button"
                type="button"
                disabled={charCount === 0 || controlsDisabled || isPreviewing}
                onClick={previewPreprocess}
                title="预览文本将被如何朗读（不会改写输入框）"
              >
                {isPreviewing ? (
                  <Loader2 className="spin" size={16} aria-hidden="true" />
                ) : (
                  <Wand2 size={16} aria-hidden="true" />
                )}
                <span>一键整理 / 预览</span>
              </button>
              {preview && (
                <button
                  className="link-button"
                  type="button"
                  onClick={() => setPreview(null)}
                >
                  收起预览
                </button>
              )}
            </div>

            {preview && <PreprocessPreviewPanel preview={preview} />}

            {showStreamPanel && (
              <StreamPanel
                stage={stage}
                chunks={chunks}
                playingIndex={playingIndex}
                ttfbMs={ttfbMs}
                bufferedAhead={bufferedAhead}
                receivedCount={receivedCount}
                playedCount={playedCount}
                totalChunks={totalChunks}
              />
            )}
          </section>

          <aside className="control-panel" aria-label="合成控制">
            <div className="field">
              <span>生成模式</span>
              <div className="mode-switch" role="group" aria-label="生成模式切换">
                <button
                  type="button"
                  className={mode === "normal" ? "active" : ""}
                  disabled={controlsDisabled}
                  onClick={() => setMode("normal")}
                >
                  普通生成
                </button>
                <button
                  type="button"
                  className={mode === "stream" ? "active" : ""}
                  disabled={controlsDisabled}
                  onClick={() => setMode("stream")}
                >
                  流式生成
                </button>
              </div>
            </div>

            <label className="field">
              <span>音色</span>
              <div className="speaker-row">
                <select
                  value={speaker}
                  disabled={controlsDisabled}
                  onChange={(event) => setSpeaker(event.target.value)}
                >
                  {speakers.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.display_name}
                    </option>
                  ))}
                </select>
                <button
                  className="icon-button"
                  type="button"
                  title="随机音色"
                  disabled={isSampling || controlsDisabled}
                  onClick={sampleRandomSpeaker}
                >
                  {isSampling ? (
                    <Loader2 className="spin" size={18} aria-hidden="true" />
                  ) : (
                    <Dices size={18} aria-hidden="true" />
                  )}
                </button>
              </div>
            </label>

            <label className="field">
              <span>语速 {speed.toFixed(1)}x</span>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.1"
                value={speed}
                disabled={controlsDisabled}
                onChange={(event) => setSpeed(Number(event.target.value))}
              />
            </label>

            {mode === "stream" && (
              <label className="field">
                <span>
                  分片大小 {streamChunkChars} 字
                  <span className="hint"> · 越小首响越快</span>
                </span>
                <input
                  type="range"
                  min={STREAM_CHUNK_MIN}
                  max={STREAM_CHUNK_MAX}
                  step={5}
                  value={streamChunkChars}
                  disabled={controlsDisabled}
                  onChange={(event) =>
                    setStreamChunkChars(Number(event.target.value))
                  }
                />
              </label>
            )}

            <div className="advanced">
              <button
                className="advanced-toggle"
                type="button"
                disabled={controlsDisabled}
                onClick={() => setShowAdvanced((prev) => !prev)}
              >
                <span>高级参数</span>
                {showAdvanced ? (
                  <ChevronUp size={18} aria-hidden="true" />
                ) : (
                  <ChevronDown size={18} aria-hidden="true" />
                )}
              </button>
              {showAdvanced && (
                <div className="advanced-body">
                  <label className="switch-row">
                    <span title="对数字/日期/单位/Markdown 等做轻量可读化处理">
                      文本增强
                    </span>
                    <input
                      type="checkbox"
                      checked={preprocess}
                      disabled={controlsDisabled}
                      onChange={(event) => setPreprocess(event.target.checked)}
                    />
                  </label>
                  <label className="field">
                    <span>增强强度</span>
                    <select
                      value={preprocessProfile}
                      disabled={controlsDisabled || !preprocess}
                      onChange={(event) =>
                        setPreprocessProfile(
                          event.target.value as typeof preprocessProfile,
                        )
                      }
                    >
                      <option value="plain">简洁</option>
                      <option value="balanced">均衡（默认）</option>
                      <option value="expressive">表现力</option>
                    </select>
                  </label>
                  <label className="field">
                    <span>朗读风格</span>
                    <select
                      value={prosody}
                      disabled={controlsDisabled || !preprocess}
                      onChange={(event) =>
                        setProsody(event.target.value as typeof prosody)
                      }
                    >
                      <option value="flat">平实</option>
                      <option value="natural">自然</option>
                      <option value="dialogue">对话</option>
                      <option value="narration">旁白</option>
                    </select>
                  </label>
                  <label className="switch-row">
                    <span title="仅适合高级用户：允许文本中的 ChatTTS 控制 token（如 [uv_break]）生效">
                      允许控制 token
                    </span>
                    <input
                      type="checkbox"
                      checked={allowControlTokens}
                      disabled={controlsDisabled || !preprocess}
                      onChange={(event) =>
                        setAllowControlTokens(event.target.checked)
                      }
                    />
                  </label>
                  <label className="switch-row">
                    <span title="开启后由 ChatTTS refine 接管停顿（与上面的「朗读风格」二者互斥）">
                      文本 refine
                    </span>
                    <input
                      type="checkbox"
                      checked={refine}
                      disabled={controlsDisabled}
                      onChange={(event) => setRefine(event.target.checked)}
                    />
                  </label>
                  <label className="field">
                    <span>temperature {temperature.toFixed(2)}</span>
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.05"
                      value={temperature}
                      disabled={controlsDisabled}
                      onChange={(event) => setTemperature(Number(event.target.value))}
                    />
                  </label>
                  <label className="field">
                    <span>top_p {topP.toFixed(2)}</span>
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.05"
                      value={topP}
                      disabled={controlsDisabled}
                      onChange={(event) => setTopP(Number(event.target.value))}
                    />
                  </label>
                  <label className="field">
                    <span>top_k {topK}</span>
                    <input
                      type="range"
                      min="1"
                      max="50"
                      step="1"
                      value={topK}
                      disabled={controlsDisabled}
                      onChange={(event) => setTopK(Number(event.target.value))}
                    />
                  </label>
                </div>
              )}
            </div>

            <button
              className="primary-button"
              type="button"
              disabled={!canGenerate}
              onClick={handleGenerate}
            >
              {isGenerating ? (
                <Loader2 className="spin" size={19} aria-hidden="true" />
              ) : (
                <Sparkles size={19} aria-hidden="true" />
              )}
              <span>{primaryLabel}</span>
            </button>

            {mode === "stream" && isGenerating && (
              <button
                className="secondary-button"
                type="button"
                onClick={cancelStream}
              >
                <X size={18} aria-hidden="true" />
                <span>取消生成</span>
              </button>
            )}

            {error && <div className="error-box">{error}</div>}

            <div className="player-panel">
              <div className="player-title">
                <Play size={18} aria-hidden="true" />
                <span>试听</span>
              </div>
              <audio controls src={audioUrl} />
              <button
                className="secondary-button"
                type="button"
                disabled={!audioBlob}
                onClick={downloadAudio}
              >
                <Download size={18} aria-hidden="true" />
                <span>下载 WAV</span>
              </button>
            </div>
          </aside>
        </div>
      </section>
    </main>
  );
}

/* ============================================================
 * 流式可视化子组件（克制版：静态布局，仅在数值变化时更新内容）
 * ============================================================ */

/** 阶段中文描述：用一行文字代替之前的步进器，没有任何动画。 */
const STAGE_LABEL: Record<StreamStage, string> = {
  idle: "待命",
  connecting: "建立连接",
  synthesizing: "合成中",
  playing: "播放中",
  finalizing: "整理音频",
  done: "完成",
  error: "出错",
};

/**
 * 分片网格：每段一个等宽小方块，只通过颜色区分四种状态，不做任何 ripple/脉冲/pop。
 * 给用户一个“整体进度地图”，而不是闪烁的舞台。
 */
function ChunkGrid({
  chunks,
  playingIndex,
}: {
  chunks: ChunkInfo[];
  playingIndex: number | null;
}) {
  if (chunks.length === 0) {
    return null;
  }
  return (
    <div className="chunk-grid" role="list" aria-label="分片进度">
      {chunks.map((chunk) => {
        const isPlaying = chunk.index === playingIndex;
        return (
          <span
            key={chunk.index}
            role="listitem"
            className={`cell cell-${chunk.state}${isPlaying ? " cell-current" : ""}`}
            title={`分片 ${chunk.index + 1}${chunk.durationMs ? ` · ${(chunk.durationMs / 1000).toFixed(2)}s` : ""}`}
            aria-label={`分片 ${chunk.index + 1} ${chunk.state}`}
          />
        );
      })}
    </div>
  );
}

/**
 * 流式综合面板：阶段文字 + 进度条 + 分片网格 + 静态数值。
 * 设计目标：让信息可见，但不靠动画吸引注意力。
 */
function StreamPanel(props: {
  stage: StreamStage;
  chunks: ChunkInfo[];
  playingIndex: number | null;
  ttfbMs: number | null;
  bufferedAhead: number;
  receivedCount: number;
  playedCount: number;
  totalChunks: number;
}) {
  const {
    stage,
    chunks,
    playingIndex,
    ttfbMs,
    bufferedAhead,
    receivedCount,
    playedCount,
    totalChunks,
  } = props;

  // 整体接收进度（0~1）；未知 total 时退化为 0，避免进度条乱跳。
  const ratio = totalChunks > 0 ? receivedCount / totalChunks : 0;

  return (
    <div className="stream-panel">
      <div className="stream-row">
        <span className="stream-stage">{STAGE_LABEL[stage]}</span>
        <span className="stream-count">
          {receivedCount}
          <span className="muted">/{totalChunks || "?"}</span>
          <span className="muted"> 已就绪</span>
        </span>
      </div>

      <div
        className="stream-bar"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={totalChunks || 1}
        aria-valuenow={receivedCount}
      >
        <div className="stream-bar-fill" style={{ width: `${ratio * 100}%` }} />
      </div>

      <ChunkGrid chunks={chunks} playingIndex={playingIndex} />

      <dl className="stream-meta">
        <div>
          <dt>已播放</dt>
          <dd>
            {playedCount}
            <span className="muted">/{totalChunks || "?"}</span>
          </dd>
        </div>
        <div>
          <dt>首响</dt>
          <dd>{ttfbMs === null ? "—" : `${(ttfbMs / 1000).toFixed(2)} s`}</dd>
        </div>
        <div>
          <dt>缓冲</dt>
          <dd>{bufferedAhead.toFixed(2)} s</dd>
        </div>
      </dl>
    </div>
  );
}

/**
 * 「一键整理」只读预览面板（落实 8.2）：展示归一化文本、分片列表、变更记录、以及
 * D1 路径（refine_mode / refine_prompt）。不改写输入框，仅供用户预览所见即所得。
 */
function PreprocessPreviewPanel({ preview }: { preview: PreprocessPreview }) {
  const modeLabel =
    preview.refine_mode === "refine_prompt"
      ? `refine 接管停顿（${preview.refine_prompt ?? ""}）`
      : "标点停顿（不注入 token）";
  return (
    <div className="preview-panel" aria-label="预处理预览">
      <div className="preview-row">
        <span className="preview-tag">朗读策略</span>
        <span>
          {preview.profile} · {preview.prosody} · {modeLabel}
        </span>
      </div>

      <div className="preview-block">
        <span className="preview-tag">归一化文本</span>
        <p className="preview-text">{preview.normalized_text || "（空）"}</p>
      </div>

      <div className="preview-block">
        <span className="preview-tag">分片（{preview.segments.length}）</span>
        <ol className="preview-segments">
          {preview.segments.map((seg) => (
            <li key={seg.index}>{seg.text}</li>
          ))}
        </ol>
      </div>

      {preview.changes.length > 0 && (
        <div className="preview-block">
          <span className="preview-tag">变更（{preview.changes.length}）</span>
          <ul className="preview-changes">
            {preview.changes.map((change, i) => (
              <li key={i}>
                <code>{change.from}</code>
                <span className="muted"> → </span>
                <code>{change.to}</code>
                <span className="muted"> · {change.type}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** 把后端错误响应解析成可展示文案，回退到 HTTP 状态码。 */
function formatApiError(payload: unknown, status: number) {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (
      Array.isArray(detail) &&
      detail[0] &&
      typeof detail[0] === "object" &&
      "msg" in detail[0]
    ) {
      return String((detail[0] as { msg: unknown }).msg);
    }
  }
  return `请求失败：${status}`;
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
