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

/** 健康检查响应，对应后端 HealthResponse。 */
type Health = {
  status: string;
  model_loaded: boolean;
  device: string;
  default_speaker: string;
  max_concurrency: number;
  queue_timeout: number;
};

// 与后端 MAX_TEXT_LEN 默认值保持一致；超过会被后端拒绝。
const MAX_TEXT_LEN = 2000;
const DEFAULT_TEXT = "你好，欢迎使用 ChatTTS 中文语音合成服务。";

function App() {
  // ---- 文本与音色 ----
  const [text, setText] = React.useState(DEFAULT_TEXT);
  const [speakers, setSpeakers] = React.useState<Speaker[]>([]);
  const [speaker, setSpeaker] = React.useState("");
  const [health, setHealth] = React.useState<Health | null>(null);

  // ---- 基础参数 ----
  const [speed, setSpeed] = React.useState(1);
  const [mode, setMode] = React.useState<Mode>("normal");

  // ---- 高级参数（折叠区）----
  const [showAdvanced, setShowAdvanced] = React.useState(false);
  const [refine, setRefine] = React.useState(false);
  const [temperature, setTemperature] = React.useState(0.3);
  const [topP, setTopP] = React.useState(0.7);
  const [topK, setTopK] = React.useState(20);

  // ---- 交互状态 ----
  const [isSampling, setIsSampling] = React.useState(false);
  const [isGenerating, setIsGenerating] = React.useState(false);
  // 流式生成时的按钮状态文案：连接中 / 生成 2/5 / 播放中 / 整理音频。
  const [statusLabel, setStatusLabel] = React.useState("");
  const [error, setError] = React.useState("");
  const [audioUrl, setAudioUrl] = React.useState("");
  const [audioBlob, setAudioBlob] = React.useState<Blob | null>(null);

  // 流式相关的可变引用：当前请求的中止控制器与播放器实例。
  const abortRef = React.useRef<AbortController | null>(null);
  const playerRef = React.useRef<StreamingAudioPlayer | null>(null);

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

  // 组件卸载或音频 URL 变化时释放对象 URL，避免内存泄漏。
  React.useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  // 组件卸载时确保中止进行中的流式请求并停止播放，避免资源泄漏。
  React.useEffect(() => {
    return () => {
      abortRef.current?.abort();
      playerRef.current?.stop();
    };
  }, []);

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

  /** 当前请求体：两种模式共用同一组参数。 */
  function requestBody() {
    return {
      text,
      speaker,
      speed,
      refine,
      temperature,
      top_p: topP,
      top_k: topK,
      format: "wav",
    };
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

  /** 开始新一轮生成前的统一清理：中止旧请求、停止旧播放、清空错误。 */
  function prepareNewRun() {
    abortRef.current?.abort();
    abortRef.current = null;
    playerRef.current?.stop();
    playerRef.current = null;
    setError("");
  }

  /** 普通生成：调用 `POST /api/tts`，一次性拿到完整 WAV。 */
  async function generateNormal() {
    prepareNewRun();
    setIsGenerating(true);

    try {
      const response = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody()),
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
    setStatusLabel("连接中");

    const controller = new AbortController();
    abortRef.current = controller;
    const player = new StreamingAudioPlayer();
    playerRef.current = player;

    // 暂存各分片的原始 PCM（按 index），用于完成后重组下载用的完整 WAV。
    const pcmChunks: { index: number; pcm: Int16Array }[] = [];
    // 以 start / 首个分片的采样率为准，作为最终下载 WAV 的采样率。
    let streamSampleRate = 0;

    try {
      const response = await fetch("/api/tts/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(requestBody()),
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

      for await (const evt of parseSseStream(response.body, controller.signal)) {
        if (evt.event === "start") {
          const data = JSON.parse(evt.data) as { sample_rate?: number };
          if (data.sample_rate) {
            streamSampleRate = data.sample_rate;
          }
        } else if (evt.event === "progress") {
          const data = JSON.parse(evt.data) as { current: number; total: number };
          setStatusLabel(`生成 ${data.current}/${data.total}`);
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
            // 各分片来自同一模型，采样率理应一致；不一致时以首个为准并告警。
            console.warn(
              `分片采样率不一致：${decoded.sampleRate} != ${streamSampleRate}，以首个为准`,
            );
          }
          // 暂存原始 PCM（下载用），并把解码后的 Float32 入队播放（播放用）。
          pcmChunks.push({ index: data.index, pcm: decoded.pcm });
          player.enqueue(data.index, pcm16ToFloat32(decoded.pcm), decoded.sampleRate);
          setStatusLabel("播放中");
        } else if (evt.event === "done") {
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
    } catch (err) {
      // 用户主动取消：不展示错误，仅回到可编辑状态。
      if (controller.signal.aborted) {
        player.stop();
      } else {
        player.stop();
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
    link.href = URL.createObjectURL(audioBlob);
    link.download = `chattts-${speaker || "voice"}.wav`;
    link.click();
    URL.revokeObjectURL(link.href);
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
                    <span>文本 refine</span>
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
