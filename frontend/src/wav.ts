/**
 * 最小 WAV 解析 / 封装工具（仅支持 PCM_16，后端输出即为该格式）。
 *
 * 为什么需要它（见需求 7.2 “解码路径区分”）：
 * - 浏览器的 `AudioContext.decodeAudioData` 会把音频重采样到 `AudioContext.sampleRate`
 *   （通常 44100/48000），用于即时播放没问题，但若拿它的结果再编码下载，得到的 WAV
 *   采样率会被改写（如 48000），与后端 24000 不一致，也与普通 `/api/tts` 输出不一致。
 * - 因此“下载用”的完整 WAV 必须直接解析每个分片 WAV 头，取出原始 PCM 与采样率，
 *   按 index 顺序拼接后用**同一采样率**重新封装。
 */

/** 解析后的单个 WAV 分片：保留原始采样率与 16-bit PCM 数据。 */
export type DecodedWav = {
  sampleRate: number;
  numChannels: number;
  /** 交错存放的 16-bit PCM 采样（后端为单声道，等价于单通道序列）。 */
  pcm: Int16Array;
};

/** 把 Base64 字符串解码为字节数组（使用浏览器内置 `atob`）。 */
export function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

/** 读取 4 个字节并拼成 ASCII chunk id（如 "RIFF"/"fmt "/"data"）。 */
function readChunkId(view: DataView, offset: number): string {
  return (
    String.fromCharCode(view.getUint8(offset)) +
    String.fromCharCode(view.getUint8(offset + 1)) +
    String.fromCharCode(view.getUint8(offset + 2)) +
    String.fromCharCode(view.getUint8(offset + 3))
  );
}

/**
 * 解析一个独立的 PCM_16 WAV 分片，返回采样率与 PCM 数据。
 *
 * 解析方式：跳过 12 字节的 RIFF/WAVE 头后，逐个遍历子 chunk，按 chunk id 取出
 * `fmt ` 中的采样率/声道，以及 `data` 中的 PCM 字节。对 chunk 的奇数长度做字节对齐。
 */
export function decodeWav(bytes: Uint8Array): DecodedWav {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (readChunkId(view, 0) !== "RIFF" || readChunkId(view, 8) !== "WAVE") {
    throw new Error("分片不是合法的 WAV 文件");
  }

  let sampleRate = 0;
  let numChannels = 1;
  let bitsPerSample = 16;
  let dataOffset = -1;
  let dataLength = 0;

  // RIFF 头共 12 字节，之后是若干子 chunk。
  let offset = 12;
  while (offset + 8 <= view.byteLength) {
    const chunkId = readChunkId(view, offset);
    const chunkSize = view.getUint32(offset + 4, true);
    const bodyOffset = offset + 8;

    if (chunkId === "fmt ") {
      numChannels = view.getUint16(bodyOffset + 2, true);
      sampleRate = view.getUint32(bodyOffset + 4, true);
      bitsPerSample = view.getUint16(bodyOffset + 14, true);
    } else if (chunkId === "data") {
      dataOffset = bodyOffset;
      dataLength = chunkSize;
    }

    // chunk 以偶数字节对齐：奇数长度后面会有 1 字节填充。
    offset = bodyOffset + chunkSize + (chunkSize % 2);
  }

  if (bitsPerSample !== 16) {
    throw new Error(`仅支持 16-bit PCM 分片，实际为 ${bitsPerSample}-bit`);
  }
  if (dataOffset < 0) {
    throw new Error("WAV 分片缺少 data chunk");
  }

  // 把 data 字节拷贝到独立缓冲，避免 Int16Array 对未对齐 offset 的限制。
  const dataBytes = bytes.slice(dataOffset, dataOffset + dataLength);
  const pcm = new Int16Array(
    dataBytes.buffer,
    dataBytes.byteOffset,
    Math.floor(dataBytes.byteLength / 2),
  );

  return { sampleRate, numChannels, pcm };
}

/**
 * 把若干 16-bit PCM 分片按顺序拼接，封装为一个完整的 PCM_16 单声道 WAV Blob。
 *
 * 用于“流式完成后下载完整音频”：保持与后端一致的采样率，不经过任何重采样。
 */
export function encodeWav(chunks: Int16Array[], sampleRate: number): Blob {
  const totalSamples = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const numChannels = 1;
  const bytesPerSample = 2;
  const dataSize = totalSamples * bytesPerSample;

  // 标准 44 字节 WAV 头 + PCM 数据体。
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeAscii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true); // 整个文件大小 - 8
  writeAscii(8, "WAVE");

  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk 大小（PCM 为 16）
  view.setUint16(20, 1, true); // audioFormat = 1（PCM）
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true); // byteRate
  view.setUint16(32, numChannels * bytesPerSample, true); // blockAlign
  view.setUint16(34, 16, true); // bitsPerSample

  writeAscii(36, "data");
  view.setUint32(40, dataSize, true);

  // 顺序写入各分片 PCM 数据。
  let offset = 44;
  for (const chunk of chunks) {
    for (let i = 0; i < chunk.length; i += 1) {
      view.setInt16(offset, chunk[i], true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/** 把 16-bit PCM 转为 Web Audio 需要的 [-1, 1] Float32 序列。 */
export function pcm16ToFloat32(pcm: Int16Array): Float32Array {
  const floats = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i += 1) {
    floats[i] = pcm[i] / 32768;
  }
  return floats;
}
