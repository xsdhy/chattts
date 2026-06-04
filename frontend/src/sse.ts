/**
 * 基于 `fetch` + `ReadableStream` 的 SSE 解析器。
 *
 * 浏览器原生 `EventSource` 只支持 GET，无法携带 JSON POST body，因此流式接口
 * 必须用 `fetch` 发起 POST，再手动按 SSE 协议解析响应字节流（见需求 7.2）。
 */

/** 一条已解析的 SSE 事件。 */
export type SseEvent = {
  event: string;
  data: string;
};

/** 同时兼容 LF 与 CRLF 事件分隔符，避免对整个 buffer 做 O(n²) 的字符串替换。 */
const EVENT_DELIMITER_RE = /\r?\n\r?\n/;

/**
 * 把 `fetch` 响应体（字节流）解析为 SSE 事件的异步迭代器。
 *
 * SSE 线格式：事件之间用空行（`\n\n` 或 `\r\n\r\n`）分隔；事件内 `event:` 与
 * `data:` 各占一行。这里按空行切分事件块，并把同一块内的多行 `data:` 用换行
 * 拼接（后端为单行 data，但按规范做兼容处理）。
 */
export async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    while (true) {
      if (signal?.aborted) {
        break;
      }
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });

      let match = buffer.match(EVENT_DELIMITER_RE);
      while (match) {
        const separatorIndex = match.index ?? 0;
        const rawBlock = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + match[0].length);

        const parsed = parseEventBlock(rawBlock);
        if (parsed) {
          yield parsed;
        }
        match = buffer.match(EVENT_DELIMITER_RE);
      }
    }
  } finally {
    // 取消或异常时释放底层读取器，触发服务端断开检测。
    try {
      await reader.cancel();
    } catch {
      // 已关闭的流再次 cancel 可能抛错，忽略。
    }
  }
}

/** 解析单个事件块（多行）为 `{event, data}`；无 data 的块返回 null。 */
function parseEventBlock(block: string): SseEvent | null {
  let event = "message";
  const dataLines: string[] = [];

  // 兼容 LF / CRLF 的行分隔。
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      // SSE 规范：字段名后允许有一个可选前导空格作为分隔符。
      event = line.slice("event:".length).replace(/^ /, "");
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).replace(/^ /, ""));
    }
    // 以 ":" 开头的注释行与其它字段按需忽略。
  }

  if (dataLines.length === 0) {
    return null;
  }
  return { event, data: dataLines.join("\n") };
}
