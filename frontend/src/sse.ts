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

/**
 * 把 `fetch` 响应体（字节流）解析为 SSE 事件的异步迭代器。
 *
 * SSE 线格式：事件之间用空行（`\n\n`）分隔；事件内 `event:` 与 `data:` 各占一行。
 * 这里按“空行”切分事件块，并把同一块内的多行 `data:` 用换行拼接（后端为单行 data，
 * 但按规范做兼容处理）。
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

      // 统一换行符后按空行切分出完整事件块；最后一段可能不完整，留在 buffer。
      buffer = buffer.replace(/\r\n/g, "\n");
      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex !== -1) {
        const rawBlock = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);

        const parsed = parseEventBlock(rawBlock);
        if (parsed) {
          yield parsed;
        }
        separatorIndex = buffer.indexOf("\n\n");
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

  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
    // 以 ":" 开头的注释行与其它字段按需忽略。
  }

  if (dataLines.length === 0) {
    return null;
  }
  return { event, data: dataLines.join("\n") };
}
