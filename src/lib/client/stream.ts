import type { ChatEvent } from '@/lib/types';

/**
 * Reads the NDJSON event stream from /api/chat.
 *
 * The server always replies 200 with events in the body, including for
 * failures, so there is no second error shape to parse. A line that will not
 * parse is skipped rather than thrown: a truncated final line should not
 * discard an answer the user has already read.
 */
export async function* readChatStream(
  response: Response,
  signal?: AbortSignal
): AsyncGenerator<ChatEvent> {
  if (!response.body) {
    yield { type: 'error', message: 'The server sent an empty response.', recoverable: true };
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      if (signal?.aborted) return;
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      let newline: number;
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        try {
          yield JSON.parse(line) as ChatEvent;
        } catch {
          // Ignore a malformed line and keep reading.
        }
      }
    }

    const tail = buffer.trim();
    if (tail) {
      try {
        yield JSON.parse(tail) as ChatEvent;
      } catch {
        /* ignore */
      }
    }
  } finally {
    reader.releaseLock();
  }
}
