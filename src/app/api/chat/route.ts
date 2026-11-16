/**
 * POST /api/chat — streams a grounded answer as newline-delimited JSON.
 *
 * Each line is one `ChatEvent`. The client renders progress while retrieval
 * runs, paints the sources the moment they are known, then appends answer text
 * as it arrives. Nothing here ever buffers a whole answer before replying: the
 * previous version did, and a three-minute silence is indistinguishable from a
 * crash to anyone waiting on it.
 *
 * Failures are reported as a terminal `error` event inside a 200 response.
 * A non-2xx status would make the browser's `fetch` body a plain error page,
 * which is exactly how the old version turned a Gemini timeout into
 * "Unexpected token 'A'" in the UI.
 */

import { NextRequest } from 'next/server';
import {
  ACCEPTED_IMAGE_TYPES,
  CONTEXT_PASSAGES,
  MAX_IMAGE_BYTES,
} from '@/lib/config';
import { buildSearchQuery, describeGeminiError, isGeminiConfigured, streamAnswer } from '@/lib/gemini';
import { buildContents, buildContextBlock, buildSources, SYSTEM_PROMPT } from '@/lib/prompt';
import { isPineconeConfigured, retrieve } from '@/lib/retrieval';
import type { ChatEvent, ChatRequest, ChatTurn } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
/** Retrieval plus generation fits comfortably; this is the ceiling, not the target. */
export const maxDuration = 60;

function sanitiseHistory(raw: unknown): ChatTurn[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter(
      (t): t is ChatTurn =>
        Boolean(t) &&
        (t.role === 'user' || t.role === 'assistant') &&
        typeof t.content === 'string' &&
        t.content.trim().length > 0
    )
    .map(t => ({ role: t.role, content: t.content.slice(0, 4000) }));
}

function validateImage(image: unknown): { data: string; mimeType: string } | null {
  if (!image || typeof image !== 'object') return null;
  const { data, mimeType } = image as { data?: unknown; mimeType?: unknown };
  if (typeof data !== 'string' || typeof mimeType !== 'string') return null;
  if (!ACCEPTED_IMAGE_TYPES.includes(mimeType)) {
    throw new Error(`Unsupported image type: ${mimeType}. Use JPEG, PNG or WebP.`);
  }
  // base64 encodes 3 bytes as 4 characters.
  const bytes = Math.floor((data.length * 3) / 4);
  if (bytes > MAX_IMAGE_BYTES) {
    throw new Error('That image is too large. Please use one under 3 MB.');
  }
  return { data, mimeType };
}

export async function POST(req: NextRequest) {
  let body: ChatRequest;
  try {
    body = await req.json();
  } catch {
    return jsonError('The request body was not valid JSON.');
  }

  const message = typeof body.message === 'string' ? body.message.trim() : '';
  const history = sanitiseHistory(body.history);

  let image: { data: string; mimeType: string } | null = null;
  try {
    image = validateImage(body.image);
  } catch (err) {
    return jsonError((err as Error).message);
  }

  if (!message && !image) {
    return jsonError('Ask a question or attach a photo.');
  }
  if (!isGeminiConfigured()) {
    return jsonError('GEMINI_API_KEY is not configured on the server.');
  }
  if (!isPineconeConfigured()) {
    return jsonError('PINECONE_API_KEY is not configured on the server.');
  }

  const encoder = new TextEncoder();
  const started = Date.now();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      let closed = false;
      const send = (event: ChatEvent) => {
        if (closed) return;
        controller.enqueue(encoder.encode(`${JSON.stringify(event)}\n`));
      };
      const finish = () => {
        if (closed) return;
        closed = true;
        controller.close();
      };

      try {
        // 1 — Resolve the turn into a standalone English search line.
        send({ type: 'status', stage: 'understanding' });
        const rewritten = await buildSearchQuery({
          message,
          history,
          image: image ?? undefined,
          signal: req.signal,
        });
        const searchQuery =
          [image ? message : '', rewritten].filter(Boolean).join(' — ') ||
          message ||
          'plant disease symptoms, identification and management';

        // 2 — Hybrid search, fusion and reranking.
        send({ type: 'status', stage: 'searching', detail: searchQuery });
        const { passages, reranked, retrievers } = await retrieve(searchQuery);

        send({
          type: 'status',
          stage: 'ranking',
          detail: reranked
            ? `${passages.length} passages reranked from ${retrievers.join(' + ')} search`
            : `${passages.length} passages from ${retrievers.join(' + ') || 'no'} search`,
        });

        if (!passages.length) {
          send({
            type: 'error',
            message:
              'Nothing in the indexed book matched that question. Try naming the crop and the symptom, or ask about a different topic.',
            recoverable: true,
          });
          finish();
          return;
        }

        send({
          type: 'sources',
          sources: buildSources(passages.slice(0, CONTEXT_PASSAGES)),
          searchQuery,
        });

        // 3 — Stream the grounded answer.
        send({ type: 'status', stage: 'writing' });
        const { model, chunks } = await streamAnswer({
          system: SYSTEM_PROMPT,
          contents: buildContents({
            message,
            history,
            contextBlock: buildContextBlock(passages),
            image: image ?? undefined,
          }),
          signal: req.signal,
        });

        for await (const text of chunks) {
          if (req.signal.aborted) break;
          send({ type: 'delta', text });
        }

        send({ type: 'done', model, elapsedMs: Date.now() - started });
      } catch (err) {
        if (req.signal.aborted) {
          finish();
          return;
        }
        console.error('[chat] turn failed:', err);
        const message = isPineconeFailure(err)
          ? 'The vector index could not be reached. Check PINECONE_API_KEY and that the index has been built.'
          : describeGeminiError(err);
        send({ type: 'error', message, recoverable: true });
      } finally {
        finish();
      }
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/x-ndjson; charset=utf-8',
      'Cache-Control': 'no-store, no-transform',
      // Defeats proxy buffering, which would hold the stream until it ends.
      'X-Accel-Buffering': 'no',
    },
  });
}

function isPineconeFailure(err: unknown): boolean {
  return /pinecone|index|embedding/i.test(String((err as Error)?.message || err));
}

/** Pre-stream validation failures still answer in the stream's own shape. */
function jsonError(message: string): Response {
  const event: ChatEvent = { type: 'error', message, recoverable: false };
  return new Response(`${JSON.stringify(event)}\n`, {
    status: 200,
    headers: { 'Content-Type': 'application/x-ndjson; charset=utf-8', 'Cache-Control': 'no-store' },
  });
}
