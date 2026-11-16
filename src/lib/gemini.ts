/**
 * Gemini client: streaming answers, query rewriting and photo reading.
 *
 * Three things here exist because of concrete failures in the previous version:
 *
 * 1. Thinking tokens are drawn from `maxOutputTokens`. Gemini 3.x Flash models
 *    default to MEDIUM thinking, so a 4k budget could be spent entirely on
 *    reasoning and return an empty answer. Every call below pins the thinking
 *    level to the model's documented floor.
 * 2. Nothing had a timeout, so a slow model could hold the request open until
 *    the platform killed it. Every call now carries an AbortSignal.
 * 3. Model fallback used to happen after bytes were already on the wire. The
 *    stream here is only committed once a model has produced real text, so a
 *    failing model is replaced invisibly instead of truncating the answer.
 */

import { GoogleGenAI, ThinkingLevel, type Content, type Part } from '@google/genai';
import {
  ANSWER_MAX_TOKENS,
  ANSWER_MODELS,
  ANSWER_TIMEOUT_MS,
  MIN_THINKING_LEVEL,
  UTILITY_MAX_TOKENS,
  UTILITY_MODELS,
  UTILITY_TIMEOUT_MS,
} from './config';

let client: GoogleGenAI | null = null;

function getClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set.');
  if (!client) client = new GoogleGenAI({ apiKey });
  return client;
}

export function isGeminiConfigured(): boolean {
  return Boolean(process.env.GEMINI_API_KEY);
}

/** gemini-3.8-flash has no MINIMAL level; its floor is LOW. */
function thinkingFor(model: string): ThinkingLevel {
  return MIN_THINKING_LEVEL[model] === 'low' ? ThinkingLevel.LOW : ThinkingLevel.MINIMAL;
}

function deadline(ms: number, parent?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(ms);
  return parent ? AbortSignal.any([parent, timeout]) : timeout;
}

/** Classifies an upstream failure so the caller can tell the user something true. */
export function describeGeminiError(err: unknown): string {
  const message = String((err as Error)?.message || err);
  if (/leaked|API key not valid|API_KEY_INVALID/i.test(message)) {
    return 'The Gemini API key was rejected. Generate a fresh one at aistudio.google.com and update GEMINI_API_KEY.';
  }
  if (/429|RESOURCE_EXHAUSTED|quota/i.test(message)) {
    return 'Every Gemini model on this key has hit its free-tier limit for now. Daily quotas reset at midnight Pacific.';
  }
  if (/abort|timeout/i.test(message)) {
    return 'Gemini took too long to respond and the request was cancelled.';
  }
  return 'Gemini could not be reached.';
}

// ---------------------------------------------------------------------------
// Streaming answers
// ---------------------------------------------------------------------------

export interface AnswerStream {
  model: string;
  /** Yields answer text only; thought parts are filtered out upstream. */
  chunks: AsyncGenerator<string>;
}

/**
 * Starts an answer, falling through the model chain until one actually emits
 * text. The first chunk is awaited before the generator is handed back, so a
 * 404 on a retired model, a 429, or an empty MAX_TOKENS response is swallowed
 * and retried on the next model with nothing shown to the user.
 */
export async function streamAnswer(options: {
  system: string;
  contents: Content[];
  signal?: AbortSignal;
}): Promise<AnswerStream> {
  const ai = getClient();
  let lastError: unknown = new Error('No Gemini model produced an answer.');

  for (const model of ANSWER_MODELS) {
    try {
      const stream = await ai.models.generateContentStream({
        model,
        contents: options.contents,
        config: {
          systemInstruction: options.system,
          temperature: 0.25,
          topP: 0.95,
          maxOutputTokens: ANSWER_MAX_TOKENS,
          thinkingConfig: { thinkingLevel: thinkingFor(model), includeThoughts: false },
          abortSignal: deadline(ANSWER_TIMEOUT_MS, options.signal),
        },
      });

      const iterator = stream[Symbol.asyncIterator]();

      // Pull until the first non-empty text chunk. A model that only ever
      // yields thought parts, or finishes on MAX_TOKENS with nothing to say,
      // falls through to the next candidate.
      let first: string | null = null;
      let exhausted = false;
      while (first === null) {
        const next = await iterator.next();
        if (next.done) {
          exhausted = true;
          break;
        }
        const text = next.value?.text;
        if (text) first = text;
      }

      if (exhausted || first === null) {
        lastError = new Error(`${model} returned no text.`);
        console.warn(`[gemini] ${model} produced an empty answer; trying the next model.`);
        continue;
      }

      const committed = first;
      async function* chunks() {
        yield committed;
        while (true) {
          const next = await iterator.next();
          if (next.done) return;
          const text = next.value?.text;
          if (text) yield text;
        }
      }

      return { model, chunks: chunks() };
    } catch (err) {
      lastError = err;
      console.warn(`[gemini] ${model} failed:`, (err as Error)?.message || err);
      // An invalid key or an aborted request will fail identically on every
      // model, so stop rather than burn four round trips proving it.
      const message = String((err as Error)?.message || err);
      if (/API_KEY_INVALID|API key not valid|leaked/i.test(message)) break;
      if (options.signal?.aborted) break;
    }
  }

  throw lastError;
}

// ---------------------------------------------------------------------------
// Small utility calls
// ---------------------------------------------------------------------------

/** Runs a short, literal prompt on the cheapest model that answers. */
async function utilityCall(parts: Part[], signal?: AbortSignal): Promise<string> {
  const ai = getClient();
  for (const model of UTILITY_MODELS) {
    try {
      const res = await ai.models.generateContent({
        model,
        contents: [{ role: 'user', parts }],
        config: {
          temperature: 0,
          maxOutputTokens: UTILITY_MAX_TOKENS,
          thinkingConfig: { thinkingLevel: thinkingFor(model), includeThoughts: false },
          abortSignal: deadline(UTILITY_TIMEOUT_MS, signal),
        },
      });
      const text = res.text?.trim();
      if (text) return text;
    } catch (err) {
      console.warn(`[gemini] utility call on ${model} failed:`, (err as Error)?.message || err);
    }
  }
  return '';
}

/**
 * Turns the user's turn into a standalone English search line.
 *
 * Three jobs at once, because they are the same job: resolve a follow-up
 * against the conversation ("and what about the stems?"), translate a
 * non-English question into the language the book is written in, and name what
 * is visible in an uploaded photo. Retrieval quality depends almost entirely on
 * this line, so it is worth one cheap call.
 *
 * Returns "" when the rewrite is unnecessary or unavailable; the caller then
 * searches with the raw question.
 */
export async function buildSearchQuery(options: {
  message: string;
  history?: Array<{ role: string; content: string }>;
  image?: { data: string; mimeType: string };
  signal?: AbortSignal;
}): Promise<string> {
  const { message, history = [], image } = options;

  const isLatin = !/(?=\p{L})\P{Script=Latin}/u.test(message);
  const isFollowUp = history.length > 0;
  // A plain English opening question is already the best possible search line.
  if (isLatin && !isFollowUp && !image) return '';

  const transcript = history
    .slice(-4)
    .map(t => `${t.role === 'user' ? 'User' : 'Assistant'}: ${t.content.slice(0, 400)}`)
    .join('\n');

  const instruction = [
    'You write search queries for an English plant pathology textbook.',
    transcript && `Conversation so far:\n${transcript}`,
    image
      ? 'The user attached a photo of a plant. Identify the crop, the most likely disease and pathogen, and the visible symptoms.'
      : '',
    `Latest user message: "${message}"`,
    '',
    'Reply with ONE English line that names the crop, the disease and pathogen if identifiable, and the symptoms or topic being asked about.',
    'Resolve any pronouns or references to earlier turns so the line stands alone.',
    'Output the line only — no preamble, no quotes, no explanation.',
  ]
    .filter(Boolean)
    .join('\n');

  const parts: Part[] = [];
  if (image) parts.push({ inlineData: { mimeType: image.mimeType, data: image.data } });
  parts.push({ text: instruction });

  const text = await utilityCall(parts, options.signal);
  return text ? text.split('\n')[0].slice(0, 320) : '';
}
