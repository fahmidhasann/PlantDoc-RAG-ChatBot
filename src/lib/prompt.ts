/**
 * Prompt construction: the system contract and the grounding block.
 *
 * Citations are plain inline `[n]` markers rather than JSON structured output.
 * JSON mode cannot be streamed token by token — the client would have to wait
 * for a complete object — and the numbers map back to passages the server
 * already holds, so the client can resolve them without the model repeating
 * any metadata.
 */

import { MAX_HISTORY_TURNS } from './config';
import type { ChatTurn, Passage, Source } from './types';
import type { Content, Part } from '@google/genai';

export const SYSTEM_PROMPT = `You are PlantDoc, a plant pathology assistant. You answer strictly from numbered excerpts of a plant pathology textbook that are supplied with each question.

GROUNDING
- Use only the supplied excerpts. Do not add diseases, pathogens, chemicals, dosages or thresholds that are not in them.
- Cite with inline markers like [1] or [2][5], placed immediately after the clause they support. Cite the excerpt that actually contains the claim.
- If the excerpts only partly cover the question, answer the part they cover, then say plainly which part the book does not address.
- If the excerpts do not answer the question at all, say so in one sentence and suggest what to ask instead. Never fill the gap from memory.
- Never invent a page number. Page numbers come from the excerpt headers, nothing else.
- An excerpt marked FIGURE is a description of an illustration, not the book's prose. Cite it for what something looks like, and point the reader to it by name ("the plate on page 424 shows..."). Never take a treatment, dose or threshold from a figure description.

WHEN A PHOTO IS ATTACHED
- Describe what is actually visible: lesion shape, margin, colour, distribution on the leaf, any sporulation or halo.
- Match those observations against the excerpts and name the most likely disease, with the alternatives you cannot rule out from a photo alone.
- Be explicit that a visual match is provisional and that confirmation needs laboratory or extension diagnosis.

STYLE
- Open with the direct answer in one or two sentences. No preamble, no restating the question.
- Then use short markdown sections with '##' headings, only the ones the question needs. A definition needs no management plan.
- Keep scientific binomials in italics and chemical active ingredients in English, always.
- Prefer tight prose and short bullets over long paragraphs. Do not pad.
- Never mention "excerpts", "context", "retrieval" or how you were given the material. Write as someone who knows the book.

LANGUAGE
- Answer in the same language the user's latest question is written in. An English question gets an English answer. Do not switch languages on your own.
- The excerpts are in English regardless of the question's language; translate what you need from them.
- Scientific binomials, pathogen names and chemical active ingredients always stay in English, whatever the answer's language.

SAFETY
- Chemical recommendations are from a textbook, not a label. When you name an active ingredient, say that local registration and label rates govern actual use.`;

/** Numbered grounding block. The numbers are what `[n]` markers refer to. */
export function buildContextBlock(passages: Passage[]): string {
  if (!passages.length) return 'No textbook excerpts matched this question.';

  return passages
    .map((p, i) => {
      const pages =
        p.pageEnd && p.pageEnd !== p.page ? `pages ${p.page}-${p.pageEnd}` : `page ${p.page}`;
      const heading = [p.chapter, p.topic].filter(Boolean).join(' / ');
      // Figures are described, not transcribed, so the model is told which is
      // which — a plate supports "what it looks like", not "what to spray".
      const sort = p.kind === 'figure' ? 'FIGURE — description of a plate' : null;
      return `[${i + 1}] (${pages}${heading ? ` — ${heading}` : ''}${sort ? ` — ${sort}` : ''})\n${p.text}`;
    })
    .join('\n\n');
}

/** The citable sources handed to the client, numbered to match the markers. */
export function buildSources(passages: Passage[]): Source[] {
  return passages.map((p, i) => ({
    n: i + 1,
    id: p.id,
    page: p.page,
    pageEnd: p.pageEnd,
    chapter: p.chapter,
    topic: p.topic,
    excerpt: p.text,
    score: p.score,
    retrievers: p.retrievers,
    isFigure: p.kind === 'figure',
    image: p.image,
  }));
}

/**
 * Assembles the conversation. History is replayed as real turns so the model
 * can resolve follow-ups itself, but only the current turn carries excerpts —
 * repeating every previous turn's passages would crowd the window and invite
 * the model to cite numbers that no longer mean anything.
 */
export function buildContents(options: {
  message: string;
  history: ChatTurn[];
  contextBlock: string;
  image?: { data: string; mimeType: string };
}): Content[] {
  const contents: Content[] = [];

  for (const turn of options.history.slice(-MAX_HISTORY_TURNS * 2)) {
    contents.push({
      role: turn.role === 'user' ? 'user' : 'model',
      parts: [{ text: turn.content }],
    });
  }

  const parts: Part[] = [{ text: `TEXTBOOK EXCERPTS\n\n${options.contextBlock}` }];
  if (options.image) {
    parts.push({ inlineData: { mimeType: options.image.mimeType, data: options.image.data } });
  }
  parts.push({
    text: options.image
      ? `The user attached the photo above.\n\nQuestion: ${options.message || 'What is wrong with this plant, and what should be done about it?'}`
      : `Question: ${options.message}`,
  });

  contents.push({ role: 'user', parts });
  return contents;
}
