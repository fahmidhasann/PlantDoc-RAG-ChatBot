/**
 * Hybrid retrieval over the indexed book.
 *
 *   question
 *      ├─ dense  (llama-text-embed-v2)       → semantic neighbours
 *      └─ sparse (pinecone-sparse-english-v0) → exact lexical matches
 *              ↓ reciprocal-rank fusion
 *        reranker (bge-reranker-v2-m3)
 *              ↓ score floor
 *          grounding passages
 *
 * Why both retrievers: a plant-pathology book is full of tokens that carry
 * their whole meaning in their spelling — *Phytophthora infestans*, mefenoxam,
 * Xanthomonas campestris pv. vesicatoria. Dense vectors blur those into
 * "oomycete-ish" neighbourhoods; a sparse index matches them exactly. Dense
 * alone also misses nothing-alike phrasings, and sparse alone cannot bridge
 * Bangla questions to English prose. Fusing the two ranks covers both.
 *
 * Every stage degrades rather than throws: if the sparse index is missing the
 * dense results still answer, and if the rerank quota is spent the fused order
 * is used as-is.
 */

import { Pinecone } from '@pinecone-database/pinecone';
import {
  ANSWERABLE_KINDS,
  CANDIDATES_PER_RETRIEVER,
  CONTEXT_PASSAGES,
  DENSE_DIMENSION,
  DENSE_INDEX,
  DENSE_MODEL,
  MIN_RERANK_SCORE,
  NAMESPACE,
  RERANK_CANDIDATES,
  RERANK_MODEL,
  RETRIEVAL_TIMEOUT_MS,
  RRF_K,
  SPARSE_INDEX,
  SPARSE_MODEL,
} from './config';
import type { Passage, PassageMetadata } from './types';

let client: Pinecone | null = null;

export function getPinecone(): Pinecone {
  const apiKey = process.env.PINECONE_API_KEY;
  if (!apiKey) throw new Error('PINECONE_API_KEY is not set.');
  if (!client) client = new Pinecone({ apiKey });
  return client;
}

export function isPineconeConfigured(): boolean {
  return Boolean(process.env.PINECONE_API_KEY);
}

/** Reject a hung upstream call instead of letting it eat the function budget. */
function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)
    ),
  ]);
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

interface SparseVector {
  indices: number[];
  values: number[];
}

export async function embedDenseQuery(pc: Pinecone, text: string): Promise<number[]> {
  const res = await pc.inference.embed({
    model: DENSE_MODEL,
    inputs: [text],
    // The SDK types parameters as Record<string, string>; `dimension` is numeric
    // on the wire and `inputType` is the one key the client snake-cases for us.
    parameters: {
      inputType: 'query',
      truncate: 'END',
      dimension: DENSE_DIMENSION,
    } as unknown as Record<string, string>,
  });
  const values = (res.data?.[0] as { values?: number[] } | undefined)?.values;
  if (!values?.length) throw new Error('Dense query embedding came back empty.');
  return values;
}

export async function embedSparseQuery(pc: Pinecone, text: string): Promise<SparseVector> {
  const res = await pc.inference.embed({
    model: SPARSE_MODEL,
    inputs: [text],
    parameters: { inputType: 'query', truncate: 'END' },
  });
  const row = res.data?.[0] as
    | { sparseIndices?: number[]; sparseValues?: number[] }
    | undefined;
  if (!row?.sparseIndices?.length || !row.sparseValues?.length) {
    throw new Error('Sparse query embedding came back empty.');
  }
  return { indices: row.sparseIndices, values: row.sparseValues };
}

// ---------------------------------------------------------------------------
// Candidate retrieval
// ---------------------------------------------------------------------------

interface Candidate {
  id: string;
  metadata: PassageMetadata;
  retriever: 'semantic' | 'lexical';
  rank: number;
}

const KIND_FILTER = { kind: { $in: [...ANSWERABLE_KINDS] } };

function toMetadata(raw: Record<string, unknown> | undefined): PassageMetadata | null {
  const text = (raw?.text as string) || '';
  if (!text) return null;
  return {
    page: Number(raw?.page) || 0,
    pageEnd: Number(raw?.pageEnd) || Number(raw?.page) || 0,
    pdfPage: Number(raw?.pdfPage) || 0,
    chapter: (raw?.chapter as string) || '',
    topic: (raw?.topic as string) || '',
    kind: (raw?.kind as string) || 'body',
    text,
  };
}

async function searchDense(pc: Pinecone, query: string): Promise<Candidate[]> {
  const vector = await embedDenseQuery(pc, query);
  const res = await pc
    .index(DENSE_INDEX)
    .namespace(NAMESPACE)
    .query({
      vector,
      topK: CANDIDATES_PER_RETRIEVER,
      includeMetadata: true,
      filter: KIND_FILTER,
    });
  return (res.matches || []).flatMap((m, rank) => {
    const metadata = toMetadata(m.metadata as Record<string, unknown>);
    return metadata ? [{ id: m.id, metadata, retriever: 'semantic' as const, rank }] : [];
  });
}

async function searchSparse(pc: Pinecone, query: string): Promise<Candidate[]> {
  const sparseVector = await embedSparseQuery(pc, query);
  const res = await pc
    .index(SPARSE_INDEX)
    .namespace(NAMESPACE)
    .query({
      // A sparse index is queried with a zero-length dense vector plus the
      // sparse one; the SDK requires `vector` to be present on the union.
      vector: [],
      sparseVector,
      topK: CANDIDATES_PER_RETRIEVER,
      includeMetadata: true,
      filter: KIND_FILTER,
    });
  return (res.matches || []).flatMap((m, rank) => {
    const metadata = toMetadata(m.metadata as Record<string, unknown>);
    return metadata ? [{ id: m.id, metadata, retriever: 'lexical' as const, rank }] : [];
  });
}

// ---------------------------------------------------------------------------
// Fusion
// ---------------------------------------------------------------------------

interface Fused {
  id: string;
  metadata: PassageMetadata;
  retrievers: Array<'semantic' | 'lexical'>;
  rrf: number;
}

/**
 * Reciprocal-rank fusion. Combines ranked lists without needing their scores to
 * be comparable — cosine similarity and a sparse dot product live on completely
 * different scales, so blending the ranks is the only honest way to merge them.
 */
export function fuse(lists: Candidate[][], k = RRF_K): Fused[] {
  const byId = new Map<string, Fused>();
  for (const list of lists) {
    for (const c of list) {
      const existing = byId.get(c.id);
      const contribution = 1 / (k + c.rank + 1);
      if (existing) {
        existing.rrf += contribution;
        if (!existing.retrievers.includes(c.retriever)) existing.retrievers.push(c.retriever);
      } else {
        byId.set(c.id, {
          id: c.id,
          metadata: c.metadata,
          retrievers: [c.retriever],
          rrf: contribution,
        });
      }
    }
  }
  return [...byId.values()].sort((a, b) => b.rrf - a.rrf);
}

// ---------------------------------------------------------------------------
// Reranking
// ---------------------------------------------------------------------------

/**
 * Cross-encoder reranking. The retrievers compare a query vector to a passage
 * vector computed independently; the reranker reads the question and the
 * passage together, which is far better at telling "mentions late blight" from
 * "explains how to manage late blight".
 *
 * Returns null when reranking is unavailable (the Starter plan allows 500
 * requests a month) so the caller can fall back to fused order.
 */
async function rerank(
  pc: Pinecone,
  query: string,
  candidates: Fused[]
): Promise<Map<string, number> | null> {
  try {
    const res = await pc.inference.rerank({
      model: RERANK_MODEL,
      query,
      // bge-reranker-v2-m3 allows 1024 tokens per query+document pair, so the
      // passage is clipped well short of that before it is sent.
      documents: candidates.map(c => ({
        id: c.id,
        text: `${c.metadata.topic}. ${c.metadata.text}`.slice(0, 2200),
      })),
      rankFields: ['text'],
      topN: CONTEXT_PASSAGES,
      returnDocuments: false,
      parameters: { truncate: 'END' },
    });
    const scores = new Map<string, number>();
    for (const row of res.data || []) {
      const candidate = candidates[row.index];
      if (candidate) scores.set(candidate.id, row.score);
    }
    return scores.size ? scores : null;
  } catch (err) {
    console.warn('[retrieval] rerank unavailable, using fused order:', err);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

export interface RetrievalResult {
  passages: Passage[];
  /** True when reranking ran; surfaced in the UI so the pipeline is legible. */
  reranked: boolean;
  /** Which retrievers actually returned anything. */
  retrievers: Array<'semantic' | 'lexical'>;
}

export async function retrieve(query: string): Promise<RetrievalResult> {
  const pc = getPinecone();

  // Both retrievers run concurrently; either may fail without sinking the turn.
  const [denseResult, sparseResult] = await Promise.allSettled([
    withTimeout(searchDense(pc, query), RETRIEVAL_TIMEOUT_MS, 'dense search'),
    withTimeout(searchSparse(pc, query), RETRIEVAL_TIMEOUT_MS, 'sparse search'),
  ]);

  const lists: Candidate[][] = [];
  const retrievers: Array<'semantic' | 'lexical'> = [];

  if (denseResult.status === 'fulfilled' && denseResult.value.length) {
    lists.push(denseResult.value);
    retrievers.push('semantic');
  } else if (denseResult.status === 'rejected') {
    console.warn('[retrieval] dense search failed:', denseResult.reason);
  }

  if (sparseResult.status === 'fulfilled' && sparseResult.value.length) {
    lists.push(sparseResult.value);
    retrievers.push('lexical');
  } else if (sparseResult.status === 'rejected') {
    console.warn('[retrieval] sparse search failed:', sparseResult.reason);
  }

  if (!lists.length) {
    if (denseResult.status === 'rejected') throw denseResult.reason;
    return { passages: [], reranked: false, retrievers };
  }

  const fused = fuse(lists).slice(0, RERANK_CANDIDATES);
  const scores = await rerank(pc, query, fused);

  let passages: Passage[] = fused.map((f, i) => ({
    ...f.metadata,
    id: f.id,
    fusedRank: i,
    retrievers: f.retrievers,
    score: scores?.get(f.id),
  }));

  if (scores) {
    passages = passages
      .filter(p => p.score !== undefined && p.score >= MIN_RERANK_SCORE)
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }

  return {
    passages: passages.slice(0, CONTEXT_PASSAGES),
    reranked: Boolean(scores),
    retrievers,
  };
}
