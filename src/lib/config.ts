/**
 * Single source of truth for model IDs, index names and retrieval tuning.
 *
 * Everything that both the ingestion scripts and the running app must agree on
 * lives here, so an index can never be built with one embedding model and
 * queried with another.
 */

// ---------------------------------------------------------------------------
// Vector store
// ---------------------------------------------------------------------------

/** Dense semantic index. Multilingual, so Bangla questions match English prose. */
export const DENSE_INDEX = process.env.PINECONE_DENSE_INDEX || 'plantdoc-dense';

/** Sparse lexical index. Exact term matching for binomials, actives, cultivars. */
export const SPARSE_INDEX = process.env.PINECONE_SPARSE_INDEX || 'plantdoc-sparse';

/** One namespace per book edition keeps corpora swappable without a new index. */
export const NAMESPACE = process.env.PINECONE_NAMESPACE || 'default';

export const DENSE_MODEL = 'llama-text-embed-v2';
export const DENSE_DIMENSION = 1024;
export const SPARSE_MODEL = 'pinecone-sparse-english-v0';

/** Only model available for reranking on the Pinecone Starter plan. */
export const RERANK_MODEL = 'bge-reranker-v2-m3';

/** Pinecone Starter allows 5 serverless indexes, all in aws/us-east-1. */
export const INDEX_CLOUD = 'aws';
export const INDEX_REGION = 'us-east-1';

// ---------------------------------------------------------------------------
// Retrieval tuning
// ---------------------------------------------------------------------------

/** Candidates pulled from each retriever before fusion. */
export const CANDIDATES_PER_RETRIEVER = 30;

/** Candidates kept after reciprocal-rank fusion and sent to the reranker. */
export const RERANK_CANDIDATES = 24;

/** Passages that actually reach the model. */
export const CONTEXT_PASSAGES = 8;

/** Standard RRF damping constant; 60 is the value from the original paper. */
export const RRF_K = 60;

/**
 * Reranker scores are normalised 0-1. Passages below this are dropped so a
 * question the book does not answer yields an honest "not covered" instead of
 * six loosely-related paragraphs.
 */
export const MIN_RERANK_SCORE = 0.02;

/**
 * Chunk kinds that may ground an answer. The whole book is indexed — including
 * the bibliography and the back-of-book index — but those are navigation aids,
 * not explanations, so they never crowd out real prose.
 */
export const ANSWERABLE_KINDS = ['body', 'glossary', 'front', 'figure'] as const;

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/**
 * Answer models, tried in order.
 *
 * Verified against the Gemini model catalogue on 2026-11-16. The lite models
 * lead deliberately: they default to minimal thinking, which is what a grounded
 * extraction-and-synthesis task wants. Heavier Flash models reason for longer
 * without reading the passages any better, and burn a much smaller daily quota.
 */
export const ANSWER_MODELS = [
  'gemini-3.5-flash-lite',
  'gemini-3.1-flash-lite',
  'gemini-3.6-flash',
  'gemini-3.8-flash',
] as const;

/** Rewriting a follow-up or captioning a leaf photo is a small, literal job. */
export const UTILITY_MODELS = ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite'] as const;

/**
 * gemini-3.8-flash rejects MINIMAL; its floor is LOW. Everything else accepts
 * MINIMAL. Keeping this as data avoids a 400 that would look like a quota error.
 */
export const MIN_THINKING_LEVEL: Record<string, 'minimal' | 'low'> = {
  'gemini-3.8-flash': 'low',
};

/**
 * Thinking tokens are drawn from maxOutputTokens, so this is a combined budget.
 * Generous enough that a thorough answer is never truncated mid-sentence.
 */
export const ANSWER_MAX_TOKENS = 3072;
export const UTILITY_MAX_TOKENS = 256;

/**
 * Two deadlines, because they guard different failures.
 *
 * FIRST_TOKEN is how long a model gets to say anything at all. Past that it is
 * wedged or throttled and the next model in the chain should take over — and
 * since nothing has been sent yet, the swap is invisible.
 *
 * ANSWER is the ceiling on the whole stream. A single deadline covering both
 * cut long answers off mid-sentence: these models think for ten seconds and
 * then emit the entire reply at once, so "slow to start" and "slow overall"
 * look identical until the text arrives.
 *
 * Vercel kills the function at 60s; retrieval is bounded separately above.
 */
export const FIRST_TOKEN_TIMEOUT_MS = 20_000;
export const ANSWER_TIMEOUT_MS = 45_000;
export const UTILITY_TIMEOUT_MS = 8_000;
export const RETRIEVAL_TIMEOUT_MS = 12_000;

// ---------------------------------------------------------------------------
// Uploads
// ---------------------------------------------------------------------------

/**
 * Vercel caps a serverless request body at 4.5 MB and base64 inflates by ~33%,
 * so the raw image ceiling is 3 MB. The client downscales before it ever gets
 * here; this is the backstop.
 */
export const MAX_IMAGE_BYTES = 3 * 1024 * 1024;
export const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

/** Turns of history replayed to the model. Keeps the prompt bounded. */
export const MAX_HISTORY_TURNS = 6;
