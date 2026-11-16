/** Shared shapes for retrieval, streaming and the chat UI. */

/** Metadata stored alongside every vector. Mirrors what the ingester writes. */
export interface PassageMetadata {
  /** Printed page number as it appears in the book — what citations quote. */
  page: number;
  /** Last printed page, when a passage runs across a page break. */
  pageEnd: number;
  /** Physical page in the PDF, for rendering the page image. */
  pdfPage: number;
  chapter: string;
  /** Nearest section or disease heading above the passage. */
  topic: string;
  /** body | references | glossary | index | toc | front | figure */
  kind: string;
  text: string;
  /**
   * Public path to the plate, for `kind: 'figure'` only, and only when the
   * book's licence allowed the image itself to be published. Figures are
   * indexed and cited either way; the picture is the optional part.
   */
  image?: string;
}

/** A retrieved passage, after fusion and reranking. */
export interface Passage extends PassageMetadata {
  id: string;
  /** Reranker score, 0-1. Absent when reranking was unavailable. */
  score?: number;
  /** Position in the fused candidate list, before reranking. */
  fusedRank: number;
  /** Which retrievers found it — shown in the UI as a retrieval badge. */
  retrievers: Array<'semantic' | 'lexical'>;
}

/** A passage as presented to the user, numbered to match inline [n] markers. */
export interface Source {
  n: number;
  id: string;
  page: number;
  pageEnd: number;
  chapter: string;
  topic: string;
  /** Trimmed quote shown in the source drawer. */
  excerpt: string;
  score?: number;
  retrievers: Array<'semantic' | 'lexical'>;
  /** Marks a source that came from a plate rather than running text. */
  isFigure?: boolean;
  /** Public path to the plate, when it was publishable. */
  image?: string;
}

/** One line of the NDJSON stream from /api/chat. */
export type ChatEvent =
  | { type: 'status'; stage: RetrievalStage; detail?: string }
  | { type: 'sources'; sources: Source[]; searchQuery: string }
  | { type: 'delta'; text: string }
  | { type: 'done'; model: string; elapsedMs: number }
  | { type: 'error'; message: string; recoverable: boolean };

export type RetrievalStage =
  | 'understanding'
  | 'searching'
  | 'ranking'
  | 'reading'
  | 'writing';

export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatRequest {
  message: string;
  history?: ChatTurn[];
  image?: { data: string; mimeType: string };
}
