import { Pinecone } from '@pinecone-database/pinecone';
import { searchPathologyKnowledge, PathologyChunk } from './knowledge-base';

export interface RetrievedDoc {
  id: string;
  score?: number;
  page?: number;
  pageEnd?: number;
  chapter?: string;
  topic?: string;
  content: string;
}

let pineconeClient: Pinecone | null = null;

// Reference lists, the back-of-book index and tables of contents are indexed too,
// but answers are grounded in the explanatory text only.
const ANSWERABLE_KINDS = ['body', 'glossary', 'front'];

// Must match scripts/ingest.mjs. Pinecone-hosted and multilingual, so Bangla questions
// still find the English textbook passages.
const EMBED_MODEL = 'llama-text-embed-v2';
const EMBED_DIMENSION = 768;

async function embedQuery(pinecone: Pinecone, query: string): Promise<number[]> {
  const res = await pinecone.inference.embed({
    model: EMBED_MODEL,
    inputs: [query],
    // `dimension` is numeric in the API; the SDK types parameters as strings.
    parameters: { inputType: 'query', truncate: 'END', dimension: EMBED_DIMENSION } as unknown as Record<string, string>,
  });
  const values = (res.data?.[0] as { values?: number[] } | undefined)?.values;
  if (!values?.length) throw new Error('Query embedding failed.');
  return values;
}

function getPineconeClient(): Pinecone | null {
  const apiKey = process.env.PINECONE_API_KEY;
  if (!apiKey) return null;

  if (!pineconeClient) {
    pineconeClient = new Pinecone({ apiKey });
  }
  return pineconeClient;
}

/**
 * Hybrid retrieval function:
 * 1. Checks if Pinecone is configured.
 * 2. If configured, embeds the query with Pinecone's hosted model and searches the full-textbook index
 *    (built by `npm run extract` + `npm run ingest`).
 * 3. If Pinecone is not configured or index returns empty, falls back to the built-in textbook knowledge base.
 */
export async function retrieveRelevantContext(query: string, topK: number = 6): Promise<RetrievedDoc[]> {
  const pinecone = getPineconeClient();
  const indexName = process.env.PINECONE_INDEX_NAME || 'plantdoc-book';

  if (pinecone) {
    try {
      const queryVector = await embedQuery(pinecone, query);
      const index = pinecone.index(indexName);

      let queryResponse = await index.query({
        vector: queryVector,
        topK,
        includeMetadata: true,
        filter: { kind: { $in: ANSWERABLE_KINDS } },
      });

      // Older indexes were built without the `kind` field; search them unfiltered.
      if (!queryResponse.matches?.length) {
        queryResponse = await index.query({ vector: queryVector, topK, includeMetadata: true });
      }

      if (queryResponse.matches && queryResponse.matches.length > 0) {
        return queryResponse.matches.map(match => ({
          id: match.id,
          score: match.score,
          page: (match.metadata?.page as number) || undefined,
          pageEnd: (match.metadata?.pageEnd as number) || undefined,
          chapter: (match.metadata?.chapter as string) || undefined,
          topic: (match.metadata?.topic as string) || undefined,
          content: (match.metadata?.text as string) || (match.metadata?.content as string) || ''
        }));
      }
    } catch (err) {
      console.warn('Pinecone vector search warning (falling back to curated knowledge base):', err);
    }
  }

  // Fallback to rich curated textbook pathology chunks
  const localMatches = searchPathologyKnowledge(query, topK);
  return localMatches.map(m => ({
    id: m.id,
    page: m.page,
    chapter: m.chapter,
    topic: m.topic,
    content: `${m.topic} (${m.pathogen})\nPage: ${m.page}\n${m.content}\nSymptoms:\n- ${m.symptoms.join('\n- ')}\nControls:\n- ${m.controls.join('\n- ')}`
  }));
}
