import { Pinecone } from '@pinecone-database/pinecone';
import { generateGeminiEmbedding } from './gemini';
import { searchPathologyKnowledge, PathologyChunk } from './knowledge-base';

export interface RetrievedDoc {
  id: string;
  score?: number;
  page?: number;
  chapter?: string;
  topic?: string;
  content: string;
}

let pineconeClient: Pinecone | null = null;

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
 * 2. If configured, generates Gemini embedding and performs vector search against the 948-page textbook index.
 * 3. If Pinecone is not configured or index returns empty, falls back to the built-in textbook knowledge base.
 */
export async function retrieveRelevantContext(query: string, topK: number = 4): Promise<RetrievedDoc[]> {
  const pinecone = getPineconeClient();
  const indexName = process.env.PINECONE_INDEX_NAME || 'plantdoc';

  if (pinecone && process.env.GEMINI_API_KEY) {
    try {
      const queryVector = await generateGeminiEmbedding(query);
      const index = pinecone.index(indexName);

      const queryResponse = await index.query({
        vector: queryVector,
        topK,
        includeMetadata: true,
      });

      if (queryResponse.matches && queryResponse.matches.length > 0) {
        return queryResponse.matches.map(match => ({
          id: match.id,
          score: match.score,
          page: (match.metadata?.page as number) || undefined,
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
