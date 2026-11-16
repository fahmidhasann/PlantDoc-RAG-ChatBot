/**
 * GET /api/health — configuration and index readiness.
 *
 * `?deep=1` also queries Pinecone for live vector counts. That costs a control
 * plane round trip, so the default stays free and instant for uptime probes.
 */

import { NextRequest, NextResponse } from 'next/server';
import {
  ANSWER_MODELS,
  DENSE_DIMENSION,
  DENSE_INDEX,
  DENSE_MODEL,
  NAMESPACE,
  RERANK_MODEL,
  SPARSE_INDEX,
  SPARSE_MODEL,
} from '@/lib/config';
import { getPinecone, isPineconeConfigured } from '@/lib/retrieval';
import { isGeminiConfigured } from '@/lib/gemini';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

async function indexStats(name: string) {
  try {
    const stats = await getPinecone().index(name).describeIndexStats();
    return {
      ok: true,
      vectors: stats.totalRecordCount ?? 0,
      namespaces: Object.keys(stats.namespaces ?? {}),
    };
  } catch (err) {
    return { ok: false, error: (err as Error)?.message ?? 'unreachable' };
  }
}

export async function GET(req: NextRequest) {
  const deep = req.nextUrl.searchParams.get('deep') === '1';

  const payload: Record<string, unknown> = {
    status: 'healthy',
    service: 'PlantDoc — grounded plant pathology assistant',
    version: '3.0.0',
    config: {
      geminiKey: isGeminiConfigured(),
      pineconeKey: isPineconeConfigured(),
      answerModels: ANSWER_MODELS,
      denseModel: `${DENSE_MODEL} (${DENSE_DIMENSION}d)`,
      sparseModel: SPARSE_MODEL,
      rerankModel: RERANK_MODEL,
      namespace: NAMESPACE,
    },
    timestamp: new Date().toISOString(),
  };

  if (deep && isPineconeConfigured()) {
    const [dense, sparse] = await Promise.all([
      indexStats(DENSE_INDEX),
      indexStats(SPARSE_INDEX),
    ]);
    payload.indexes = { [DENSE_INDEX]: dense, [SPARSE_INDEX]: sparse };
  }

  return NextResponse.json(payload);
}
