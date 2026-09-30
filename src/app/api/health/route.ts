import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    service: 'PlantDoc Multimodal Pathology AI',
    version: '2.0.0',
    geminiKeyConfigured: Boolean(process.env.GEMINI_API_KEY),
    pineconeKeyConfigured: Boolean(process.env.PINECONE_API_KEY),
    timestamp: new Date().toISOString()
  });
}
