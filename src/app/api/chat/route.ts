import { NextRequest, NextResponse } from 'next/server';
import { retrieveRelevantContext } from '@/lib/pinecone';
import { generateGeminiDiagnosis } from '@/lib/gemini';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { query, image, mimeType } = body;

    if (!query && !image) {
      return NextResponse.json(
        { error: 'Please provide a question or upload a leaf image.' },
        { status: 400 }
      );
    }

    const searchQuery = query || 'Plant disease symptoms, identification and management';

    // 1. Retrieve grounded textbook passages via Pinecone / Knowledge Base
    const retrievedDocs = await retrieveRelevantContext(searchQuery, 4);

    const contextChunks = retrievedDocs
      .map(
        (doc, index) =>
          `[Excerpt ${index + 1} - Page ${doc.page || 'N/A'} | ${doc.chapter || ''} | ${doc.topic || ''}]\n${doc.content}`
      )
      .join('\n\n---\n\n');

    // 2. Synthesize clinical diagnosis with Gemini 3.8 Flash
    const diagnosis = await generateGeminiDiagnosis({
      userQuery: query || 'Identify the disease affecting this plant and give treatment steps with textbook citations.',
      contextChunks,
      imageBase64: image,
      imageMimeType: mimeType || 'image/jpeg'
    });

    const sources = retrievedDocs
      .filter(doc => doc.page)
      .map(doc => ({
        page: doc.page,
        chapter: doc.chapter || 'Plant Pathology Textbook',
        topic: doc.topic || 'General Pathology'
      }));

    return NextResponse.json({
      success: true,
      answer: diagnosis,
      sources,
      retrievedCount: retrievedDocs.length
    });
  } catch (error: any) {
    console.error('PlantDoc Chat API Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'An error occurred while analyzing with PlantDoc.'
      },
      { status: 500 }
    );
  }
}
