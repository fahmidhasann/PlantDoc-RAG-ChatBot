import { NextRequest, NextResponse } from 'next/server';
import { retrieveRelevantContext } from '@/lib/pinecone';
import { buildSearchQuery, generateGeminiDiagnosis } from '@/lib/gemini';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function formatPages(page?: number, pageEnd?: number): string {
  if (!page) return 'Page N/A';
  return pageEnd && pageEnd !== page ? `Pages ${page}–${pageEnd}` : `Page ${page}`;
}

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

    // The textbook is English: photos and non-English questions are first turned into an
    // English search line. Plain English questions are searched as typed (no extra Gemini call).
    const needsRewrite = Boolean(image) || /(?=\p{L})\P{Script=Latin}/u.test(query || '');
    const rewritten = needsRewrite
      ? await buildSearchQuery({ userQuery: query, imageBase64: image, imageMimeType: image ? mimeType || 'image/jpeg' : undefined })
      : '';
    const searchQuery =
      (image ? [query, rewritten].filter(Boolean).join(' — ') : rewritten || query) ||
      'Plant disease symptoms, identification and management';

    // 1. Retrieve grounded textbook passages via Pinecone / Knowledge Base
    const retrievedDocs = await retrieveRelevantContext(searchQuery, 6);

    const contextChunks = retrievedDocs
      .map(
        (doc, index) =>
          `[Excerpt ${index + 1} - ${formatPages(doc.page, doc.pageEnd)} | ${doc.chapter || ''} | ${doc.topic || ''}]\n${doc.content}`
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
      .filter((doc, i, all) => all.findIndex(d => d.page === doc.page) === i)
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
