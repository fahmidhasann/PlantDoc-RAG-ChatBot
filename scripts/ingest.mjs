/**
 * PlantDoc Knowledge Ingestion Script
 * Embeds pathology chunks using Gemini Embeddings (text-embedding-004, 768 dims)
 * and upserts into Pinecone Serverless Vector DB.
 *
 * Usage:
 *   node scripts/ingest.mjs
 */

import { Pinecone } from '@pinecone-database/pinecone';
import fs from 'node:fs';
import path from 'node:path';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'plantdoc';

if (!GEMINI_API_KEY || !PINECONE_API_KEY) {
  console.error('❌ Error: GEMINI_API_KEY and PINECONE_API_KEY environment variables are required.');
  console.log('Example:');
  console.log('  export GEMINI_API_KEY="AIzaSy..."');
  console.log('  export PINECONE_API_KEY="pcsk_..."');
  process.exit(1);
}

const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });

async function getEmbedding(text) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${GEMINI_API_KEY}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'models/text-embedding-004',
      content: { parts: [{ text: text.slice(0, 8000) }] }
    })
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Embedding failed (${res.status}): ${err}`);
  }
  const data = await res.json();
  return data.embedding.values;
}

async function main() {
  console.log('🌿 PlantDoc Vector Ingestion Pipeline Starting...');

  // 1. Ensure Pinecone index exists
  const existingIndexes = await pinecone.listIndexes();
  const indexExists = existingIndexes.indexes?.some(idx => idx.name === PINECONE_INDEX_NAME);

  if (!indexExists) {
    console.log(`📦 Creating Serverless Pinecone index "${PINECONE_INDEX_NAME}" (dim: 768, metric: cosine)...`);
    await pinecone.createIndex({
      name: PINECONE_INDEX_NAME,
      dimension: 768,
      metric: 'cosine',
      spec: {
        serverless: {
          cloud: 'aws',
          region: 'us-east-1'
        }
      }
    });
    console.log('⏳ Waiting for index to initialize...');
    await new Promise(r => setTimeout(r, 10000));
  } else {
    console.log(`✅ Index "${PINECONE_INDEX_NAME}" exists.`);
  }

  const index = pinecone.index(PINECONE_INDEX_NAME);

  // 2. Load textbook chunks
  // Fallback to sample pathology data if chunks.json doesn't exist
  let chunks = [];
  const chunksPath = path.resolve('data/processed/chunks.json');
  if (fs.existsSync(chunksPath)) {
    chunks = JSON.parse(fs.readFileSync(chunksPath, 'utf8'));
    console.log(`📖 Loaded ${chunks.length} chunks from ${chunksPath}`);
  } else {
    console.log('ℹ️ No data/processed/chunks.json found, ingesting core reference pathology chapters...');
    chunks = [
      {
        id: 'chunk_late_blight',
        page: 421,
        chapter: 'Oomycetes',
        topic: 'Late Blight of Potato/Tomato',
        text: 'Late blight caused by Phytophthora infestans produces rapid leaf necrosis, water-soaked black spots, and white sporulation underneath. Controlled with Mancozeb and Metalaxyl.'
      },
      {
        id: 'chunk_bacterial_canker',
        page: 638,
        chapter: 'Prokaryotes',
        topic: 'Bacterial Canker of Tomato',
        text: 'Clavibacter michiganensis causes unilateral wilting, bird-eye spots on fruit, and vascular browning. Controlled with copper bactericides and certified seed.'
      },
      {
        id: 'chunk_rice_blast',
        page: 495,
        chapter: 'Ascomycetes',
        topic: 'Rice Blast',
        text: 'Magnaporthe oryzae produces spindle-shaped lesions and neck rot. Managed with resistant varieties and tricyclazole foliar application.'
      },
      {
        id: 'chunk_powdery_mildew',
        page: 462,
        chapter: 'Ascomycetes',
        topic: 'Powdery Mildew',
        text: 'Superficial white powdery fungal colonies on leaf surface. Controlled with wettable sulfur, potassium bicarbonate, and triazoles.'
      }
    ];
  }

  console.log(`🚀 Ingesting and embedding ${chunks.length} chunks with Gemini text-embedding-004...`);

  const batchSize = 10;
  for (let i = 0; i < chunks.length; i += batchSize) {
    const batch = chunks.slice(i, i + batchSize);
    const vectors = [];

    for (const item of batch) {
      const textToEmbed = item.text || item.content || '';
      if (!textToEmbed) continue;
      const embedding = await getEmbedding(textToEmbed);
      vectors.push({
        id: String(item.id || `doc_${i}`),
        values: embedding,
        metadata: {
          page: Number(item.page || 0),
          chapter: String(item.chapter || ''),
          topic: String(item.topic || ''),
          text: textToEmbed.slice(0, 1000)
        }
      });
      // Small pause to respect rate limits
      await new Promise(r => setTimeout(r, 150));
    }

    if (vectors.length > 0) {
      await index.upsert(vectors);
      console.log(`✅ Upserted ${i + vectors.length} / ${chunks.length} chunks`);
    }
  }

  console.log('🎉 Ingestion complete! PlantDoc is ready for production RAG queries.');
}

main().catch(err => {
  console.error('Fatal ingestion error:', err);
  process.exit(1);
});
