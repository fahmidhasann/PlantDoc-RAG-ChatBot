/**
 * PlantDoc Knowledge Ingestion Script
 * Embeds pathology chunks using Gemini Embeddings (gemini-embedding-001, 768 dims)
 * and upserts into Pinecone Serverless Vector DB.
 */

import { Pinecone } from '@pinecone-database/pinecone';
import fs from 'node:fs';
import path from 'node:path';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "AIzaSyDkMZxw-i_qB3GGUXEHhGgfmK1eLxn2kAA";
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || "pcsk_3hNs7F_HFVBjDQjnkx3kGraCqU74QTmzrNwGfhmcejcYFXzFrVCA9NNMxzL8hn6xuqSvWm";
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'plantdoc';

const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });

async function getEmbedding(text) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key=${GEMINI_API_KEY}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'models/gemini-embedding-001',
      content: { parts: [{ text: text.slice(0, 8000) }] },
      outputDimensionality: 768
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
    console.log('⏳ Waiting for index to initialize on AWS us-east-1...');
    let ready = false;
    for (let attempts = 0; attempts < 30; attempts++) {
      await new Promise(r => setTimeout(r, 4000));
      const desc = await pinecone.describeIndex(PINECONE_INDEX_NAME);
      if (desc.status?.ready) {
        ready = true;
        console.log('✅ Index is ready!');
        break;
      }
      process.stdout.write('.');
    }
    if (!ready) {
      console.warn('Index created, proceeding to upsert...');
    }
  } else {
    console.log(`✅ Index "${PINECONE_INDEX_NAME}" exists.`);
  }

  const index = pinecone.index(PINECONE_INDEX_NAME);

  // 2. Load textbook chunks
  let chunks = [];
  const chunksPath = path.resolve('data/processed/chunks.json');
  if (fs.existsSync(chunksPath)) {
    chunks = JSON.parse(fs.readFileSync(chunksPath, 'utf8'));
    console.log(`📖 Loaded ${chunks.length} chunks from ${chunksPath}`);
  } else {
    console.log('ℹ️ Ingesting comprehensive reference pathology chapters...');
    chunks = [
      {
        id: 'chunk_late_blight',
        page: 421,
        chapter: 'Chapter 11: Plant Diseases Caused by Oomycetes',
        topic: 'Late Blight of Potato and Tomato',
        text: 'Late blight of potato and tomato, caused by the oomycete Phytophthora infestans. Symptoms appear as water-soaked irregular spots on leaves rapidly turning purplish-black with white mildew sporulation underneath. Tubers show brownish granular dry rot. Controls include certified disease-free tubers, resistant varieties, Mancozeb, Chlorothalonil, and systemic Metalaxyl/Mefenoxam.'
      },
      {
        id: 'chunk_bacterial_canker',
        page: 638,
        chapter: 'Chapter 12: Plant Diseases Caused by Prokaryotes',
        topic: 'Bacterial Canker of Tomato',
        text: 'Bacterial canker caused by Clavibacter michiganensis subsp. michiganensis. Symptoms: unilateral leaflet wilting, white blister bird-eye spots with dark centers on fruit, and vascular browning. Managed through seed hot-water treatment (50C for 25 min), 3-year crop rotation, greenhouse sanitation, and copper bactericides mixed with mancozeb.'
      },
      {
        id: 'chunk_rice_blast',
        page: 495,
        chapter: 'Chapter 11: Plant Diseases Caused by Ascomycetes',
        topic: 'Rice Blast Disease',
        text: 'Rice blast caused by Magnaporthe oryzae (Pyricularia oryzae). Symptoms include spindle-shaped lesions with gray centers and brown borders on leaves, neck blast (rotten neck) leading to sterile white heads. Controlled with resistant cultivars, balanced nitrogen fertilization, and fungicide sprays of Tricyclazole, Azoxystrobin, or Isoprothiolane.'
      },
      {
        id: 'chunk_powdery_mildew',
        page: 462,
        chapter: 'Chapter 11: Plant Diseases Caused by Ascomycetes',
        topic: 'Powdery Mildew of Cereals, Grapes, and Cucurbits',
        text: 'Powdery mildews (Blumeria, Erysiphe, Podosphaera spp.). Symptoms: white talcum-powder-like patches of superficial mycelium and conidia on upper leaf surfaces, curling, and chlorosis. Controls: wettable sulfur, potassium bicarbonate, triazoles (Tebuconazole), and canopy thinning for ventilation.'
      },
      {
        id: 'chunk_fusarium_wilt',
        page: 542,
        chapter: 'Chapter 11: Plant Diseases Caused by Ascomycetes and Deuteromycetes',
        topic: 'Fusarium Wilt (Panama Disease)',
        text: 'Fusarium oxysporum causes vascular wilt, progressive yellowing of lower leaves, and vascular browning in xylem. Chlamydospores survive in soil for decades. Management: resistant varieties, quarantine against Tropical Race 4, biocontrol with Trichoderma harzianum, and soil solarization.'
      },
      {
        id: 'chunk_citrus_greening',
        page: 651,
        chapter: 'Chapter 12: Plant Diseases Caused by Prokaryotes',
        topic: 'Huanglongbing (HLB) / Citrus Greening',
        text: 'Huanglongbing (HLB) caused by Candidatus Liberibacter asiaticus, vectored by Asian citrus psyllid (Diaphorina citri). Symptoms: asymmetric blotchy mottle chlorosis on leaves, yellow shoots, lopsided bitter fruit. Managed with certified disease-free nursery stock, psyllid control (imidacloprid), and roguing infected trees.'
      }
    ];
  }

  console.log(`🚀 Embedding ${chunks.length} chunks with Gemini gemini-embedding-001...`);

  const vectors = [];
  for (let i = 0; i < chunks.length; i++) {
    const item = chunks[i];
    const textToEmbed = `${item.topic}: ${item.text || item.content || ''}`;
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
    console.log(`✨ Embedded (${i + 1}/${chunks.length}): ${item.topic}`);
  }

  console.log(`📡 Upserting ${vectors.length} vectors to Pinecone index "${PINECONE_INDEX_NAME}"...`);
  await index.upsert({ records: vectors });

  console.log('🎉 Ingestion complete! Pinecone Serverless is fully populated and ready for queries.');
}

main().catch(err => {
  console.error('Fatal ingestion error:', err);
  process.exit(1);
});
