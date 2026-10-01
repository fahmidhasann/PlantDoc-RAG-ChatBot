/**
 * PlantDoc — embed textbook chunks and upsert them into Pinecone.
 *
 * Usage:
 *   npm run extract -- <textbook.pdf>   # writes data/processed/chunks.json
 *   npm run ingest                      # embeds + upserts every chunk
 *   npm run ingest -- --fresh           # wipe the index namespace first (removes old vectors)
 *
 * Embeddings come from Pinecone's hosted `llama-text-embed-v2` model (768 dims, multilingual),
 * which is included in the free Starter plan — no Gemini quota is used for search.
 * PINECONE_API_KEY comes from the environment (.env.local is loaded by the npm script).
 * Embeddings are cached in data/processed/embeddings.jsonl, keyed by model + chunk text,
 * so re-running resumes where it stopped and re-extracting only re-embeds changed chunks.
 */

import { Pinecone } from '@pinecone-database/pinecone';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'plantdoc-book';
const EMBED_MODEL = 'llama-text-embed-v2'; // keep in sync with src/lib/pinecone.ts
const DIMENSION = 768;
const EMBED_BATCH = 96; // model maximum per request
const UPSERT_BATCH = 100; // records per Pinecone upsert (keeps requests well under 2 MB)
const MAX_METADATA_TEXT = 3000; // chars of chunk text stored for retrieval (Pinecone allows 40 KB)

const CHUNKS_PATH = path.resolve('data/processed/chunks.json');
const CACHE_PATH = path.resolve('data/processed/embeddings.jsonl');

const args = new Set(process.argv.slice(2));
const FRESH = args.has('--fresh');

if (!PINECONE_API_KEY) {
  console.error('❌ PINECONE_API_KEY must be set (put it in .env.local).');
  process.exit(1);
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

function embeddingText(chunk) {
  const heading = [chunk.chapter, chunk.topic !== chunk.chapter ? chunk.topic : null].filter(Boolean).join(' — ');
  return `${heading}\n${chunk.text || chunk.content || ''}`;
}

function cacheKey(chunk) {
  return createHash('sha256').update(`${EMBED_MODEL}|${DIMENSION}|${embeddingText(chunk)}`).digest('hex').slice(0, 32);
}

async function embedBatch(pinecone, chunks) {
  for (let attempt = 1; attempt <= 10; attempt++) {
    try {
      const res = await pinecone.inference.embed({
        model: EMBED_MODEL,
        inputs: chunks.map(embeddingText),
        parameters: { inputType: 'passage', truncate: 'END', dimension: DIMENSION }
      });
      const vectors = res.data.map(d => d.values);
      if (vectors.length !== chunks.length || vectors.some(v => v?.length !== DIMENSION)) {
        throw new Error('Unexpected embedding response shape.');
      }
      return vectors;
    } catch (err) {
      const msg = String(err?.message || err);
      const retryable = /429|rate|quota|too many|RESOURCE_EXHAUSTED|5\d\d|timeout|ECONNRESET|fetch failed/i.test(msg);
      if (!retryable || attempt === 10) throw err;
      const wait = Math.min(60, 5 * 2 ** (attempt - 1)) * 1000;
      console.warn(`⏳ Embedding retry in ${wait / 1000}s (attempt ${attempt}/10): ${msg.slice(0, 160)}`);
      await sleep(wait);
    }
  }
}

function loadCache() {
  const cache = new Map();
  if (!fs.existsSync(CACHE_PATH)) return cache;
  for (const line of fs.readFileSync(CACHE_PATH, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try {
      const { key, values } = JSON.parse(line);
      if (key && Array.isArray(values) && values.length === DIMENSION) cache.set(key, values);
    } catch {
      // ignore a partially written last line
    }
  }
  return cache;
}

async function ensureIndex(pinecone) {
  const existing = await pinecone.listIndexes();
  const found = existing.indexes?.find(idx => idx.name === PINECONE_INDEX_NAME);
  if (found) {
    if (found.dimension !== DIMENSION) {
      throw new Error(`Index "${PINECONE_INDEX_NAME}" has dimension ${found.dimension}, expected ${DIMENSION}.`);
    }
    const model = found.tags?.embed_model;
    if (model && model !== EMBED_MODEL) {
      throw new Error(`Index "${PINECONE_INDEX_NAME}" was built with ${model}; use a new index name for ${EMBED_MODEL}.`);
    }
    if (!model && !FRESH) {
      throw new Error(`Index "${PINECONE_INDEX_NAME}" has no embed_model tag (likely built with another model). Re-run with --fresh to rebuild it.`);
    }
    if (!model) await pinecone.configureIndex({ name: PINECONE_INDEX_NAME, tags: { embed_model: EMBED_MODEL } });
    console.log(`✅ Index "${PINECONE_INDEX_NAME}" exists.`);
    return;
  }
  console.log(`📦 Creating serverless index "${PINECONE_INDEX_NAME}" (dim ${DIMENSION}, cosine, aws us-east-1 — free Starter plan region)...`);
  await pinecone.createIndex({
    name: PINECONE_INDEX_NAME,
    dimension: DIMENSION,
    metric: 'cosine',
    spec: { serverless: { cloud: 'aws', region: 'us-east-1' } },
    tags: { embed_model: EMBED_MODEL },
    waitUntilReady: true
  });
  console.log('✅ Index is ready.');
}

async function main() {
  console.log('🌿 PlantDoc ingestion starting...');

  if (!fs.existsSync(CHUNKS_PATH)) {
    console.error(`❌ ${CHUNKS_PATH} not found. Run: npm run extract -- <path/to/textbook.pdf>`);
    process.exit(1);
  }
  const chunks = JSON.parse(fs.readFileSync(CHUNKS_PATH, 'utf8'));
  const pages = new Set();
  for (const c of chunks) {
    const start = c.pdfPage ?? c.page;
    const span = Math.max(0, (c.pageEnd ?? c.page) - (c.page ?? 0));
    for (let p = start; p <= start + span; p++) pages.add(p);
  }
  console.log(`📖 ${chunks.length} chunks covering ${pages.size} PDF pages`);

  const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
  await ensureIndex(pinecone);
  const index = pinecone.index(PINECONE_INDEX_NAME);

  // 1. Embed (resumable).
  const cache = loadCache();
  const keys = new Map(chunks.map(c => [c.id, cacheKey(c)]));
  const todo = chunks.filter(c => !cache.has(keys.get(c.id)));
  console.log(`🧠 ${cache.size} embeddings cached, ${todo.length} to embed with ${EMBED_MODEL}`);
  const out = fs.createWriteStream(CACHE_PATH, { flags: 'a' });
  for (let i = 0; i < todo.length; i += EMBED_BATCH) {
    const batch = todo.slice(i, i + EMBED_BATCH);
    const vectors = await embedBatch(pinecone, batch);
    batch.forEach((c, j) => {
      cache.set(keys.get(c.id), vectors[j]);
      out.write(`${JSON.stringify({ key: keys.get(c.id), values: vectors[j] })}\n`);
    });
    console.log(`✨ Embedded ${Math.min(i + EMBED_BATCH, todo.length)}/${todo.length}`);
  }
  await new Promise(r => out.end(r));

  // 2. Upsert everything that has an embedding.
  if (FRESH) {
    console.log('🧹 --fresh: deleting all existing vectors in the default namespace...');
    await index.deleteAll().catch(err => {
      if (!/not ?found/i.test(String(err))) throw err;
    });
  }

  const ready = chunks.filter(c => cache.has(keys.get(c.id)));
  console.log(`📡 Upserting ${ready.length} vectors to "${PINECONE_INDEX_NAME}"...`);
  for (let i = 0; i < ready.length; i += UPSERT_BATCH) {
    const records = ready.slice(i, i + UPSERT_BATCH).map(c => ({
      id: c.id,
      values: cache.get(keys.get(c.id)),
      metadata: {
        page: Number(c.page || 0),
        pageEnd: Number(c.pageEnd || c.page || 0),
        pdfPage: Number(c.pdfPage || c.page || 0),
        chapter: String(c.chapter || ''),
        topic: String(c.topic || ''),
        kind: String(c.kind || 'body'),
        text: String(c.text || c.content || '').slice(0, MAX_METADATA_TEXT)
      }
    }));
    await index.upsert({ records });
    process.stdout.write(`\r   ${Math.min(i + UPSERT_BATCH, ready.length)}/${ready.length}`);
  }
  process.stdout.write('\n');

  // Serverless stats are eventually consistent; give them a moment.
  await sleep(5000);
  const stats = await index.describeIndexStats();
  console.log(`📊 Index now holds ${stats.totalRecordCount ?? stats.totalVectorCount} vectors.`);

  console.log('🎉 Ingestion complete — the full textbook is searchable.');
}

main().catch(err => {
  console.error('Fatal ingestion error:', err);
  process.exit(1);
});
