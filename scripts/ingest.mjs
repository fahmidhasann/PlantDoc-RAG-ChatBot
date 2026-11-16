/**
 * PlantDoc — build the dense and sparse indexes from extracted chunks.
 *
 *   npm run extract -- <book.pdf>   # writes data/processed/chunks.json
 *   npm run ingest                  # embeds + upserts into both indexes
 *   npm run ingest -- --fresh       # wipe the namespace first
 *   npm run ingest -- --dense-only  # skip the lexical index
 *
 * Both embedding models are hosted by Pinecone, so ingestion needs only
 * PINECONE_API_KEY — no Gemini quota is consumed and nothing has to run on
 * this machine but the PDF parsing.
 *
 * Embeddings are cached in data/processed/embeddings.jsonl, keyed by model and
 * chunk text. Re-running resumes where it stopped, and re-extracting only
 * re-embeds the chunks whose text actually changed.
 */

import { Pinecone } from '@pinecone-database/pinecone';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

// Keep in sync with src/lib/config.ts.
const DENSE_INDEX = process.env.PINECONE_DENSE_INDEX || 'plantdoc-dense';
const SPARSE_INDEX = process.env.PINECONE_SPARSE_INDEX || 'plantdoc-sparse';
const NAMESPACE = process.env.PINECONE_NAMESPACE || 'default';
const DENSE_MODEL = 'llama-text-embed-v2';
const DENSE_DIMENSION = 1024;
const SPARSE_MODEL = 'pinecone-sparse-english-v0';
const CLOUD = 'aws';
const REGION = 'us-east-1';

const EMBED_BATCH = 96; // hosted-model maximum per request
const UPSERT_BATCH = 100; // keeps each request well under 2 MB
const MAX_METADATA_TEXT = 3500; // Pinecone allows 40 KB; chunks are ~1.5 KB

const CHUNKS_PATH = path.resolve('data/processed/chunks.json');
const FIGURES_PATH = path.resolve('data/processed/figures.json');
const CACHE_PATH = path.resolve('data/processed/embeddings.jsonl');

const args = new Set(process.argv.slice(2));
const FRESH = args.has('--fresh');
const DENSE_ONLY = args.has('--dense-only');

const API_KEY = process.env.PINECONE_API_KEY;
if (!API_KEY) {
  console.error('PINECONE_API_KEY must be set (put it in .env.local).');
  process.exit(1);
}

const sleep = ms => new Promise(r => setTimeout(r, ms));
const fmt = n => n.toLocaleString('en-US');

/**
 * What actually gets embedded. Prefixing each chunk with its chapter and
 * section heading is cheap contextual retrieval: a passage reading "Symptoms
 * appear first on lower leaves" is nearly meaningless as a standalone vector,
 * but "Chapter 11 — Late Blight of Potato: Symptoms appear first on lower
 * leaves" lands in the right neighbourhood.
 */
function embeddingText(chunk) {
  const heading = [chunk.chapter, chunk.topic !== chunk.chapter ? chunk.topic : null]
    .filter(Boolean)
    .join(' — ');
  return `${heading}\n${chunk.text || chunk.content || ''}`;
}

function cacheKey(chunk, model, dimension) {
  return createHash('sha256')
    .update(`${model}|${dimension}|${embeddingText(chunk)}`)
    .digest('hex')
    .slice(0, 32);
}

function loadCache() {
  const cache = new Map();
  if (!fs.existsSync(CACHE_PATH)) return cache;
  for (const line of fs.readFileSync(CACHE_PATH, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.key) cache.set(entry.key, entry);
    } catch {
      // A partially written final line from an interrupted run; ignore it.
    }
  }
  return cache;
}

/** Hosted inference is rate limited per minute; back off rather than fail. */
async function withRetry(label, fn) {
  for (let attempt = 1; attempt <= 8; attempt++) {
    try {
      return await fn();
    } catch (err) {
      const msg = String(err?.message || err);
      const retryable =
        /429|rate|quota|too many|RESOURCE_EXHAUSTED|5\d\d|timeout|ECONNRESET|fetch failed/i.test(msg);
      if (!retryable || attempt === 8) throw err;
      const wait = Math.min(60, 5 * 2 ** (attempt - 1));
      console.warn(`   ${label} retry in ${wait}s (${attempt}/8): ${msg.slice(0, 140)}`);
      await sleep(wait * 1000);
    }
  }
}

async function embedDenseBatch(pc, chunks) {
  const res = await withRetry('dense embed', () =>
    pc.inference.embed({
      model: DENSE_MODEL,
      inputs: chunks.map(embeddingText),
      parameters: { inputType: 'passage', truncate: 'END', dimension: DENSE_DIMENSION },
    })
  );
  const vectors = res.data.map(d => d.values);
  if (vectors.length !== chunks.length || vectors.some(v => v?.length !== DENSE_DIMENSION)) {
    throw new Error('Dense embedding response had an unexpected shape.');
  }
  return vectors;
}

async function embedSparseBatch(pc, chunks) {
  const res = await withRetry('sparse embed', () =>
    pc.inference.embed({
      model: SPARSE_MODEL,
      inputs: chunks.map(embeddingText),
      // 2048 is the model's ceiling; chunks are far shorter, but a long one
      // should be covered whole rather than silently clipped at the 512
      // default. The SDK types parameters as strings; the API wants a number.
      parameters: {
        inputType: 'passage',
        truncate: 'END',
        max_tokens_per_sequence: 2048,
      },
    })
  );
  const rows = res.data.map(d => ({ indices: d.sparseIndices, values: d.sparseValues }));
  if (rows.length !== chunks.length || rows.some(r => !r.indices?.length)) {
    throw new Error('Sparse embedding response had an unexpected shape.');
  }
  return rows;
}

async function ensureIndex(pc, { name, vectorType, dimension, metric, model }) {
  const existing = await pc.listIndexes();
  const found = existing.indexes?.find(i => i.name === name);

  if (found) {
    if (dimension && found.dimension && found.dimension !== dimension) {
      throw new Error(
        `Index "${name}" has dimension ${found.dimension}, expected ${dimension}. ` +
          `Delete it or set a different index name.`
      );
    }
    const tagged = found.tags?.embed_model;
    if (tagged && tagged !== model) {
      throw new Error(
        `Index "${name}" was built with "${tagged}" but this run uses "${model}". ` +
          `Use a different index name, or delete the index and re-ingest.`
      );
    }
    if (!tagged) {
      await pc.configureIndex({ name, tags: { embed_model: model } }).catch(() => {});
    }
    console.log(`   index "${name}" ready (${vectorType})`);
    return;
  }

  console.log(`   creating ${vectorType} index "${name}" in ${CLOUD}/${REGION}...`);
  await pc.createIndex({
    name,
    vectorType,
    ...(dimension ? { dimension } : {}),
    ...(metric ? { metric } : {}),
    spec: { serverless: { cloud: CLOUD, region: REGION } },
    tags: { embed_model: model },
    waitUntilReady: true,
  });
  console.log(`   index "${name}" created`);
}

async function upsertAll(index, records, label) {
  for (let i = 0; i < records.length; i += UPSERT_BATCH) {
    const batch = records.slice(i, i + UPSERT_BATCH);
    await withRetry(`${label} upsert`, () => index.upsert({ records: batch }));
    process.stdout.write(`\r   ${label}: ${fmt(Math.min(i + UPSERT_BATCH, records.length))}/${fmt(records.length)}`);
  }
  process.stdout.write('\n');
}

async function main() {
  console.log('PlantDoc ingestion\n');

  if (!fs.existsSync(CHUNKS_PATH)) {
    console.error(`${CHUNKS_PATH} not found.`);
    console.error('Run: npm run extract -- <path/to/book.pdf>');
    process.exit(1);
  }

  const chunks = JSON.parse(fs.readFileSync(CHUNKS_PATH, 'utf8'));

  // Figures join the corpus as ordinary chunks whose text is the printed
  // caption plus the vision model's description of what is actually visible.
  // That is what makes a plate findable by the symptom someone can see but
  // cannot name.
  if (fs.existsSync(FIGURES_PATH)) {
    const figures = JSON.parse(fs.readFileSync(FIGURES_PATH, 'utf8'));
    let added = 0;
    for (const f of figures) {
      if (!f.description) continue; // not yet described, or nothing to index
      const heading = f.label || 'Figure';
      chunks.push({
        id: f.id,
        page: Number(f.page || f.pdfPage || 0),
        pageEnd: Number(f.page || f.pdfPage || 0),
        pdfPage: Number(f.pdfPage || 0),
        chapter: f.chapter || '',
        topic: f.caption ? `${heading}: ${f.caption.slice(0, 80)}` : heading,
        kind: 'figure',
        text: [f.label, f.caption, f.description].filter(Boolean).join('. '),
        image: f.image || '',
      });
      added++;
    }
    if (added) console.log(`${fmt(added)} described figures folded in`);
  }

  const byKind = chunks.reduce((acc, c) => ((acc[c.kind] = (acc[c.kind] || 0) + 1), acc), {});
  console.log(`${fmt(chunks.length)} chunks  ${JSON.stringify(byKind)}`);
  console.log(`namespace "${NAMESPACE}"\n`);

  const pc = new Pinecone({ apiKey: API_KEY });

  console.log('1. Indexes');
  await ensureIndex(pc, {
    name: DENSE_INDEX,
    vectorType: 'dense',
    dimension: DENSE_DIMENSION,
    metric: 'cosine',
    model: DENSE_MODEL,
  });
  if (!DENSE_ONLY) {
    await ensureIndex(pc, { name: SPARSE_INDEX, vectorType: 'sparse', model: SPARSE_MODEL });
  }

  // -------------------------------------------------------------------------
  // 2. Embed (resumable)
  // -------------------------------------------------------------------------
  console.log('\n2. Embeddings');
  const cache = loadCache();
  const denseKeys = new Map(chunks.map(c => [c.id, cacheKey(c, DENSE_MODEL, DENSE_DIMENSION)]));
  const sparseKeys = new Map(chunks.map(c => [c.id, cacheKey(c, SPARSE_MODEL, 0)]));

  const out = fs.createWriteStream(CACHE_PATH, { flags: 'a' });
  const record = entry => {
    cache.set(entry.key, entry);
    out.write(`${JSON.stringify(entry)}\n`);
  };

  const denseTodo = chunks.filter(c => !cache.has(denseKeys.get(c.id)));
  console.log(`   dense:  ${fmt(chunks.length - denseTodo.length)} cached, ${fmt(denseTodo.length)} to embed`);
  for (let i = 0; i < denseTodo.length; i += EMBED_BATCH) {
    const batch = denseTodo.slice(i, i + EMBED_BATCH);
    const vectors = await embedDenseBatch(pc, batch);
    batch.forEach((c, j) => record({ key: denseKeys.get(c.id), values: vectors[j] }));
    process.stdout.write(`\r   dense:  ${fmt(Math.min(i + EMBED_BATCH, denseTodo.length))}/${fmt(denseTodo.length)}`);
  }
  if (denseTodo.length) process.stdout.write('\n');

  if (!DENSE_ONLY) {
    const sparseTodo = chunks.filter(c => !cache.has(sparseKeys.get(c.id)));
    console.log(`   sparse: ${fmt(chunks.length - sparseTodo.length)} cached, ${fmt(sparseTodo.length)} to embed`);
    for (let i = 0; i < sparseTodo.length; i += EMBED_BATCH) {
      const batch = sparseTodo.slice(i, i + EMBED_BATCH);
      const rows = await embedSparseBatch(pc, batch);
      batch.forEach((c, j) =>
        record({ key: sparseKeys.get(c.id), indices: rows[j].indices, values: rows[j].values })
      );
      process.stdout.write(`\r   sparse: ${fmt(Math.min(i + EMBED_BATCH, sparseTodo.length))}/${fmt(sparseTodo.length)}`);
    }
    if (sparseTodo.length) process.stdout.write('\n');
  }

  await new Promise(r => out.end(r));

  // -------------------------------------------------------------------------
  // 3. Upsert
  // -------------------------------------------------------------------------
  console.log('\n3. Upsert');

  const metadataFor = c => ({
    page: Number(c.page || 0),
    pageEnd: Number(c.pageEnd || c.page || 0),
    pdfPage: Number(c.pdfPage || c.page || 0),
    chapter: String(c.chapter || ''),
    topic: String(c.topic || ''),
    kind: String(c.kind || 'body'),
    text: String(c.text || c.content || '').slice(0, MAX_METADATA_TEXT),
    // Only set when the book's licence allowed the plate to be published.
    ...(c.image ? { image: String(c.image) } : {}),
  });

  const denseIdx = pc.index(DENSE_INDEX).namespace(NAMESPACE);
  const sparseIdx = pc.index(SPARSE_INDEX).namespace(NAMESPACE);

  if (FRESH) {
    console.log(`   --fresh: clearing namespace "${NAMESPACE}"`);
    const ignoreMissing = err => {
      if (!/not ?found|404/i.test(String(err))) throw err;
    };
    await denseIdx.deleteAll().catch(ignoreMissing);
    if (!DENSE_ONLY) await sparseIdx.deleteAll().catch(ignoreMissing);
  }

  const denseRecords = chunks
    .filter(c => cache.has(denseKeys.get(c.id)))
    .map(c => ({ id: c.id, values: cache.get(denseKeys.get(c.id)).values, metadata: metadataFor(c) }));
  await upsertAll(denseIdx, denseRecords, 'dense ');

  if (!DENSE_ONLY) {
    const sparseRecords = chunks
      .filter(c => cache.has(sparseKeys.get(c.id)))
      .map(c => {
        const entry = cache.get(sparseKeys.get(c.id));
        return {
          id: c.id,
          sparseValues: { indices: entry.indices, values: entry.values },
          metadata: metadataFor(c),
        };
      });
    await upsertAll(sparseIdx, sparseRecords, 'sparse');
  }

  // Serverless stats are eventually consistent.
  console.log('\n4. Verifying');
  await sleep(6000);
  const denseStats = await denseIdx.describeIndexStats();
  console.log(`   ${DENSE_INDEX}: ${fmt(denseStats.totalRecordCount ?? 0)} vectors`);
  if (!DENSE_ONLY) {
    const sparseStats = await sparseIdx.describeIndexStats();
    console.log(`   ${SPARSE_INDEX}: ${fmt(sparseStats.totalRecordCount ?? 0)} vectors`);
  }

  console.log('\nDone. Run `npm run doctor` to sanity-check retrieval.');
}

main().catch(err => {
  console.error('\nIngestion failed:', err);
  process.exit(1);
});
