/**
 * PlantDoc — check that the indexes exist and that retrieval actually works.
 *
 *   npm run doctor
 *   npm run doctor -- "how do I manage late blight in tomato?"
 *
 * Runs the same dense + sparse + fusion + rerank path the app uses and prints
 * what came back, so a retrieval problem is visible without opening the UI.
 */

import { Pinecone } from '@pinecone-database/pinecone';

const DENSE_INDEX = process.env.PINECONE_DENSE_INDEX || 'plantdoc-dense';
const SPARSE_INDEX = process.env.PINECONE_SPARSE_INDEX || 'plantdoc-sparse';
const NAMESPACE = process.env.PINECONE_NAMESPACE || 'default';
const DENSE_MODEL = 'llama-text-embed-v2';
const DENSE_DIMENSION = 1024;
const SPARSE_MODEL = 'pinecone-sparse-english-v0';
const RERANK_MODEL = 'bge-reranker-v2-m3';
const ANSWERABLE_KINDS = ['body', 'glossary', 'front', 'figure'];
const RRF_K = 60;

const QUERY =
  process.argv.slice(2).find(a => !a.startsWith('--')) ||
  'How do I identify and manage late blight in tomato?';

const API_KEY = process.env.PINECONE_API_KEY;
if (!API_KEY) {
  console.error('PINECONE_API_KEY must be set (put it in .env.local).');
  process.exit(1);
}

const pc = new Pinecone({ apiKey: API_KEY });
const ok = s => `  OK   ${s}`;
const bad = s => `  FAIL ${s}`;
const warn = s => `  WARN ${s}`;

let failures = 0;

async function checkIndex(name, expectDimension) {
  const list = await pc.listIndexes();
  const meta = list.indexes?.find(i => i.name === name);
  if (!meta) {
    console.log(bad(`index "${name}" does not exist`));
    console.log(`       existing: ${list.indexes?.map(i => i.name).join(', ') || 'none'}`);
    failures++;
    return null;
  }

  const bits = [`type=${meta.vectorType ?? 'dense'}`];
  if (meta.dimension) bits.push(`dim=${meta.dimension}`);
  if (meta.metric) bits.push(`metric=${meta.metric}`);
  bits.push(`model=${meta.tags?.embed_model ?? 'untagged'}`);
  console.log(ok(`index "${name}"  ${bits.join('  ')}`));

  if (expectDimension && meta.dimension && meta.dimension !== expectDimension) {
    console.log(bad(`  expected dimension ${expectDimension}, found ${meta.dimension}`));
    failures++;
  }

  const stats = await pc.index(name).describeIndexStats();
  const total = stats.totalRecordCount ?? 0;
  const inNamespace = stats.namespaces?.[NAMESPACE]?.recordCount ?? 0;
  console.log(`       ${total.toLocaleString()} vectors total, ${inNamespace.toLocaleString()} in namespace "${NAMESPACE}"`);
  if (!inNamespace) {
    console.log(bad(`  namespace "${NAMESPACE}" is empty — run: npm run ingest`));
    failures++;
  }
  return meta;
}

function fuse(lists, k = RRF_K) {
  const byId = new Map();
  for (const list of lists) {
    list.forEach((m, rank) => {
      const prev = byId.get(m.id);
      const add = 1 / (k + rank + 1);
      if (prev) {
        prev.rrf += add;
        prev.from.add(m.from);
      } else {
        byId.set(m.id, { ...m, rrf: add, from: new Set([m.from]) });
      }
    });
  }
  return [...byId.values()].sort((a, b) => b.rrf - a.rrf);
}

async function main() {
  console.log(`\nPlantDoc doctor\nquery: "${QUERY}"\n`);

  console.log('1. Indexes');
  const dense = await checkIndex(DENSE_INDEX, DENSE_DIMENSION);
  const sparse = await checkIndex(SPARSE_INDEX, null);

  console.log('\n2. Query embedding');
  let denseVector = null;
  let sparseVector = null;

  if (dense) {
    const t = Date.now();
    const res = await pc.inference.embed({
      model: DENSE_MODEL,
      inputs: [QUERY],
      parameters: { inputType: 'query', truncate: 'END', dimension: DENSE_DIMENSION },
    });
    denseVector = res.data?.[0]?.values ?? null;
    if (denseVector?.length === DENSE_DIMENSION) {
      console.log(ok(`dense  ${denseVector.length} dims in ${Date.now() - t}ms`));
    } else {
      console.log(bad('dense embedding came back empty'));
      failures++;
    }
  }

  if (sparse) {
    const t = Date.now();
    const res = await pc.inference.embed({
      model: SPARSE_MODEL,
      inputs: [QUERY],
      parameters: { inputType: 'query', truncate: 'END' },
    });
    const row = res.data?.[0];
    if (row?.sparseIndices?.length) {
      sparseVector = { indices: row.sparseIndices, values: row.sparseValues };
      console.log(ok(`sparse ${row.sparseIndices.length} terms in ${Date.now() - t}ms`));
    } else {
      console.log(bad('sparse embedding came back empty'));
      failures++;
    }
  }

  console.log('\n3. Search');
  const filter = { kind: { $in: ANSWERABLE_KINDS } };
  const lists = [];

  if (denseVector) {
    const t = Date.now();
    const res = await pc.index(DENSE_INDEX).namespace(NAMESPACE).query({
      vector: denseVector,
      topK: 15,
      includeMetadata: true,
      filter,
    });
    const matches = (res.matches || []).map(m => ({ id: m.id, score: m.score, md: m.metadata, from: 'semantic' }));
    console.log(ok(`semantic: ${matches.length} matches in ${Date.now() - t}ms`));
    if (matches.length) lists.push(matches);
    else failures++;
  }

  if (sparseVector) {
    const t = Date.now();
    // A sparse index is queried with `sparseVector` alone — passing an empty
    // `vector` alongside it is rejected by the client validator.
    const res = await pc.index(SPARSE_INDEX).namespace(NAMESPACE).query({
      sparseVector,
      topK: 15,
      includeMetadata: true,
      filter,
    });
    const matches = (res.matches || []).map(m => ({ id: m.id, score: m.score, md: m.metadata, from: 'lexical' }));
    console.log(ok(`lexical:  ${matches.length} matches in ${Date.now() - t}ms`));
    if (matches.length) lists.push(matches);
    else failures++;
  }

  if (!lists.length) {
    console.log(bad('nothing retrieved — cannot continue'));
    process.exit(1);
  }

  const fused = fuse(lists).slice(0, 12);
  console.log(`\n4. Fusion\n       ${fused.length} unique candidates after RRF`);
  const overlap = fused.filter(f => f.from.size > 1).length;
  console.log(`       ${overlap} found by both retrievers`);

  console.log('\n5. Rerank');
  let ranked = fused.slice(0, 6).map(f => ({ ...f, rerank: null }));
  try {
    const t = Date.now();
    const res = await pc.inference.rerank({
      model: RERANK_MODEL,
      query: QUERY,
      documents: fused.map(f => ({ id: f.id, text: `${f.md.topic}. ${f.md.text}`.slice(0, 2200) })),
      rankFields: ['text'],
      topN: 6,
      returnDocuments: false,
      parameters: { truncate: 'END' },
    });
    ranked = res.data.map(r => ({ ...fused[r.index], rerank: r.score }));
    console.log(ok(`${RERANK_MODEL} ranked ${fused.length} candidates in ${Date.now() - t}ms`));
    console.log(`       rerank units used: ${res.usage?.rerankUnits ?? 'n/a'}`);
  } catch (err) {
    console.log(warn(`rerank unavailable (${String(err?.message || err).slice(0, 100)})`));
    console.log('       the app falls back to fused order, so this is not fatal');
  }

  console.log('\n6. Top passages\n');
  for (const r of ranked) {
    const score = r.rerank !== null && r.rerank !== undefined ? r.rerank.toFixed(3) : `rrf ${r.rrf.toFixed(4)}`;
    const via = [...r.from].join('+');
    console.log(`  ${score}  p.${r.md.page}  [${via}]  ${String(r.md.topic).slice(0, 60)}`);
    console.log(`           ${String(r.md.text).replace(/\s+/g, ' ').slice(0, 150)}...\n`);
  }

  console.log(failures ? `\n${failures} check(s) failed.\n` : '\nAll checks passed.\n');
  process.exit(failures ? 1 : 0);
}

main().catch(err => {
  console.error('\nDoctor failed:', err);
  process.exit(1);
});
