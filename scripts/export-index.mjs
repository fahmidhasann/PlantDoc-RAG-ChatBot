/**
 * PlantDoc — rebuild data/processed/chunks.json from an index that already
 * holds the book.
 *
 *   npm run export-index                      # from the default dense index
 *   npm run export-index -- --index plantdoc-book --namespace ''
 *
 * The chunk text lives in each vector's metadata, so a corpus that was ingested
 * on another machine — or from a PDF that is no longer to hand — can be
 * recovered and re-ingested into the current schema without the source file.
 *
 * Vectors are not exported: the embedding model or dimension may differ, so the
 * text is re-embedded by `npm run ingest`.
 */

import fs from 'node:fs';
import path from 'node:path';
import { Pinecone } from '@pinecone-database/pinecone';

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i !== -1 && argv[i + 1] !== undefined ? argv[i + 1] : fallback;
};

const INDEX = flag('index', process.env.PINECONE_DENSE_INDEX || 'plantdoc-dense');
const NAMESPACE = flag('namespace', process.env.PINECONE_NAMESPACE ?? '');
const OUT_PATH = path.resolve(flag('out', 'data/processed/chunks.json'));
const FETCH_BATCH = 100;

const API_KEY = process.env.PINECONE_API_KEY;
if (!API_KEY) {
  console.error('PINECONE_API_KEY must be set (put it in .env.local).');
  process.exit(1);
}

const fmt = n => n.toLocaleString('en-US');

async function main() {
  console.log(`Exporting "${INDEX}" namespace "${NAMESPACE || '(default)'}"\n`);

  const pc = new Pinecone({ apiKey: API_KEY });
  const index = NAMESPACE ? pc.index(INDEX).namespace(NAMESPACE) : pc.index(INDEX);

  const stats = await index.describeIndexStats();
  const expected = stats.totalRecordCount ?? 0;
  console.log(`index reports ${fmt(expected)} vectors`);
  if (!expected) {
    console.error('Nothing to export.');
    process.exit(1);
  }

  // 1. Page through every id.
  const ids = [];
  let paginationToken;
  do {
    const page = await index.listPaginated({ limit: 100, paginationToken });
    for (const v of page.vectors ?? []) if (v.id) ids.push(v.id);
    paginationToken = page.pagination?.next;
    process.stdout.write(`\r  listing: ${fmt(ids.length)} ids`);
  } while (paginationToken);
  process.stdout.write('\n');

  if (!ids.length) {
    console.error('\nThe index returned no ids. `listPaginated` needs a serverless index.');
    process.exit(1);
  }

  // 2. Fetch metadata in batches.
  const chunks = [];
  let missingText = 0;
  for (let i = 0; i < ids.length; i += FETCH_BATCH) {
    const batch = ids.slice(i, i + FETCH_BATCH);
    const res = await index.fetch({ ids: batch });
    for (const [id, record] of Object.entries(res.records ?? {})) {
      const md = record.metadata ?? {};
      const text = String(md.text ?? md.content ?? '');
      if (!text) {
        missingText++;
        continue;
      }
      chunks.push({
        id,
        page: Number(md.page) || 0,
        pageEnd: Number(md.pageEnd) || Number(md.page) || 0,
        pdfPage: Number(md.pdfPage) || Number(md.page) || 0,
        chapter: String(md.chapter ?? ''),
        topic: String(md.topic ?? md.chapter ?? ''),
        kind: String(md.kind ?? 'body'),
        text,
      });
    }
    process.stdout.write(`\r  fetching: ${fmt(Math.min(i + FETCH_BATCH, ids.length))}/${fmt(ids.length)}`);
  }
  process.stdout.write('\n');

  // Stable order makes diffs between exports readable.
  chunks.sort((a, b) => a.pdfPage - b.pdfPage || a.id.localeCompare(b.id));

  fs.mkdirSync(path.dirname(OUT_PATH), { recursive: true });
  fs.writeFileSync(OUT_PATH, JSON.stringify(chunks, null, 2));

  const byKind = chunks.reduce((acc, c) => ((acc[c.kind] = (acc[c.kind] || 0) + 1), acc), {});
  const pages = new Set(chunks.map(c => c.page).filter(Boolean));
  console.log(`\n${fmt(chunks.length)} chunks  ${JSON.stringify(byKind)}`);
  console.log(`covering ${fmt(pages.size)} printed pages`);
  if (missingText) console.log(`${fmt(missingText)} vectors had no text in metadata and were skipped`);
  console.log(`\nWrote ${OUT_PATH}`);
  console.log('Next: npm run ingest');
}

main().catch(err => {
  console.error('\nExport failed:', err);
  process.exit(1);
});
