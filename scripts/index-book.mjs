/**
 * PlantDoc — one command to index a book end to end.
 *
 *   npm run index-book -- /path/to/book.pdf [--page-offset N]
 *
 * Runs `extract` and then `ingest --fresh`. If ingestion is interrupted,
 * finish it with `npm run ingest` — embeddings are cached, so it resumes.
 */

import { spawnSync } from 'node:child_process';

const args = process.argv.slice(2);
if (!args.some(a => !a.startsWith('--'))) {
  console.error('Usage: npm run index-book -- /path/to/book.pdf [--page-offset N]');
  process.exit(1);
}

const run = cmdArgs => spawnSync(process.execPath, cmdArgs, { stdio: 'inherit' }).status ?? 1;

const extracted = run(['scripts/extract-pdf.mjs', ...args]);
if (extracted !== 0) process.exit(extracted);

process.exit(run(['--env-file-if-exists=.env.local', 'scripts/ingest.mjs', '--fresh']));
