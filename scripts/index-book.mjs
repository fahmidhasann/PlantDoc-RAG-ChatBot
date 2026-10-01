/**
 * One command to (re)index the whole textbook:
 *   npm run index-book -- /path/to/textbook.pdf [--page-offset N]
 * Runs `extract` and then `ingest --fresh`. If the Gemini daily quota stops ingestion,
 * finish later with `npm run ingest` (it resumes from the cache).
 */

import { spawnSync } from 'node:child_process';

const args = process.argv.slice(2);
if (!args.find(a => !a.startsWith('--'))) {
  console.error('Usage: npm run index-book -- /path/to/textbook.pdf [--page-offset N]');
  process.exit(1);
}

const run = (cmdArgs) => {
  const r = spawnSync(process.execPath, cmdArgs, { stdio: 'inherit' });
  return r.status ?? 1;
};

const extracted = run(['scripts/extract-pdf.mjs', ...args]);
if (extracted !== 0) process.exit(extracted);
process.exit(run(['--env-file-if-exists=.env.local', 'scripts/ingest.mjs', '--fresh']));
