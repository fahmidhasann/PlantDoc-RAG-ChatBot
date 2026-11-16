/**
 * PlantDoc — figures from a book PDF into searchable records.
 *
 *   npm run extract-figures -- <book.pdf> [--max N] [--publish] [--page-offset N]
 *
 * Writes data/processed/figures.json, which `npm run ingest` folds in beside
 * the text chunks.
 *
 * A plant pathology book carries much of its diagnostic value in its plates:
 * what a lesion margin looks like, how sporangiophores branch, the shape of a
 * leaf spot. None of that is in the running text. This pass pulls each figure
 * out, pairs it with its printed caption, and has a vision model describe what
 * is actually visible. The description is what gets embedded, so "pale ring
 * around a sunken brown centre" finds the plate even though the caption only
 * says "Figure 11-27".
 *
 * Images are NOT published by default. Reproducing a copyrighted textbook's
 * plates on a public site is a different proposition from quoting a paragraph,
 * so by default only the derived text is indexed and the figure is cited by
 * number and page. Pass --publish for a book whose licence allows it, and the
 * images are written to public/figures/ and shown in the source panel.
 *
 * Requires Poppler. --publish additionally requires `sharp`:
 *   npm install --save-dev sharp
 */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// A figure worth indexing is big enough to carry detail and not a rule,
// a logo or a decorative band.
const MIN_WIDTH = 180;
const MIN_HEIGHT = 180;
const MIN_PIXELS = 60_000;
const MAX_ASPECT = 6;

const OUT_DIR = path.resolve('data/processed');
const PUBLIC_DIR = path.resolve('public/figures');
const OUT_PATH = path.join(OUT_DIR, 'figures.json');

const CAPTION_RE = /^(fig(?:ure)?|plate|photo|image|box)\.?\s*([0-9]+(?:[.\-][0-9]+)*)\s*[.:)]?\s*(.*)$/i;

function parseArgs(argv) {
  const args = { pdf: null, max: Infinity, publish: false, pageOffset: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--max') args.max = Number(argv[++i]);
    else if (a === '--publish') args.publish = true;
    else if (a === '--page-offset') args.pageOffset = Number(argv[++i]);
    else if (!a.startsWith('--')) args.pdf = a;
  }
  return args;
}

function requireTool(name, hint) {
  try {
    execFileSync(name, ['-v'], { stdio: 'ignore' });
  } catch (err) {
    // Some tools exit non-zero on -v but still exist.
    if (err?.code === 'ENOENT') {
      console.error(`${name} is missing. ${hint}`);
      process.exit(1);
    }
  }
}

/** `pdfimages -list` rows, filtered to things that look like real figures. */
function listImages(pdf) {
  const raw = execFileSync('pdfimages', ['-list', pdf], {
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 64,
    stdio: ['ignore', 'pipe', 'ignore'],
  });

  const rows = [];
  for (const line of raw.split('\n').slice(2)) {
    const f = line.trim().split(/\s+/);
    if (f.length < 5) continue;
    const page = Number(f[0]);
    const num = Number(f[1]);
    const width = Number(f[3]);
    const height = Number(f[4]);
    if (!Number.isFinite(page) || !Number.isFinite(width) || !Number.isFinite(height)) continue;

    const aspect = Math.max(width / height, height / width);
    if (width < MIN_WIDTH || height < MIN_HEIGHT) continue;
    if (width * height < MIN_PIXELS) continue;
    if (aspect > MAX_ASPECT) continue;

    rows.push({ page, num, width, height });
  }
  return rows;
}

/** Text of one page, for finding the caption that belongs to a figure. */
function pageLines(pdf, page) {
  try {
    const raw = execFileSync('pdftotext', ['-enc', 'UTF-8', '-f', String(page), '-l', String(page), pdf, '-'], {
      encoding: 'utf8',
      maxBuffer: 1024 * 1024 * 16,
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    return raw.split('\n').map(l => l.trim()).filter(Boolean);
  } catch {
    return [];
  }
}

/**
 * The caption printed under a figure.
 *
 * Captions wrap, so once the "Figure 11-27." line is found the following lines
 * are taken until one ends a sentence or another caption starts.
 */
function findCaptions(lines) {
  const captions = [];
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(CAPTION_RE);
    if (!m) continue;
    let text = m[3] || '';
    for (let j = i + 1; j < Math.min(i + 6, lines.length); j++) {
      if (CAPTION_RE.test(lines[j])) break;
      text += ` ${lines[j]}`;
      if (/[.!?]$/.test(lines[j])) break;
    }
    captions.push({
      label: `${m[1].replace(/^\w/, c => c.toUpperCase())} ${m[2]}`,
      text: text.replace(/\s{2,}/g, ' ').trim(),
    });
  }
  return captions;
}

/** Extracts one page's images as PNGs into a temp directory. */
function extractPageImages(pdf, page, dir) {
  const prefix = path.join(dir, `p${page}`);
  try {
    execFileSync('pdfimages', ['-png', '-f', String(page), '-l', String(page), pdf, prefix], {
      stdio: ['ignore', 'ignore', 'ignore'],
    });
  } catch {
    return [];
  }
  return fs
    .readdirSync(dir)
    .filter(f => f.startsWith(`p${page}-`))
    .sort()
    .map(f => path.join(dir, f));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.pdf) {
    console.error('Usage: npm run extract-figures -- <path/to/book.pdf> [--max N] [--publish]');
    process.exit(1);
  }
  const pdf = path.resolve(args.pdf);
  if (!fs.existsSync(pdf)) {
    console.error(`PDF not found: ${pdf}`);
    process.exit(1);
  }

  requireTool('pdfimages', 'Install Poppler: brew install poppler');
  requireTool('pdftotext', 'Install Poppler: brew install poppler');

  let sharp = null;
  if (args.publish) {
    try {
      ({ default: sharp } = await import('sharp'));
    } catch {
      console.error('--publish needs sharp: npm install --save-dev sharp');
      process.exit(1);
    }
  }

  console.log(`${path.basename(pdf)}`);
  const candidates = listImages(pdf);
  console.log(`${candidates.length} images look like figures\n`);
  if (!candidates.length) {
    console.log('Nothing to do. The book may use vector line art, which is not');
    console.log('extractable this way, or the images are below the size floor.');
    return;
  }

  const byPage = new Map();
  for (const c of candidates) {
    if (!byPage.has(c.page)) byPage.set(c.page, []);
    byPage.get(c.page).push(c);
  }

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'plantdoc-figs-'));
  if (args.publish) fs.mkdirSync(PUBLIC_DIR, { recursive: true });

  const figures = [];
  const seen = new Set(); // content hashes, to drop repeated logos and rules
  let processed = 0;
  let duplicates = 0;

  for (const [page, items] of [...byPage.entries()].sort((a, b) => a[0] - b[0])) {
    if (figures.length >= args.max) break;

    const lines = pageLines(pdf, page);
    const captions = findCaptions(lines);
    const files = extractPageImages(pdf, page, tmp);

    for (let i = 0; i < items.length && figures.length < args.max; i++) {
      const file = files[i];
      if (!file || !fs.existsSync(file)) continue;

      const bytes = fs.readFileSync(file);
      const hash = createHash('sha1').update(bytes).digest('hex');
      if (seen.has(hash)) {
        duplicates++;
        fs.unlinkSync(file);
        continue;
      }
      seen.add(hash);

      // Captions and images are both in reading order on the page, so the
      // nth image usually belongs to the nth caption. Where the counts differ
      // the figure keeps the page's first caption, which is still the right
      // section even if the number is off.
      const caption = captions[i] || captions[0] || null;
      const id = `fig_p${String(page).padStart(4, '0')}_${i}`;

      const record = {
        id,
        pdfPage: page,
        page: args.pageOffset != null ? Math.max(0, page - args.pageOffset) : page,
        label: caption?.label ?? null,
        caption: caption?.text ?? '',
        width: items[i].width,
        height: items[i].height,
        // Filled in by caption-figures.mjs.
        description: '',
        image: null,
      };

      if (args.publish && sharp) {
        const outName = `${id}.webp`;
        await sharp(bytes)
          .resize({ width: 900, height: 900, fit: 'inside', withoutEnlargement: true })
          .webp({ quality: 78 })
          .toFile(path.join(PUBLIC_DIR, outName));
        record.image = `/figures/${outName}`;
      }

      // The PNG is kept for the captioning pass, which reads it from here.
      record.tempFile = file;
      figures.push(record);
      processed++;
    }

    process.stdout.write(`\r  page ${page}: ${figures.length} figures kept`);
  }
  process.stdout.write('\n');

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_PATH, JSON.stringify(figures, null, 2));

  const withCaption = figures.filter(f => f.caption).length;
  console.log(`\n${processed} figures extracted`);
  console.log(`${withCaption} paired with a printed caption`);
  if (duplicates) console.log(`${duplicates} repeated images skipped (logos, rules)`);
  console.log(args.publish ? `images written to ${PUBLIC_DIR}` : 'images not published (pass --publish for an openly licensed book)');
  console.log(`\nWrote ${OUT_PATH}`);
  console.log('Next: npm run caption-figures');
}

main().catch(err => {
  console.error('\nFigure extraction failed:', err);
  process.exit(1);
});
