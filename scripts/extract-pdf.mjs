/**
 * PlantDoc — textbook PDF → page-aware, section-aware text chunks
 *
 * Usage:
 *   npm run extract -- <path/to/textbook.pdf> [--page-offset N] [--first N] [--last N]
 *
 * Reads every page with Poppler's `pdftotext` (brew install poppler), strips running headers,
 * page numbers and figure panel letters, detects chapters ("chapter eleven" opening pages),
 * disease/section headings (ALL-CAPS lines) and "Selected References" lists, then writes
 * overlapping chunks to data/processed/chunks.json for `npm run ingest`.
 *
 * Every chunk gets a `kind`: body | references | glossary | index | toc | front.
 * The app searches body/glossary/front only, so reference lists and the back-of-book
 * index stay in Pinecone (the whole book is indexed) without crowding out real answers.
 *
 * --page-offset N  printed page = PDF page − N (auto-detected from printed page numbers)
 * --first/--last   only extract a PDF page range (quick trial runs)
 */

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const CHUNK_CHARS = 1500;
const OVERLAP_CHARS = 250;
const MIN_CHUNK_CHARS = 250;
const MIN_BLOCK_CHARS = 400; // a heading only starts a new block once the current one has real content
const OUT_DIR = path.resolve('data/processed');

const NUMBER_WORDS = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17,
  eighteen: 18, nineteen: 19, twenty: 20
};
const STRAND_TERMS = { SSRNA: 'ssRNA', DSRNA: 'dsRNA', SSDNA: 'ssDNA', DSDNA: 'dsDNA' };
const ACRONYMS = new Set(['RNA', 'DNA', 'PCR', 'ELISA', 'HLB', 'USA', 'II', 'III', 'IV', 'TMV', 'HR', 'SAR', 'ISR', 'PR', 'IPM', 'TYLCV', 'CMV']);
const SMALL_WORDS = new Set(['a', 'an', 'and', 'as', 'at', 'by', 'for', 'from', 'in', 'into', 'of', 'on', 'or', 'the', 'to', 'with', 'vs']);

function parseArgs(argv) {
  const args = { pdf: null, pageOffset: null, first: null, last: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--page-offset') args.pageOffset = Number(argv[++i]);
    else if (a === '--first') args.first = Number(argv[++i]);
    else if (a === '--last') args.last = Number(argv[++i]);
    else if (!a.startsWith('--')) args.pdf = a;
  }
  return args;
}

function pdfPageCount(pdf) {
  const info = execFileSync('pdfinfo', [pdf], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
  const m = info.match(/^Pages:\s+(\d+)/m);
  if (!m) throw new Error('Could not read page count from pdfinfo.');
  return Number(m[1]);
}

function extractPages(pdf, first, last) {
  const raw = execFileSync('pdftotext', ['-enc', 'UTF-8', '-f', String(first), '-l', String(last), pdf, '-'], {
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 512,
    stdio: ['ignore', 'pipe', 'ignore']
  });
  const pages = raw.split('\f');
  pages.length = last - first + 1; // drop the empty string after the final form feed
  return pages.map(p => (p || '').split('\n').map(l => l.trim()).filter(Boolean));
}

const despace = s => s.replace(/[^A-Za-z0-9]/g, '').toUpperCase();
const isNumberLine = l => /^\d{1,4}$/.test(l);
const isPanelLabel = l => l.length <= 2 && !/^\d+$/.test(l); // figure panel letters: "A", "B", "c"

/** "1 1 . P L A N T D I S E A S E S …" — letter-spaced running headers. */
function isSpacedCaps(line) {
  const tokens = line.split(' ');
  if (tokens.length < 6 || line !== line.toUpperCase()) return false;
  return tokens.filter(t => t.length <= 2).length / tokens.length > 0.6;
}

/** "8. PLANT DISEASE EPIDEMIOLOGY" — even-page running header in early chapters. */
const NUMBERED_HEADER = /^\d{1,2}\.\s+[A-Z][A-Z0-9 ,:;&'’()\-–]+$/;

function isCapsHeading(line) {
  if (line.length < 5 || line.length > 90) return false;
  if (line !== line.toUpperCase()) return false;
  const letters = (line.match(/\p{L}/gu) || []).length;
  if (letters < 4 || letters < line.replace(/\s/g, '').length * 0.7) return false;
  if (/^(FIGURE|TABLE)\b/.test(line)) return false;
  return true;
}

function titleCase(s) {
  if (s !== s.toUpperCase()) return s;
  return s
    .toLowerCase()
    .split(/(\s+|-|–|\/)/)
    .map((w, i) => {
      const bare = w.toUpperCase().replace(/[^A-Z]/g, '');
      if (STRAND_TERMS[bare]) return w.toUpperCase().replace(bare, STRAND_TERMS[bare]);
      if (ACRONYMS.has(bare)) return w.toUpperCase();
      return i > 0 && SMALL_WORDS.has(w) ? w : w.replace(/^\p{L}/u, c => c.toUpperCase());
    })
    .join('')
    .replace(/:\s*(\p{Ll})/gu, (m, c) => m.slice(0, -1) + c.toUpperCase());
}

/** Most common (PDF page − printed page) among pages that show a page number. */
function detectPageOffset(pages, firstPdfPage) {
  const counts = new Map();
  pages.forEach((lines, i) => {
    const candidates = new Set([...lines.slice(0, 8), ...lines.slice(-3)].filter(isNumberLine).map(Number));
    for (const n of candidates) {
      const offset = firstPdfPage + i - n;
      if (offset >= 0) counts.set(offset, (counts.get(offset) || 0) + 1);
    }
  });
  let best = 0;
  let bestCount = 0;
  for (const [offset, count] of counts) {
    if (count > bestCount) [best, bestCount] = [offset, count];
  }
  return bestCount >= Math.max(3, pages.length * 0.2) ? best : 0;
}

/** Remove running headers, the printed page number and figure panel letters. */
function stripPageFurniture(lines, printedPage) {
  let header = null;
  const out = [];
  // Remove one copy of the page number: at the foot of the page if present, otherwise near the top.
  // (Chapter opening outlines can list their own page number too, and that copy must stay.)
  const isPageNo = l => isNumberLine(l) && Number(l) === printedPage;
  let pageNoIdx = -1;
  for (let i = lines.length - 1; i >= Math.max(0, lines.length - 3); i--) {
    if (isPageNo(lines[i])) { pageNoIdx = i; break; }
  }
  if (pageNoIdx === -1) pageNoIdx = lines.slice(0, 6).findIndex(isPageNo);
  if (pageNoIdx === -1) pageNoIdx = lines.findIndex(isPageNo); // two-column pages can put it mid-stream
  for (const [idx, line] of lines.entries()) {
    if (idx === pageNoIdx) continue;
    if (isPanelLabel(line)) continue;
    // Letter-spaced and "8. PLANT DISEASE EPIDEMIOLOGY" lines are always running headers,
    // even when figure text pushes them below the top of the page.
    if (isSpacedCaps(line) || NUMBERED_HEADER.test(line)) {
      header = header || line;
      continue;
    }
    out.push(line);
  }
  return { lines: out, header };
}

// Reference-list lines: "Griffith, R. (1987). Red ring disease…", "193–196.", "In “Compendium of…”".
const YEAR_RE = /\((19|20)\d{2}[a-z]?\)/;
const AUTHOR_START_RE = /^[\p{Lu}][\p{L}'’-]+(?:\s(?:de|van|von|du|da|le|[\p{Lu}][\p{L}'’-]+))?,\s(?:[\p{Lu}]\.\s?-?\s?)+/u;
const isStrongRef = l => AUTHOR_START_RE.test(l) && YEAR_RE.test(l);
const isRefLike = l =>
  YEAR_RE.test(l) || AUTHOR_START_RE.test(l) || /^(pp\.|Vol\.|In [“"])/.test(l) || /\d+\s?[–-]\s?\d+\.$/.test(l) || /et al\./.test(l);

const CHAPTER_OPEN = /^chapter\s+(\d{1,2}|[a-z]+)\s*[.:—-]?\s*(.*)$/i;

function chapterNumber(token) {
  return /^\d+$/.test(token) ? Number(token) : NUMBER_WORDS[token.toLowerCase()] || null;
}

/**
 * Title lines after "chapter eleven". The opening page then lists the chapter's sections
 * (an outline with page numbers), so the title ends at "INTRODUCTION", a long outline line,
 * or a line that is followed by a page number.
 */
function chapterTitleLines(lines, startIdx, inlineTitle) {
  if (inlineTitle) return { title: titleCase(inlineTitle), consumed: 0 };
  const out = [];
  for (let i = startIdx + 1; i < Math.min(lines.length, startIdx + 6); i++) {
    const line = lines[i];
    if (!isCapsHeading(line)) break;
    if (out.length) {
      const looksLikeOutline = /^INTRODUCTION\b/.test(line) || line.includes(' – ') || line.length > 60;
      if (looksLikeOutline || isNumberLine(lines[i + 1] || '')) break;
    }
    out.push(line);
  }
  return { title: titleCase(out.join(' ')), consumed: out.length };
}

/** Split text into overlapping windows, preferring sentence boundaries. */
function windowText(text) {
  if (text.length <= CHUNK_CHARS) return [{ start: 0, text }];
  const out = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(start + CHUNK_CHARS, text.length);
    if (end < text.length) {
      const slice = text.slice(start, end);
      const lastStop = Math.max(slice.lastIndexOf('. '), slice.lastIndexOf('? '), slice.lastIndexOf('! '));
      if (lastStop > CHUNK_CHARS * 0.6) end = start + lastStop + 1;
    }
    out.push({ start, text: text.slice(start, end).trim() });
    if (end >= text.length) break;
    let next = end - OVERLAP_CHARS;
    const space = text.indexOf(' ', next);
    if (space !== -1 && space < end) next = space + 1;
    start = next;
  }
  // Fold a tiny tail into the previous window.
  if (out.length > 1 && out.at(-1).text.length < MIN_CHUNK_CHARS) {
    const tail = out.pop();
    const prev = out.at(-1);
    prev.text = text.slice(prev.start, tail.start + tail.text.length).trim();
  }
  return out;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.pdf) {
    console.error('Usage: npm run extract -- <path/to/textbook.pdf> [--page-offset N] [--first N] [--last N]');
    process.exit(1);
  }
  const pdf = path.resolve(args.pdf);
  if (!fs.existsSync(pdf)) {
    console.error(`PDF not found: ${pdf}`);
    process.exit(1);
  }
  try {
    execFileSync('pdftotext', ['-v'], { stdio: 'ignore' });
  } catch {
    console.error('pdftotext is missing. Install Poppler first: brew install poppler (macOS) or apt install poppler-utils.');
    process.exit(1);
  }

  const total = pdfPageCount(pdf);
  const first = Math.max(1, args.first || 1);
  const last = Math.min(total, args.last || total);
  console.log(`📄 ${path.basename(pdf)} — ${total} pages (extracting ${first}–${last})`);

  const pages = extractPages(pdf, first, last);
  const manualOffset = Number.isFinite(args.pageOffset);
  const pageOffset = manualOffset ? args.pageOffset : detectPageOffset(pages, first);
  console.log(`🔢 Printed page = PDF page − ${pageOffset} (${manualOffset ? 'manual' : 'auto-detected'})`);

  // ---- Walk the book, building blocks of continuous text that share chapter/kind/topic ----
  const blocks = [];
  let chapter = 'Front Matter';
  let kind = 'front';
  let topic = null;
  let seenHeadings = new Set();
  let block = null;
  let inOutline = false;
  let emptyPages = 0;
  const chaptersFound = [];

  const startBlock = () => {
    if (block && block.text.length) blocks.push(block);
    block = { chapter, kind, topic, text: '', marks: [] };
  };
  const append = (text, pdfPage, printed) => {
    if (!block) startBlock();
    const offset = block.text.length ? block.text.length + 1 : 0;
    if (!block.marks.length || block.marks.at(-1).pdfPage !== pdfPage) block.marks.push({ offset, pdfPage, page: printed });
    block.text += (block.text ? ' ' : '') + text;
  };

  // Reference lists often end without a heading (the next section may start with a Title Case
  // heading, or the list sits in one column next to body text). Lines are held back until it is
  // clear which side they belong to.
  let held = []; // { line, pdfPage, printed }
  let heldStrong = 0;
  let proseRun = 0;
  const flushHeld = () => {
    for (const h of held) append(h.line, h.pdfPage, h.printed);
    held = [];
    heldStrong = 0;
    proseRun = 0;
  };
  const classifyLine = (line, pdfPage, printed) => {
    const refLike = isRefLike(line);
    if (kind === 'body') {
      if (isStrongRef(line)) {
        held.push({ line, pdfPage, printed });
        heldStrong++;
        proseRun = 0;
        if (heldStrong >= 3) {
          const lines = held;
          held = [];
          heldStrong = 0;
          kind = 'references';
          startBlock();
          for (const h of lines) append(h.line, h.pdfPage, h.printed);
        }
        return;
      }
      if (held.length) {
        held.push({ line, pdfPage, printed });
        proseRun = refLike ? 0 : proseRun + 1;
        if (proseRun >= 3) flushHeld();
        return;
      }
      append(line, pdfPage, printed);
      return;
    }
    // kind === 'references'
    if (refLike) {
      flushHeld();
      append(line, pdfPage, printed);
      return;
    }
    held.push({ line, pdfPage, printed });
    if (held.length >= 6) {
      const lines = held;
      held = [];
      kind = 'body';
      startBlock();
      for (const h of lines) append(h.line, h.pdfPage, h.printed);
    }
  };

  pages.forEach((rawLines, i) => {
    const pdfPage = first + i;
    const printedRaw = pdfPage - pageOffset;
    const printed = printedRaw > 0 ? printedRaw : 0;
    const { lines } = stripPageFurniture(rawLines, printed);
    if (lines.join(' ').length < 40) {
      emptyPages++;
      return;
    }

    // Structural page types.
    if (/^(glossary|index|contents)$/i.test(lines[0]) || (kind === 'toc' && /^(preface|foreword|acknowledg)/i.test(lines[0]))) flushHeld();
    const top = lines.slice(0, 3).map(l => l.toLowerCase());
    if (kind === 'front' && (top.includes('contents') || top[0] === 'contents')) {
      kind = 'toc';
      startBlock();
    } else if (kind === 'toc' && /^(preface|foreword|acknowledg)/i.test(lines[0])) {
      kind = 'front';
      startBlock();
    }
    if (/^glossary$/i.test(lines[0])) {
      chapter = 'Glossary';
      kind = 'glossary';
      topic = null;
      startBlock();
    } else if (/^index$/i.test(lines[0]) && chapter !== 'Index') {
      chapter = 'Index';
      kind = 'index';
      topic = null;
      startBlock();
    }

    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];

      const open = li < 4 && kind !== 'toc' ? line.match(CHAPTER_OPEN) : null;
      const openNumber = open ? chapterNumber(open[1]) : null;
      if (openNumber) {
        flushHeld();
        const { title, consumed } = chapterTitleLines(lines, li, open[2].trim());
        li += consumed;
        chapter = `Chapter ${openNumber}${title ? `: ${title}` : ''}`;
        chaptersFound.push({ chapter, pdfPage });
        kind = 'toc'; // the chapter outline that follows the title
        inOutline = true;
        topic = 'Chapter outline';
        seenHeadings = new Set();
        startBlock();
        continue;
      }

      if (inOutline) {
        if (isCapsHeading(line) || isNumberLine(line)) {
          append(line, pdfPage, printed);
          continue;
        }
        inOutline = false;
        kind = 'body';
        topic = null;
        startBlock();
      }

      if (/^(selected |suggested )?references$/i.test(line) && (kind === 'body' || kind === 'references')) {
        flushHeld();
        kind = 'references';
        startBlock();
        continue;
      }

      if ((kind === 'body' || kind === 'references') && isCapsHeading(line)) {
        // Join headings that wrap onto the next line.
        let heading = line;
        while (li + 1 < lines.length && isCapsHeading(lines[li + 1]) && (heading + lines[li + 1]).length < 110) {
          heading += ` ${lines[++li]}`;
        }
        const key = despace(heading);
        // A caps line repeating an earlier heading near the top of a page is a running header.
        if (li < 4 && seenHeadings.has(key)) continue;
        if (key === despace(topic || '')) continue;
        seenHeadings.add(key);
        flushHeld();
        const readable = titleCase(heading);
        if (kind === 'references' || !block || block.text.length >= MIN_BLOCK_CHARS) {
          kind = 'body';
          topic = readable;
          startBlock();
        } else {
          topic = readable;
          block.topic = readable;
        }
        append(`${readable}.`, pdfPage, printed);
        continue;
      }

      if (kind === 'body' || kind === 'references') {
        classifyLine(line, pdfPage, printed);
        continue;
      }
      append(line, pdfPage, printed);
    }
  });
  flushHeld();
  if (block && block.text.length) blocks.push(block);

  // ---- Window each block into chunks ----
  const markAt = (marks, offset) => {
    let found = marks[0];
    for (const m of marks) {
      if (m.offset <= offset) found = m;
      else break;
    }
    return found;
  };

  const chunks = [];
  for (const b of blocks) {
    const text = b.text.replace(/\s{2,}/g, ' ').trim();
    if (text.length < 60) continue;
    for (const piece of windowText(text)) {
      const startMark = markAt(b.marks, piece.start);
      const endMark = markAt(b.marks, piece.start + piece.text.length - 1);
      chunks.push({
        id: '',
        page: startMark.page,
        pageEnd: endMark.page,
        pdfPage: startMark.pdfPage,
        chapter: b.chapter,
        topic: b.topic || b.chapter,
        kind: b.kind,
        text: piece.text
      });
    }
  }
  chunks.forEach((c, i) => {
    c.id = `agrios_p${String(c.pdfPage).padStart(4, '0')}_${String(i).padStart(5, '0')}`;
  });

  fs.mkdirSync(OUT_DIR, { recursive: true });
  const outPath = path.join(OUT_DIR, 'chunks.json');
  fs.writeFileSync(outPath, JSON.stringify(chunks, null, 2));

  const byKind = chunks.reduce((acc, c) => ((acc[c.kind] = (acc[c.kind] || 0) + 1), acc), {});
  const coveredPages = new Set(chunks.flatMap(c => {
    const out = [];
    for (let p = c.pdfPage; p <= c.pdfPage + Math.max(0, c.pageEnd - c.page); p++) out.push(p);
    return out;
  }));
  console.log(`📚 Chapters detected: ${chaptersFound.length}`);
  for (const c of chaptersFound) console.log(`   • PDF p.${c.pdfPage} — ${c.chapter}`);
  console.log(`🧩 ${chunks.length} chunks ${JSON.stringify(byKind)} covering ${coveredPages.size}/${last - first + 1} PDF pages`);
  if (emptyPages) console.log(`   (${emptyPages} pages had no text — part dividers, blank or image-only pages)`);
  if (emptyPages > (last - first + 1) * 0.2) {
    console.warn('⚠️  Many pages had no extractable text. If this PDF is a scan, OCR it first (e.g. ocrmypdf in.pdf out.pdf).');
  }
  console.log(`💾 Wrote ${outPath}`);
}

main();
