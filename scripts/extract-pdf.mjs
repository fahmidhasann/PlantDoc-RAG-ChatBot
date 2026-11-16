/**
 * PlantDoc — book PDF to page-aware, section-aware chunks.
 *
 *   npm run extract -- <book.pdf> [--page-offset N] [--first N] [--last N] [--verbose]
 *
 * Writes data/processed/chunks.json for `npm run ingest`.
 *
 * This is deliberately book-agnostic. Nothing below is tuned to a particular
 * title: running headers are found by looking for lines that repeat across
 * many pages, headings by typographic shape, and the printed-page offset by
 * reading the page numbers the book prints on itself. Give it a different
 * textbook and it should behave the same way.
 *
 * Every chunk carries:
 *   page / pageEnd  printed page numbers, which is what citations quote
 *   pdfPage         physical page, for locating the passage in the file
 *   chapter, topic  nearest chapter and section headings above it
 *   kind            body | references | glossary | index | toc | front
 *
 * `kind` matters because a bibliography page is dense with disease names and
 * will out-score real prose on any keyword search, while explaining nothing.
 * The whole book is indexed; only body, glossary and front text can ground an
 * answer.
 *
 * Requires Poppler: brew install poppler (macOS) / apt install poppler-utils.
 */

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const CHUNK_CHARS = 1400;
const OVERLAP_CHARS = 220;
const MIN_CHUNK_CHARS = 240;
const MIN_BLOCK_CHARS = 350;
const OUT_DIR = path.resolve('data/processed');

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const args = { pdf: null, pageOffset: null, first: null, last: null, verbose: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--page-offset') args.pageOffset = Number(argv[++i]);
    else if (a === '--first') args.first = Number(argv[++i]);
    else if (a === '--last') args.last = Number(argv[++i]);
    else if (a === '--verbose') args.verbose = true;
    else if (!a.startsWith('--')) args.pdf = a;
  }
  return args;
}

// ---------------------------------------------------------------------------
// Poppler
// ---------------------------------------------------------------------------

function requirePoppler() {
  try {
    execFileSync('pdftotext', ['-v'], { stdio: 'ignore' });
  } catch {
    console.error('pdftotext is missing. Install Poppler:');
    console.error('  macOS:  brew install poppler');
    console.error('  Debian: sudo apt install poppler-utils');
    process.exit(1);
  }
}

function pdfPageCount(pdf) {
  const info = execFileSync('pdfinfo', [pdf], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
  const m = info.match(/^Pages:\s+(\d+)/m);
  if (!m) throw new Error('pdfinfo did not report a page count.');
  return Number(m[1]);
}

function extractPages(pdf, first, last) {
  const raw = execFileSync('pdftotext', ['-enc', 'UTF-8', '-f', String(first), '-l', String(last), pdf, '-'], {
    encoding: 'utf8',
    maxBuffer: 1024 * 1024 * 768,
    stdio: ['ignore', 'pipe', 'ignore'],
  });
  const pages = raw.split('\f');
  pages.length = last - first + 1; // drop the trailing empty split
  return pages.map(p => (p || '').split('\n').map(l => l.trim()).filter(Boolean));
}

// ---------------------------------------------------------------------------
// Page furniture
// ---------------------------------------------------------------------------

const isNumberLine = l => /^[\divxlcIVXLC]{1,6}$/.test(l);

/** Normalised form used to spot the same header printed on many pages. */
function signature(line) {
  return line
    .replace(/\d+/g, '#')
    .replace(/[^A-Za-z#]/g, '')
    .toUpperCase()
    .slice(0, 48);
}

/**
 * Finds running heads and feet empirically.
 *
 * A book repeats its title, the chapter name or the author across most pages
 * in the top or bottom couple of lines. Rather than guessing at the format,
 * count how often each normalised line shape appears in those positions: a
 * shape on more than a few percent of pages is furniture, not prose.
 */
function findFurniture(pages) {
  const head = new Map();
  const foot = new Map();
  const bump = (map, line) => {
    const sig = signature(line);
    if (sig.length < 4) return;
    map.set(sig, (map.get(sig) || 0) + 1);
  };

  for (const lines of pages) {
    if (lines.length < 4) continue;
    for (const l of lines.slice(0, 2)) bump(head, l);
    for (const l of lines.slice(-2)) bump(foot, l);
  }

  const threshold = Math.max(4, Math.floor(pages.length * 0.04));
  const keep = map => new Set([...map.entries()].filter(([, n]) => n >= threshold).map(([s]) => s));
  return { head: keep(head), foot: keep(foot) };
}

/** Removes running heads/feet, page numbers and stray figure panel letters. */
function stripFurniture(lines, furniture) {
  let start = 0;
  let end = lines.length;

  while (start < end && start < 3) {
    const l = lines[start];
    if (isNumberLine(l) || furniture.head.has(signature(l))) start++;
    else break;
  }
  while (end > start && end > lines.length - 3) {
    const l = lines[end - 1];
    if (isNumberLine(l) || furniture.foot.has(signature(l))) end--;
    else break;
  }

  return lines.slice(start, end).filter(l => {
    if (l.length <= 2 && !/^\d+$/.test(l)) return false; // figure panel letters
    if (/^(Fig(ure)?|Table|Plate)\.?\s*\d+[.:]?$/i.test(l)) return false;
    return true;
  });
}

// ---------------------------------------------------------------------------
// Printed page numbers
// ---------------------------------------------------------------------------

/**
 * Printed page = PDF page − offset.
 *
 * Front matter means the two rarely line up, and citations must quote the
 * number a reader would see on the page. Taking the most common difference
 * across every page that prints a number is robust to the handful of pages
 * that print something else in the same position.
 */
function detectPageOffset(pages, firstPdfPage) {
  const counts = new Map();
  pages.forEach((lines, i) => {
    const candidates = new Set(
      [...lines.slice(0, 3), ...lines.slice(-3)].filter(l => /^\d{1,4}$/.test(l)).map(Number)
    );
    for (const n of candidates) {
      const offset = firstPdfPage + i - n;
      if (offset >= 0 && offset < 200) counts.set(offset, (counts.get(offset) || 0) + 1);
    }
  });

  let best = 0;
  let bestCount = 0;
  for (const [offset, count] of counts) {
    if (count > bestCount) [best, bestCount] = [offset, count];
  }
  return { offset: best, confidence: bestCount / Math.max(1, pages.length) };
}

// ---------------------------------------------------------------------------
// Headings
// ---------------------------------------------------------------------------

const NUMBER_WORDS = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17,
  eighteen: 18, nineteen: 19, twenty: 20, 'twenty-one': 21, 'twenty-two': 22, 'twenty-three': 23,
  'twenty-four': 24, 'twenty-five': 25,
};

const ROMAN = /^[IVXL]+$/;

function romanToInt(s) {
  const map = { I: 1, V: 5, X: 10, L: 50, C: 100 };
  let total = 0;
  for (let i = 0; i < s.length; i++) {
    const v = map[s[i]];
    total += v < map[s[i + 1]] ? -v : v;
  }
  return total;
}

/**
 * "Chapter 11", "CHAPTER ELEVEN", "Part IV" — returns {number, title} or null.
 *
 * Guarded hard, because this pattern is the one most likely to fire on
 * something that is not a chapter opening: a publisher's address block
 * ("Unit 405, Office Block…"), a journal citation ("Part 2, pp. 81-86"), or a
 * cross-reference mid-paragraph. A real chapter opening is a short standalone
 * line with a plausible number and, if it has a title at all, one that starts
 * with a letter.
 */
function matchChapterLine(line) {
  if (line.length > 80) return null;
  const m = line.match(/^(chapter|part|section)\s+([A-Za-z0-9-]+)\s*[.:—–]?\s*(.*)$/i);
  if (!m) return null;

  const token = m[2];
  let number = null;
  if (/^\d+$/.test(token)) number = Number(token);
  else if (ROMAN.test(token.toUpperCase())) number = romanToInt(token.toUpperCase());
  else if (NUMBER_WORDS[token.toLowerCase()] !== undefined) number = NUMBER_WORDS[token.toLowerCase()];
  if (number === null || number < 1 || number > 60) return null;

  const title = m[3].trim();
  // A trailing fragment that opens with punctuation or a lowercase word is a
  // sentence continuing, not a chapter title.
  if (title && !/^[A-Z(]/.test(title)) return null;
  if (/\b(pp?\.|ISSN|ISBN|vol\.?|no\.?)\b/i.test(line)) return null;

  return { kind: m[1].toLowerCase(), number, title };
}

/** "3.2.1 Inoculum dynamics" — decimal-numbered section headings. */
function matchNumberedHeading(line) {
  const m = line.match(/^(\d+(?:\.\d+){0,3})\.?\s+([A-Z][^.!?]{3,80})$/);
  if (!m) return null;
  return { number: m[1], title: m[2].trim() };
}

const ACRONYMS = new Set([
  'RNA', 'DNA', 'PCR', 'ELISA', 'HLB', 'USA', 'UK', 'II', 'III', 'IV', 'TMV', 'HR', 'SAR',
  'ISR', 'PR', 'IPM', 'TYLCV', 'CMV', 'PVY', 'ToMV', 'CTV', 'BLB', 'GMO', 'QTL', 'UV',
]);
const SMALL_WORDS = new Set([
  'a', 'an', 'and', 'as', 'at', 'by', 'for', 'from', 'in', 'into', 'of', 'on', 'or', 'the',
  'to', 'with', 'vs', 'via',
]);
const STRAND_TERMS = { SSRNA: 'ssRNA', DSRNA: 'dsRNA', SSDNA: 'ssDNA', DSDNA: 'dsDNA' };

/** All-caps heading, e.g. "LATE BLIGHT OF POTATO AND TOMATO". */
function isCapsHeading(line) {
  if (line.length < 5 || line.length > 90) return false;
  if (line !== line.toUpperCase()) return false;
  const letters = (line.match(/\p{L}/gu) || []).length;
  if (letters < 4) return false;
  if (letters < line.replace(/\s/g, '').length * 0.65) return false;
  if (/^(FIGURE|TABLE|PLATE|BOX)\b/.test(line)) return false;
  // Letter-spaced running heads: "L A T E  B L I G H T".
  const tokens = line.split(/\s+/);
  if (tokens.length >= 6 && tokens.filter(t => t.length <= 2).length / tokens.length > 0.6) return false;
  return true;
}

/**
 * Title Case heading, e.g. "Symptoms and Signs".
 *
 * Only accepted when it stands alone as a short line that does not end in
 * sentence punctuation, which is what separates a heading from the first line
 * of a wrapped paragraph.
 */
function isTitleCaseHeading(line, nextLine) {
  if (line.length < 6 || line.length > 70) return false;
  if (/[.!?,;:]$/.test(line)) return false;
  if (/^[a-z]/.test(line)) return false;
  const words = line.split(/\s+/);
  if (words.length < 2 || words.length > 9) return false;
  const significant = words.filter(w => !SMALL_WORDS.has(w.toLowerCase()) && w.length > 1);
  if (!significant.length) return false;
  const capitalised = significant.filter(w => /^[A-Z(]/.test(w)).length;
  if (capitalised / significant.length < 0.8) return false;
  // A heading is followed by body text, not by another fragment of a list.
  if (nextLine && /^[a-z]/.test(nextLine) && !/[.!?]$/.test(line)) return false;
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
    .replace(/:\s*(\p{Ll})/gu, (m, c) => `${m.slice(0, -1)}${c.toUpperCase()}`);
}

// ---------------------------------------------------------------------------
// Back matter
// ---------------------------------------------------------------------------

const SECTION_MARKERS = [
  { kind: 'references', re: /^(selected\s+)?(references|bibliography|further\s+reading|literature\s+cited|works\s+cited)\s*$/i },
  { kind: 'glossary', re: /^(glossary|glossary\s+of\s+terms)\s*$/i },
  { kind: 'index', re: /^(index|subject\s+index|author\s+index|general\s+index)\s*$/i },
  { kind: 'toc', re: /^(contents|table\s+of\s+contents)\s*$/i },
];

function matchSectionMarker(line) {
  for (const m of SECTION_MARKERS) if (m.re.test(line)) return m.kind;
  return null;
}

/** A line shaped like a bibliography entry: "Agrios, G. N. (2005). Plant..." */
function isReferenceLine(line) {
  if (line.length < 20) return false;
  if (/^[A-Z][a-z]+,\s+[A-Z]\.(\s*[A-Z]\.)*/.test(line)) return true;
  if (/\(\d{4}[a-z]?\)/.test(line) && /[A-Z][a-z]+,/.test(line)) return true;
  if (/\b\d{4}\b.*\b\d+\s*[-–]\s*\d+\b/.test(line) && /[A-Z][a-z]+,/.test(line)) return true;
  return false;
}

/** A line shaped like a back-of-book index entry: "Blight, late, 421, 423". */
function isIndexLine(line) {
  if (line.length > 110) return false;
  const numbers = line.match(/\d+/g) || [];
  if (numbers.length < 2) return false;
  const digits = numbers.join('').length;
  return digits / line.replace(/\s/g, '').length > 0.22;
}

// ---------------------------------------------------------------------------
// Windowing
// ---------------------------------------------------------------------------

/** Splits a block into overlapping windows, breaking on sentence boundaries. */
function windowText(text) {
  if (text.length <= CHUNK_CHARS) return [{ start: 0, text }];

  const out = [];
  let start = 0;
  while (start < text.length) {
    let end = Math.min(text.length, start + CHUNK_CHARS);
    if (end < text.length) {
      // Prefer a sentence end in the last quarter of the window.
      const tail = text.slice(start + Math.floor(CHUNK_CHARS * 0.72), end);
      const breakAt = Math.max(tail.lastIndexOf('. '), tail.lastIndexOf('? '), tail.lastIndexOf('! '));
      if (breakAt > 0) end = start + Math.floor(CHUNK_CHARS * 0.72) + breakAt + 1;
    }
    out.push({ start, text: text.slice(start, end).trim() });
    if (end >= text.length) break;

    let next = end - OVERLAP_CHARS;
    const space = text.indexOf(' ', next);
    if (space !== -1 && space < end) next = space + 1;
    start = Math.max(next, start + 1);
  }

  // Fold a stub tail back into the previous window.
  if (out.length > 1 && out.at(-1).text.length < MIN_CHUNK_CHARS) {
    const tail = out.pop();
    const prev = out.at(-1);
    prev.text = text.slice(prev.start, tail.start + tail.text.length).trim();
  }
  return out;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.pdf) {
    console.error('Usage: npm run extract -- <path/to/book.pdf> [--page-offset N] [--first N] [--last N]');
    process.exit(1);
  }

  const pdf = path.resolve(args.pdf);
  if (!fs.existsSync(pdf)) {
    console.error(`PDF not found: ${pdf}`);
    process.exit(1);
  }
  requirePoppler();

  const total = pdfPageCount(pdf);
  const first = Math.max(1, args.first || 1);
  const last = Math.min(total, args.last || total);
  console.log(`${path.basename(pdf)} — ${total} pages (reading ${first}–${last})`);

  const rawPages = extractPages(pdf, first, last);

  const furniture = findFurniture(rawPages);
  console.log(`running heads/feet detected: ${furniture.head.size} head, ${furniture.foot.size} foot`);

  const manual = Number.isFinite(args.pageOffset);
  const detected = detectPageOffset(rawPages, first);
  const pageOffset = manual ? args.pageOffset : detected.offset;
  console.log(
    `printed page = PDF page − ${pageOffset}` +
      (manual ? ' (manual)' : ` (detected on ${Math.round(detected.confidence * 100)}% of pages)`)
  );
  if (!manual && detected.confidence < 0.25) {
    console.warn('  page numbers were hard to read; pass --page-offset N if citations look wrong');
  }

  // -------------------------------------------------------------------------
  // Walk the book, accumulating blocks of continuous text
  // -------------------------------------------------------------------------

  const blocks = [];
  const chaptersFound = [];
  const seenHeadings = new Set();

  let chapter = 'Front Matter';
  let topic = null;
  let kind = 'front';
  let block = null;
  let emptyPages = 0;
  let bodyStarted = false;

  const startBlock = () => {
    if (block && block.text.length >= 60) blocks.push(block);
    block = { chapter, topic, kind, text: '', marks: [] };
  };

  const append = (text, pdfPage, printed) => {
    if (!block) startBlock();
    const offset = block.text.length ? block.text.length + 1 : 0;
    if (!block.marks.length || block.marks.at(-1).pdfPage !== pdfPage) {
      block.marks.push({ offset, pdfPage, page: printed });
    }
    block.text += (block.text ? ' ' : '') + text;
  };

  // Back-matter detection is run-based: a single reference-shaped line inside
  // prose means nothing, but several in a row means the bibliography started.
  let refRun = 0;
  let indexRun = 0;
  let proseRun = 0;

  rawPages.forEach((raw, i) => {
    const pdfPage = first + i;
    const printedRaw = pdfPage - pageOffset;
    const printed = printedRaw > 0 ? printedRaw : 0;
    const lines = stripFurniture(raw, furniture);

    if (lines.join(' ').length < 40) {
      emptyPages++;
      return;
    }

    for (let li = 0; li < lines.length; li++) {
      const line = lines[li];
      const nextLine = lines[li + 1];

      // --- explicit section markers -------------------------------------
      const marker = matchSectionMarker(line);
      if (marker) {
        kind = marker;
        topic = titleCase(line);
        refRun = indexRun = proseRun = 0;
        startBlock();
        continue;
      }

      // --- chapter openings ---------------------------------------------
      // Never inside back matter: a bibliography is full of "Part 2" and
      // "Section III" that refer to other works.
      const chapterMatch =
        kind === 'references' || kind === 'index' ? null : matchChapterLine(line);
      if (chapterMatch) {
        // The chapter title is often on the next line or two.
        let title = chapterMatch.title;
        let consumed = 0;
        while (!title && consumed < 2 && lines[li + 1 + consumed]) {
          const candidate = lines[li + 1 + consumed];
          if (isCapsHeading(candidate) || isTitleCaseHeading(candidate, lines[li + 2 + consumed])) {
            title = candidate;
            consumed++;
            break;
          }
          consumed++;
        }
        li += consumed;
        chapter = title
          ? `Chapter ${chapterMatch.number}: ${titleCase(title)}`
          : `Chapter ${chapterMatch.number}`;
        topic = null;
        kind = 'body';
        bodyStarted = true;
        refRun = indexRun = proseRun = 0;
        chaptersFound.push({ pdfPage, chapter });
        startBlock();
        continue;
      }

      // --- back matter, detected by run ---------------------------------
      if (kind === 'body' || kind === 'front') {
        if (isReferenceLine(line)) {
          refRun++;
          proseRun = 0;
          if (refRun >= 3) {
            kind = 'references';
            topic = 'References';
            startBlock();
          }
        } else if (isIndexLine(line)) {
          indexRun++;
          proseRun = 0;
          if (indexRun >= 6) {
            kind = 'index';
            topic = 'Index';
            startBlock();
          }
        } else {
          refRun = 0;
          indexRun = 0;
        }
      } else if (kind === 'references' || kind === 'index') {
        // Prose resuming means the list ended.
        const listLike = kind === 'references' ? isReferenceLine(line) : isIndexLine(line);
        if (listLike) proseRun = 0;
        else if (line.length > 60 && /[a-z]{4}/.test(line)) {
          proseRun++;
          if (proseRun >= 3) {
            kind = 'body';
            proseRun = 0;
            // The heading that labelled the list no longer describes what
            // follows, so wait for the next real heading.
            topic = null;
            startBlock();
          }
        }
      }

      // --- section headings ---------------------------------------------
      if (kind === 'body' || kind === 'front') {
        const numbered = matchNumberedHeading(line);
        const caps = isCapsHeading(line);
        const titled = !numbered && !caps && isTitleCaseHeading(line, nextLine);

        if (numbered || caps || titled) {
          const heading = numbered ? numbered.title : line;
          const key = heading.replace(/[^A-Za-z]/g, '').toUpperCase();

          // A heading repeated near the top of a page is a running head the
          // frequency pass did not catch.
          const isRepeatNearTop = li < 3 && seenHeadings.has(key);
          if (!isRepeatNearTop && key && key !== (topic || '').replace(/[^A-Za-z]/g, '').toUpperCase()) {
            seenHeadings.add(key);
            const readable = titleCase(heading);
            if (!bodyStarted && caps) {
              // Still in front matter; a caps line here is a part title.
              topic = readable;
            } else {
              if (!block || block.text.length >= MIN_BLOCK_CHARS) {
                topic = readable;
                if (kind === 'front') kind = 'body';
                startBlock();
              } else {
                topic = readable;
                block.topic = readable;
              }
            }
            append(`${readable}.`, pdfPage, printed);
            continue;
          }
        }
      }

      // Anything past the first few pages that is ordinary prose is body text,
      // even in a book with no detectable chapter structure.
      if (kind === 'front' && pdfPage - first > 12 && line.length > 70) {
        kind = 'body';
        bodyStarted = true;
        startBlock();
      }

      append(line, pdfPage, printed);
    }
  });

  if (block && block.text.length >= 60) blocks.push(block);

  // -------------------------------------------------------------------------
  // Window into chunks
  // -------------------------------------------------------------------------

  const markAt = (marks, offset) => {
    let found = marks[0];
    for (const m of marks) {
      if (m.offset <= offset) found = m;
      else break;
    }
    return found;
  };

  // Books without printed "Chapter N" openings (edited volumes, merged
  // chapter PDFs) would otherwise have every passage labelled "Front Matter",
  // which is worse than no label at all.
  const hasChapters = chaptersFound.length > 0;

  const chunks = [];
  for (const b of blocks) {
    if (!hasChapters) b.chapter = '';
    else if (b.chapter === 'Front Matter' && b.kind !== 'front') b.chapter = '';
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
        text: piece.text,
      });
    }
  }

  chunks.forEach((c, i) => {
    c.id = `p${String(c.pdfPage).padStart(4, '0')}_${String(i).padStart(5, '0')}`;
  });

  fs.mkdirSync(OUT_DIR, { recursive: true });
  const outPath = path.join(OUT_DIR, 'chunks.json');
  fs.writeFileSync(outPath, JSON.stringify(chunks, null, 2));

  // -------------------------------------------------------------------------
  // Report
  // -------------------------------------------------------------------------

  const byKind = chunks.reduce((acc, c) => ((acc[c.kind] = (acc[c.kind] || 0) + 1), acc), {});
  const covered = new Set(chunks.map(c => c.pdfPage));
  const answerable = chunks.filter(c => ['body', 'glossary', 'front'].includes(c.kind)).length;

  console.log(`\nchapters detected: ${chaptersFound.length}`);
  if (args.verbose) for (const c of chaptersFound) console.log(`   p.${c.pdfPage}  ${c.chapter}`);

  const topics = new Set(chunks.map(c => c.topic).filter(Boolean));
  console.log(`sections detected: ${topics.size}`);
  console.log(`${chunks.length} chunks  ${JSON.stringify(byKind)}`);
  console.log(`${answerable} are answerable (body/glossary/front)`);
  console.log(`covering ${covered.size}/${last - first + 1} pages`);
  if (emptyPages) console.log(`${emptyPages} pages had no extractable text`);
  if (emptyPages > (last - first + 1) * 0.2) {
    console.warn('\nMany pages had no text. If this PDF is a scan, OCR it first:');
    console.warn('  ocrmypdf input.pdf output.pdf');
  }
  if (!answerable) {
    console.error('\nNo answerable chunks were produced — check the warnings above.');
    process.exit(1);
  }
  console.log(`\nWrote ${outPath}`);
  console.log('Next: npm run ingest -- --fresh');
}

main();
