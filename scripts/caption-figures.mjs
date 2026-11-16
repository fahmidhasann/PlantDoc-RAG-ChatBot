/**
 * PlantDoc — describe each extracted figure with a vision model.
 *
 *   npm run caption-figures                 # describe everything not yet done
 *   npm run caption-figures -- --limit 50   # stop after 50, to stay in quota
 *
 * Reads data/processed/figures.json (from `npm run extract-figures`) and fills
 * in each figure's `description`, then `npm run ingest` embeds those alongside
 * the text chunks.
 *
 * Why a description rather than the image itself: no hosted embedding model
 * covers images and text in one shared space on the free tiers this project
 * targets, so the standard approach is to turn the picture into words and
 * embed those. It also has a real advantage — a reader searching for "pale
 * halo around a sunken lesion" is describing the symptom in their own words,
 * which is much closer to a written description than to raw pixels.
 *
 * The prompt asks for what is visible, not for a diagnosis. A figure that is
 * labelled in the book as one disease should not be re-diagnosed by a model
 * that cannot see the caption; the book's own text does the naming.
 *
 * Progress is written back after every batch, so an interrupted run resumes.
 */

import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';

const FIGURES_PATH = path.resolve('data/processed/figures.json');
const MODELS = ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite'];
const TIMEOUT_MS = 20_000;
const PAUSE_MS = 1200; // stay inside the free tier's per-minute allowance

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i !== -1 && args[i + 1] !== undefined ? Number(args[i + 1]) : fallback;
};
const LIMIT = flag('limit', Infinity);

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error('GEMINI_API_KEY must be set (put it in .env.local).');
  process.exit(1);
}
if (!fs.existsSync(FIGURES_PATH)) {
  console.error(`${FIGURES_PATH} not found. Run: npm run extract-figures -- <book.pdf>`);
  process.exit(1);
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

const PROMPT = `You are indexing the figures of a plant pathology textbook so they can be found by search.

Describe what is VISIBLE in this image in 2-4 sentences. Be concrete and specific:
- the plant part shown (leaf, stem, fruit, tuber, root, whole plant, culture plate, microscope field)
- lesion or structure appearance: shape, margin, colour, zonation, any halo, concentric rings, sunken or raised, wet or dry
- distribution across the tissue, and the stage it appears to be at
- for micrographs or diagrams: the structures drawn or magnified, and anything labelled
- for charts, tables or equipment: what is being shown or measured

Rules:
- Describe only what you can see. Do not diagnose a disease or name a pathogen unless the name is printed in the image itself.
- Use the terms a plant pathologist would search with (chlorosis, necrosis, water-soaked, sporulation, pycnidia, concentric zonation, and so on) where they genuinely fit.
- No preamble, no "this image shows". Begin with the subject.
- If the image is a logo, a decorative band or otherwise carries no subject matter, reply with exactly: SKIP`;

const ai = new GoogleGenAI({ apiKey: API_KEY });

/** Text around the figure, so the model knows the topic it sits in. */
function pageContext(figure) {
  const bits = [];
  if (figure.label) bits.push(figure.label);
  if (figure.caption) bits.push(figure.caption);
  return bits.join(' — ').slice(0, 400);
}

async function describe(figure) {
  const file = figure.tempFile;
  if (!file || !fs.existsSync(file)) return { status: 'missing' };

  const data = fs.readFileSync(file).toString('base64');
  const context = pageContext(figure);
  const parts = [
    { inlineData: { mimeType: 'image/png', data } },
    { text: context ? `${PROMPT}\n\nThe book prints this alongside it: "${context}"` : PROMPT },
  ];

  for (const model of MODELS) {
    try {
      const res = await ai.models.generateContent({
        model,
        contents: [{ role: 'user', parts }],
        config: {
          temperature: 0,
          maxOutputTokens: 512,
          thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL, includeThoughts: false },
          abortSignal: AbortSignal.timeout(TIMEOUT_MS),
        },
      });
      const text = res.text?.trim();
      if (!text) continue;
      if (/^SKIP$/i.test(text)) return { status: 'skip' };
      return { status: 'ok', text: text.replace(/\s{2,}/g, ' ') };
    } catch (err) {
      const msg = String(err?.message || err);
      if (/429|RESOURCE_EXHAUSTED|quota/i.test(msg)) return { status: 'quota', error: msg };
      console.warn(`\n  ${model} failed: ${msg.slice(0, 120)}`);
    }
  }
  return { status: 'failed' };
}

async function main() {
  const figures = JSON.parse(fs.readFileSync(FIGURES_PATH, 'utf8'));
  const todo = figures.filter(f => !f.description && f.description !== null);
  console.log(`${figures.length} figures, ${todo.length} still to describe\n`);
  if (!todo.length) {
    console.log('Nothing to do.');
    return;
  }

  let done = 0;
  let skipped = 0;
  let failed = 0;
  const save = () => fs.writeFileSync(FIGURES_PATH, JSON.stringify(figures, null, 2));

  for (const figure of todo.slice(0, LIMIT)) {
    const result = await describe(figure);

    if (result.status === 'ok') {
      figure.description = result.text;
      done++;
    } else if (result.status === 'skip') {
      // null means "looked at it, there is nothing to index" — so a re-run
      // does not pay to look again.
      figure.description = null;
      skipped++;
    } else if (result.status === 'quota') {
      save();
      console.log(`\n\nDaily quota reached after ${done} descriptions.`);
      console.log('Progress is saved. Re-run `npm run caption-figures` after the quota resets.');
      return;
    } else {
      failed++;
    }

    if ((done + skipped + failed) % 5 === 0) save();
    process.stdout.write(`\r  ${done} described, ${skipped} skipped, ${failed} failed`);
    await sleep(PAUSE_MS);
  }

  save();
  process.stdout.write('\n');
  console.log(`\n${done} figures described`);
  if (skipped) console.log(`${skipped} had no subject matter and were marked to ignore`);
  if (failed) console.log(`${failed} failed and will be retried on the next run`);
  console.log('\nNext: npm run ingest');
}

main().catch(err => {
  console.error('\nCaptioning failed:', err);
  process.exit(1);
});
