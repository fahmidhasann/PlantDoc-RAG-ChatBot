# PlantDoc 🌱 — Multimodal Plant Pathology AI

An advanced, multimodal document Q&A and plant disease diagnostic system using **Retrieval-Augmented Generation (RAG)** grounded in Agrios' authoritative *Plant Pathology (5th Edition)* textbook (948 pages).

Built with **Next.js 15 (App Router)**, **Google Gemini 3.8 Flash**, and **Pinecone Serverless Vector DB** — engineered for **100% Free Tier** operation with **zero cold-start** and instant Vercel deployment.

---

## ⚡ Key Highlights for AI Engineering & Agronomy

- **Multimodal Visual Diagnosis**: Upload leaf photos directly. Gemini 3.8 Flash detects visual symptoms (chlorosis, halos, water-soaked margins, fungal sporulation) and triggers grounded retrieval.
- **Authoritative Textbook Grounding**: Cites exact page numbers (e.g. `[Page 421]`) directly from Agrios' Plant Pathology to prevent hallucinations.
- **Full-Textbook Retrieval**: All 948 pages are chunked (~2,670 passages tagged with chapter, section and printed page) and embedded with Pinecone's hosted multilingual `llama-text-embed-v2` (768-dim), so Bangla questions find the English textbook passages too.
- **100% Free Tier Architecture**:
  - **LLM**: Google AI Studio Gemini 3.8 Flash, falling back through 3.7 / 3.6 / 3.5 Flash and Flash-Lite — each model has its own free daily quota (~20 requests/day for Flash models).
  - **Embeddings**: Pinecone Inference `llama-text-embed-v2` (Starter plan includes 5M tokens/month; the whole book is ~1M).
  - **Vector DB**: Pinecone Serverless Starter (2 GB on AWS `us-east-1`; the book uses ~15 MB).
  - **Hosting**: Vercel Hobby Tier (Global Edge CDN, 0 cold-start, instant response).

---

## 🛠️ Architecture

```
User Query / Leaf Photo
         │
         ▼
Next.js 15 Serverless API Route (/api/chat)
         │
         ├──► (photo or non-English question) Gemini Flash-Lite → English search line
         │
         ├──► Pinecone Inference llama-text-embed-v2 (768-dim query embedding)
         │           │
         │           ▼
         ├──► Pinecone Serverless Vector Search (Top-K Textbook Chunks)
         │           │
         ▼           ▼
Google Gemini 3.8 Flash (Multimodal Diagnosis + Grounded Synthesis)
         │
         ▼
Clinical Diagnosis + Management Protocol + Exact Page Citations
```

---

## 🚀 Quick Start (Local Development)

### 1. Prerequisites
- Node.js 18+ (tested on Node.js v22)
- Free Google AI Studio API Key: [aistudio.google.com](https://aistudio.google.com/)
- Free Pinecone API Key: [pinecone.io](https://app.pinecone.io/)

### 2. Clone and Install
```bash
git clone https://github.com/fahmidhasann/PlantDoc-RAG-ChatBot.git
cd PlantDoc-RAG-ChatBot
npm install
```

### 3. Setup Environment Variables
Create `.env.local`:
```env
GEMINI_API_KEY=your_google_ai_studio_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=plantdoc-book
```

*(Note: PlantDoc includes an embedded textbook pathology knowledge base out-of-the-box, allowing it to run and test immediately even before setting up Pinecone!)*

### 4. Run Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📦 Ingesting the Full Textbook into Pinecone

The textbook PDF is **not** in this repository (it is copyrighted). Use your own copy.

**One command:** `npm run index-book -- /path/to/textbook.pdf` (runs both steps below).

1. Install Poppler once (for `pdftotext`): `brew install poppler` (macOS) or `apt install poppler-utils`.
2. Split every page into overlapping, page-tagged chunks:
   ```bash
   npm run extract -- "/path/to/Agrios (2005) - Plant pathology 5. ed.pdf"
   ```
   Writes `data/processed/chunks.json` (git-ignored). The extractor:
   - maps PDF pages to the **printed page numbers** used in citations (auto-detected; override with `--page-offset N`),
   - detects the 16 chapters and each disease/section heading (e.g. *Late Blight of Potatoes*),
   - strips running headers, page numbers and figure panel letters,
   - tags every chunk with a `kind`: `body`, `references`, `glossary`, `index`, `toc` or `front`.

   For Agrios 5th ed. this gives ~2,670 chunks covering every page that has text (933 of 948 PDF pages;
   the rest are blank or part-divider pages).
3. Embed and upload every chunk:
   ```bash
   npm run ingest -- --fresh
   ```
   `--fresh` clears old vectors (after embedding succeeds). The script creates the `plantdoc-book` index
   (768 dims, cosine, AWS `us-east-1`) if needed. Embeddings are cached in `data/processed/embeddings.jsonl`,
   keyed by the chunk text — re-running resumes where it stopped and only re-embeds changed chunks.

**How the app searches:** every chunk is in Pinecone, but questions are answered from `body`, `glossary` and
`front` chunks only, so reference lists and the back-of-book index never crowd out real explanations.
For leaf photos, Gemini first writes a one-line description (crop, likely disease, symptoms) and that is
used to search the book.

**Free-tier fit:** ~2,700 vectors × 768 dims plus text metadata is ~15 MB — far below Pinecone Starter's
2 GB storage and 2M monthly write units. Embedding uses Pinecone's hosted `llama-text-embed-v2`
(~1M tokens for the whole book, within the Starter plan's 5M/month), so ingestion needs only
`PINECONE_API_KEY` and takes a few minutes; rate-limit pauses are retried automatically.
(Gemini's free embedding quota is only ~1,000 chunks/day, which is why it is not used.)

---

## 🌐 Deploy to Vercel (1-Click)

1. Push this repository to GitHub:
   ```bash
   git add .
   git commit -m "feat: upgrade to PlantDoc v2 (Next.js + Gemini 3.8 Flash + Pinecone RAG)"
   git push origin main
   ```
2. Go to [vercel.com](https://vercel.com) and import the repository.
3. In **Environment Variables**, add:
   - `GEMINI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`
4. Click **Deploy**. Your app is live with 100% uptime and instant load speeds!
