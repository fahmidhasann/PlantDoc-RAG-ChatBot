# PlantDoc 🌱 — Multimodal Plant Pathology AI

An advanced, multimodal document Q&A and plant disease diagnostic system using **Retrieval-Augmented Generation (RAG)** grounded in Agrios' authoritative *Plant Pathology (5th Edition)* textbook (948 pages).

Built with **Next.js 15 (App Router)**, **Google Gemini 3.8 Flash**, and **Pinecone Serverless Vector DB** — engineered for **100% Free Tier** operation with **zero cold-start** and instant Vercel deployment.

---

## ⚡ Key Highlights for AI Engineering & Agronomy

- **Multimodal Visual Diagnosis**: Upload leaf photos directly. Gemini 3.8 Flash detects visual symptoms (chlorosis, halos, water-soaked margins, fungal sporulation) and triggers grounded retrieval.
- **Authoritative Textbook Grounding**: Cites exact page numbers (e.g. `[Page 421]`) directly from Agrios' Plant Pathology to prevent hallucinations.
- **Serverless Hybrid Retrieval**: Combines semantic embeddings (`text-embedding-004`) via Pinecone Serverless with structured pathology metadata (pathogen, symptoms, cultural, and chemical controls).
- **100% Free Tier Architecture**:
  - **LLM**: Google AI Studio Gemini 3.8 Flash (15 RPM, 1M TPM free quota).
  - **Embeddings**: Google AI Studio `text-embedding-004` (Free tier).
  - **Vector DB**: Pinecone Serverless Free Tier (100,000 vectors on AWS `us-east-1`).
  - **Hosting**: Vercel Hobby Tier (Global Edge CDN, 0 cold-start, instant response).

---

## 🛠️ Architecture

```
User Query / Leaf Photo
         │
         ▼
Next.js 15 Serverless API Route (/api/chat)
         │
         ├──► Google Gemini text-embedding-004 (768-dim)
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
PINECONE_INDEX_NAME=plantdoc
```

*(Note: PlantDoc includes an embedded textbook pathology knowledge base out-of-the-box, allowing it to run and test immediately even before setting up Pinecone!)*

### 4. Run Development Server
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 📦 Ingesting Full Textbook to Pinecone

To embed the 948-page textbook chunks and upsert to your Pinecone index:
```bash
npm run ingest
```
The script will automatically create the `plantdoc` serverless index (768 dimensions, cosine metric) on AWS `us-east-1` and batch upload all textbook vectors.

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
