# PlantDoc

Ask a plant pathology textbook directly, and get answers that cite the page they came from.

PlantDoc indexes a whole book and answers questions from it — hybrid semantic + lexical
retrieval, cross-encoder reranking, and a streaming chat interface where every claim links
back to the passage it was drawn from. You can also attach a photo of an affected plant and
have it read alongside the question.

Built on Next.js 16, Pinecone and Gemini, and sized to run entirely on free tiers.

---

## Why it is built this way

Most retrieval demos embed a PDF, search it with cosine similarity and hand the top five
chunks to a model. That works until it meets a real textbook. These are the problems that
shaped this version.

**A plant pathology book is full of words that mean nothing in a vector space.**
*Phytophthora infestans*, mefenoxam, *Xanthomonas campestris* pv. *vesicatoria* — dense
embeddings blur these into "oomycete-ish" neighbourhoods, so a question about one pathogen
retrieves passages about a different one. So retrieval runs twice: a dense index for meaning
and a sparse lexical index for exact terms, fused by reciprocal rank. Dense alone also cannot
bridge a Bangla question to English prose, and sparse alone cannot match a paraphrase. Both
ranks are needed.

**Retrieval scores are not relevance.** A chunk that *mentions* late blight scores nearly as
well as one that *explains how to manage it*. The bi-encoders that make the index compare a
query vector to a passage vector computed in isolation. A cross-encoder reads the question and
the passage together, so the fused candidates are reranked before anything reaches the model,
and anything below a relevance floor is dropped — which is what lets the assistant say "the
book does not cover this" instead of improvising from six loose matches.

**A bibliography out-scores real prose.** Reference lists and back-of-book indexes are dense
with disease names and explain nothing. The whole book is indexed, but every chunk is tagged
`body`, `references`, `glossary`, `index`, `toc` or `front`, and only explanatory text can
ground an answer.

**A citation you cannot check is just a number.** Every `[n]` marker in an answer opens the
exact passage the model was given, with its printed page number. Page numbers come from the
page the book prints on itself, not the PDF's own index.

**Half of a pathology book is pictures, and none of it is in the text.** What a lesion margin
looks like, how a halo grades into necrosis, the shape of a leaf spot — a plate carries that
and the prose around it says "Figure 11-27". So figures are extracted, paired with their
printed caption, and described by a vision model in the vocabulary a pathologist would search
with. Those descriptions are embedded alongside the text, which means a reader can search by
what they can see but cannot name.

**Latency is a correctness problem.** An answer that takes three minutes is indistinguishable
from a crash. Answers stream, retrieval progress is visible while it happens, every upstream
call has a timeout, and a model that returns nothing is replaced before a single byte reaches
the browser.

---

## Pipeline

```
          question (+ optional photo, + conversation history)
                              │
                              ▼
        ┌──────── query rewriting (Gemini Flash-Lite) ────────┐
        │  resolves follow-ups · translates to English        │
        │  describes the attached photo                       │
        └─────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     dense retrieval                  sparse retrieval
  llama-text-embed-v2              pinecone-sparse-english-v0
   semantic neighbours               exact term matches
              │                               │
              └───────────────┬───────────────┘
                              ▼
               reciprocal rank fusion (k = 60)
                              │
                              ▼
          cross-encoder rerank · bge-reranker-v2-m3
                              │
                              ▼
                  relevance floor, top 8 passages
                              │
                              ▼
        Gemini · grounded synthesis with inline [n] citations
                              │
                              ▼
                  NDJSON event stream → browser
```

Both retrievers run concurrently and either may fail without sinking the turn. If the
reranker is unavailable — the free tier allows 500 calls a month — fused order is used
instead. Nothing in the chain throws away an answer it could still give.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| App | Next.js 16 (App Router), React 19, Tailwind v4 | Streaming route handlers, no separate backend |
| Dense retrieval | Pinecone serverless + `llama-text-embed-v2` (1024d) | Multilingual, hosted, no second vendor quota |
| Sparse retrieval | Pinecone serverless + `pinecone-sparse-english-v0` | Exact matching for binomials and actives |
| Reranking | `bge-reranker-v2-m3` | The cross-encoder available on the free plan |
| Generation | Gemini Flash-Lite → Flash fallback chain | Minimal thinking level, streamed, timeout-bounded |
| PDF parsing | Poppler `pdftotext` / `pdfimages` | Deterministic, fast, no model in the text path |
| Figure indexing | Gemini vision → embedded descriptions | No hosted image-embedding model on these tiers |

Embeddings are generated by Pinecone, so ingestion needs only `PINECONE_API_KEY` and consumes
no Gemini quota.

---

## Running it

### 1. Install

```bash
npm install
cp .env.example .env.local
```

Fill in `.env.local`:

```
GEMINI_API_KEY=...     # https://aistudio.google.com/apikey
PINECONE_API_KEY=...   # https://app.pinecone.io/
```

### 2. Index a book

The textbook PDF is not in this repository. Use your own copy.

```bash
brew install poppler                      # or: apt install poppler-utils
npm run index-book -- /path/to/book.pdf
```

That runs extraction and ingestion together. To run them separately:

```bash
npm run extract -- /path/to/book.pdf      # → data/processed/chunks.json
npm run ingest -- --fresh                 # embeds + upserts both indexes
```

Extraction is book-agnostic. Running heads are found by looking for lines that repeat across
many pages, headings by typographic shape, and the printed-page offset by reading the numbers
the book prints on itself — nothing is hard-coded to a particular title. If citations come out
wrong, override the offset with `--page-offset N`.

### 2b. Index the figures (optional)

```bash
npm run extract-figures -- /path/to/book.pdf   # pulls plates, pairs captions
npm run caption-figures                        # vision model describes each
npm run ingest                                 # folds them into both indexes
```

Figures become searchable by appearance: a query like "brown necrotic lesions with a yellow
chlorotic halo, coalescing" retrieves the right plates. Captioning costs one vision call per
figure, so `--limit N` keeps a run inside the daily free quota; progress is saved after every
batch and re-running resumes.

**The images themselves are not published by default.** Reproducing a copyrighted textbook's
plates on a public site is a different proposition from quoting a paragraph, so only the
derived description is indexed and the figure is cited by number and page. For a book whose
licence allows it, pass `--publish` to `extract-figures` (needs `npm i -D sharp`) and the
plates are written to `public/figures/` and shown in the source panel.

Ingestion is resumable: embeddings are cached in `data/processed/embeddings.jsonl` keyed by
chunk text, so an interrupted run picks up where it stopped and re-extracting only re-embeds
the chunks that actually changed.

### 3. Check it

```bash
npm run doctor
npm run doctor -- "how do I manage late blight in tomato?"
```

Runs the real retrieval path and prints what came back — index dimensions, vector counts,
embedding latency, how many candidates each retriever found, how many both found, rerank
scores and the top passages. A retrieval problem shows up here without opening the UI.

### 4. Develop

```bash
npm run dev
```

### Recovering a corpus without the PDF

If a book is already indexed somewhere but the source PDF is gone, the chunk text is in the
vector metadata and can be pulled back out:

```bash
npm run export-index -- --index <old-index-name> --namespace ''
npm run ingest -- --fresh
```

---

## Deploying

1. Push to GitHub and import the repository at [vercel.com](https://vercel.com).
2. Set `GEMINI_API_KEY` and `PINECONE_API_KEY` in the project's environment variables.
3. Deploy.

Indexing happens on your machine, not in the build, so deployments stay fast and the free
Vercel plan is enough. `GET /api/health?deep=1` reports configuration and live vector counts.

---

## Configuration

Everything tunable is in [`src/lib/config.ts`](src/lib/config.ts) — model chains, index names,
candidate counts, the fusion constant, the relevance floor, timeouts and upload limits. The
ingestion scripts read the same index and model names from the environment, so an index can
never be built with one embedding model and queried with another.

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Required. Generation and query rewriting. |
| `PINECONE_API_KEY` | — | Required. Retrieval and ingestion. |
| `PINECONE_DENSE_INDEX` | `plantdoc-dense` | Semantic index name. |
| `PINECONE_SPARSE_INDEX` | `plantdoc-sparse` | Lexical index name. |
| `PINECONE_NAMESPACE` | `default` | One namespace per book edition. |
| `NEXT_PUBLIC_BOOK_TITLE` | `Plant Pathology` | Shown in the interface. |
| `NEXT_PUBLIC_BOOK_SUBTITLE` | `Agrios, 5th edition` | Shown in the interface. |

---

## Free-tier limits worth knowing

- **Pinecone Starter**: 2 GB storage, 5 indexes, 1M read units and 2M write units a month,
  5M embedding tokens per model, and **500 rerank calls a month**. A 950-page book is roughly
  2,700 vectors — about 15 MB. Reranking is the binding constraint, and the app degrades to
  fused order when it runs out.
- **Gemini free tier**: per-model daily request caps. The answer chain tries four models in
  order, so one model running out does not end the day.

---

## Limitations

- Answers are only as good as the indexed book. Nothing is retrieved from the open web.
- Photo diagnosis is provisional. A visual match narrows the possibilities; it does not
  confirm a pathogen, and the interface says so.
- Chemical recommendations are textbook guidance, not product labels. Local registration and
  label rates govern actual use.
- Figures are indexed as descriptions, not as image vectors. No hosted embedding model covers
  images and text in one space on these free tiers, and a written description is in any case
  closer to how someone phrases a symptom than raw pixels are. Vector line art that is drawn
  rather than embedded as a bitmap is not extractable this way.
- The reranker is English-only, so a Bangla question is rewritten to English for retrieval
  before the answer is written back in Bangla.

---

## Licence

MIT for the code. The indexed book is not included and remains under its own copyright.
