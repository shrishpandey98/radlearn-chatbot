# RadLearn — Radiology CME AI Assistant
### Architecture Blueprint for a Solo, Zero-Budget Founder

---

> [!IMPORTANT]
> This document is designed for a **non-technical solo founder**. Every recommended tool is either free, open-source, or has a generous free tier. Paid alternatives are explicitly called out. Build order is prioritized so you can launch an MVP in **2–4 weeks**.

---

## System Overview

RadLearn answers radiologist questions by combining:
1. **Your curated knowledge base** (PDFs, guidelines, textbooks, articles)
2. **A retrieval engine** that finds the most relevant passages
3. **An LLM** that synthesizes a grounded, cited answer
4. **An image layer** that surfaces relevant figures from source documents

The core pattern is **RAG (Retrieval-Augmented Generation)**: the LLM never "invents" facts — it only summarizes what your documents say, with citations.

---

## A. Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RADLEARN SYSTEM                          │
│                                                                 │
│  ┌──────────────┐     ┌──────────────────────────────────────┐  │
│  │   INGESTION  │     │            QUERY ENGINE              │  │
│  │   PIPELINE   │     │                                      │  │
│  │              │     │  User ──► Query Interface (Chatbot)  │  │
│  │  PDFs        │     │               │                      │  │
│  │  Websites    │     │               ▼                      │  │
│  │  Articles    │     │       Query Preprocessor             │  │
│  │  Guidelines  │     │     (clean + expand query)           │  │
│  │  Images      │     │               │                      │  │
│  │     │        │     │       ┌───────┴────────┐             │  │
│  │     ▼        │     │       ▼                ▼             │  │
│  │  Parser      │     │  Text Retriever   Image Retriever    │  │
│  │  Chunker     │     │  (Vector Search)  (Metadata Search)  │  │
│  │  Embedder    │     │       │                │             │  │
│  │  Indexer     │     │       └───────┬────────┘             │  │
│  │     │        │     │               ▼                      │  │
│  └──┬───────────┘     │          Re-ranker                   │  │
│     │                 │     (pick best K chunks)             │  │
│     ▼                 │               │                      │  │
│  ┌──────────────┐     │               ▼                      │  │
│  │  DATA STORES │     │         LLM + Prompt                 │  │
│  │              │     │    (grounded answer synthesis)       │  │
│  │  Vector DB   │◄────┤               │                      │  │
│  │  (Supabase/  │     │               ▼                      │  │
│  │   Chroma)    │     │    Answer + Citations + Images       │  │
│  │              │     │               │                      │  │
│  │  File Store  │     │               ▼                      │  │
│  │  (local /    │     │          Response UI                 │  │
│  │   Drive)     │     │  (structured text + image cards)     │  │
│  │              │     └──────────────────────────────────────┘  │
│  │  Metadata DB │                                               │
│  │  (SQLite /   │                                               │
│  │   Supabase)  │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## B. Recommended Technology Stack

> **Guiding principle**: Use managed/hosted services for everything you can. Never manage a server yourself in the MVP.

| Layer | MVP Choice (Free) | Why | Paid Alternative |
|---|---|---|---|
| **Chat UI** | [Streamlit](https://streamlit.io) (free cloud) | Python-native, zero frontend code, deploys in minutes | Vercel + Next.js |
| **Orchestration / RAG** | [LangChain](https://langchain.com) or [LlamaIndex](https://llamaindex.ai) | Most popular RAG frameworks, huge community | Cohere Coral, Azure AI Search |
| **LLM** | Google Gemini 1.5 Flash (free tier: 1M tokens/day) | Best free quota, long context window (1M tokens) | OpenAI GPT-4o ($$$) |
| **Embeddings** | `text-embedding-004` (Google, free) or `nomic-embed-text` (local) | Free, high quality, matches Gemini ecosystem | OpenAI `text-embedding-3-large` |
| **Vector Database** | [Supabase pgvector](https://supabase.com) (free tier: 500MB) | SQL + Vector in one, free hosting, no infra | Pinecone, Weaviate Cloud |
| **Metadata DB** | Supabase PostgreSQL (same instance) | Already included with Supabase free tier | PlanetScale, Railway |
| **File Storage** | Supabase Storage (free 1GB) or Google Drive | Simple, free, accessible | AWS S3 |
| **PDF Parser** | [PyMuPDF](https://pymupdf.readthedocs.io) (open-source) | Best PDF text + image extraction, free | Adobe PDF API |
| **Web Scraper** | [Firecrawl](https://firecrawl.dev) (free tier) or `newspaper3k` | Clean article extraction | Diffbot |
| **Document Chunker** | LangChain `RecursiveCharacterTextSplitter` | Built into LangChain, free | — |
| **Re-ranker** | [Cohere Rerank](https://cohere.com) (free tier: 1k calls/mo) or `cross-encoder/ms-marco` (local) | Dramatically improves retrieval quality | Cohere Rerank (paid) |
| **Deployment** | [Streamlit Community Cloud](https://streamlit.io/cloud) (free) | Zero-config deployment from GitHub | Railway, Render, Fly.io |
| **Auth (future)** | Supabase Auth (free tier) | Built in, no extra config | Auth0, Clerk |

> [!NOTE]
> **Gemini 1.5 Flash free tier** gives you 15 RPM and 1 million tokens per day — more than enough for an MVP with dozens of daily active users. This alone eliminates your LLM cost.

---

## C. Knowledge Ingestion Architecture

This pipeline runs **offline** (you trigger it manually or on a schedule). It transforms raw documents into searchable, embeddable chunks.

```
Raw Source
    │
    ▼
┌─────────────────────────────────────────┐
│ STEP 1: PARSING                         │
│                                         │
│  PDF  ──► PyMuPDF                       │
│         • Extract text by page          │
│         • Extract images (figures)      │
│         • Extract captions              │
│                                         │
│  URL  ──► Firecrawl / newspaper3k       │
│         • Clean article text            │
│         • Remove boilerplate            │
│                                         │
│  DOCX ──► python-docx                   │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ STEP 2: METADATA TAGGING                │
│                                         │
│  Per document, store:                   │
│  • source_title                         │
│  • source_url / filename                │
│  • publication_year                     │
│  • specialty_tag (e.g. "neuro", "breast")│
│  • document_type (guideline, textbook,  │
│    case report, review)                 │
│  • page_number                          │
│  • figure_ids (list of image refs)      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ STEP 3: CHUNKING                        │
│                                         │
│  Strategy: Semantic chunking            │
│  • Chunk size: 512 tokens               │
│  • Overlap: 64 tokens                   │
│  • Never break mid-sentence             │
│  • Keep section headings with chunks    │
│                                         │
│  Each chunk stores:                     │
│  { text, metadata, chunk_id,            │
│    doc_id, page_num, section_heading }  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ STEP 4: EMBEDDING                       │
│                                         │
│  text-embedding-004 (Google, free)      │
│  → 768-dimensional vector per chunk     │
│                                         │
│  Image captions also embedded           │
│  → enables image retrieval by text      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ STEP 5: INDEXING                        │
│                                         │
│  Vectors → Supabase pgvector table      │
│  Metadata → Supabase PostgreSQL table   │
│  Images → Supabase Storage bucket       │
│  Image metadata → PostgreSQL table      │
└─────────────────────────────────────────┘
```

### Key Ingestion Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Chunk size | 512 tokens | Balances context richness vs. retrieval precision |
| Overlap | 64 tokens | Prevents information loss at chunk boundaries |
| Image handling | Extract + store separately, embed captions | Enables text-to-image retrieval without multimodal LLM |
| Deduplication | Hash content before inserting | Prevents duplicate answers from same source |
| Incremental ingestion | Track ingested doc hashes in DB | Allows adding new docs without re-processing all |

---

## D. Retrieval Architecture

```
User Query: "MRI findings of multiple sclerosis"
       │
       ▼
┌─────────────────────────────────┐
│ Query Preprocessor              │
│                                 │
│ 1. Spell-check medical terms    │
│ 2. Expand abbreviations         │
│    (MS → multiple sclerosis)    │
│ 3. Generate sub-questions       │
│    (HyDE: hypothetical document │
│     embedding for better recall)│
└────────────┬────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌────────────┐  ┌────────────────┐
│  Semantic  │  │  Keyword / BM25│
│  Search    │  │  Search        │
│            │  │                │
│  pgvector  │  │  Full-text     │
│  cosine    │  │  search        │
│  similarity│  │  (Supabase)    │
└─────┬──────┘  └──────┬─────────┘
      │                │
      └───────┬─────────┘
              │
              ▼
    ┌──────────────────┐
    │    Fusion /      │
    │  Re-ranking      │
    │                  │
    │  Reciprocal Rank │
    │  Fusion (RRF)    │
    │  + Cohere Rerank │
    │  (optional)      │
    └────────┬─────────┘
             │
             ▼
    Top 5–8 most relevant chunks
    (with full metadata)
```

### Retrieval Strategy: Hybrid Search

Using **both** semantic search AND keyword search gives dramatically better results than either alone:

- **Semantic search**: Finds conceptually related content ("brain lesions" matches "demyelinating plaques")
- **Keyword search**: Finds exact medical terms ("BI-RADS 4", "Dawson's fingers")
- **Fusion**: Combines ranked lists from both, removes duplicates

---

## E. Citation Architecture

Every answer chunk must be traceable to its source. The citation system works as follows:

```
Retrieved Chunk Object:
{
  "chunk_id": "abc123",
  "text": "Periventricular plaques aligned along medullary veins...",
  "source_title": "Radiopaedia - Multiple Sclerosis",
  "source_url": "https://radiopaedia.org/articles/multiple-sclerosis",
  "page_number": null,
  "publication_year": 2023,
  "figure_refs": ["fig_001", "fig_002"],
  "relevance_score": 0.91
}
```

### Citation Display Format

The LLM prompt instructs it to cite inline using `[1]`, `[2]` etc. The UI then renders:

```
Answer:
Multiple sclerosis on MRI classically shows periventricular white 
matter lesions ("Dawson's fingers") oriented perpendicular to the 
corpus callosum [1]. The McDonald criteria require ≥1 T2 lesion in 
≥2 of 4 characteristic locations [2].

Sources:
[1] Radiopaedia — Multiple Sclerosis (2023) → [View Source ↗]
[2] MAGNIMS-CMSC-NAIMS MS Lesion Guidelines (2021) → [View PDF ↗]
```

### Citation Truthfulness Guardrails

| Guardrail | Implementation |
|---|---|
| Source-grounded answers only | Prompt engineering: "Only use information from provided context" |
| Confidence scoring | Pass relevance scores to LLM; flag low-confidence answers |
| "I don't know" fallback | If top chunk similarity < 0.7, return "Not found in knowledge base" |
| No web search | LLM cannot access internet; only retrieves from your curated docs |

---

## F. Image Retrieval Architecture

Images (figures, diagrams, MRI/CT examples) are stored separately and retrieved in parallel with text.

```
Ingestion Time:
PDF Page with Figure
       │
       ▼
  PyMuPDF extracts image → saved to Supabase Storage
       │
       ▼
  Caption text extracted → embedded as text vector
       │
       ▼
  Image metadata stored in DB:
  {
    image_id, doc_id, page_num,
    caption_text, caption_embedding,
    storage_url, figure_number,
    specialty_tag
  }

Query Time:
User Query → embed → cosine similarity against caption_embeddings
       │
       ▼
  Top 3 matching images retrieved
       │
       ▼
  Displayed as image cards below answer
```

### Image Display in UI (Streamlit)

```python
# Streamlit renders images inline
st.image(image_url, caption="Fig 2: Periventricular MS lesions (T2 FLAIR)")
```

> [!NOTE]
> **You do NOT need a multimodal LLM** to display images. You only need to match the user's query to image captions using text embeddings. This keeps costs at zero.

### What Images to Include

| Source | Image Type | Value |
|---|---|---|
| Radiopaedia | Case images with labels | High — real clinical examples |
| ACR guidelines | Diagrams, flowcharts | High — authoritative |
| Textbook PDFs | Annotated figures | High — educational |
| CT/MRI atlases | Reference anatomy | Medium |

> [!WARNING]
> **Copyright**: Only include images from sources that permit reproduction (Creative Commons, open-access journals, or documents you own). Do not scrape Radiopaedia images directly — link to them instead.

---

## G. Data Storage Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   SUPABASE (Free Tier)                  │
│                                                         │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │  PostgreSQL Tables  │  │     pgvector Extension   │  │
│  │                     │  │                          │  │
│  │  documents          │  │  chunk_embeddings        │  │
│  │  ├─ id              │  │  ├─ id                   │  │
│  │  ├─ title           │  │  ├─ chunk_id             │  │
│  │  ├─ source_url      │  │  ├─ embedding (vector)   │  │
│  │  ├─ doc_type        │  │  ├─ text                 │  │
│  │  ├─ specialty       │  │  ├─ doc_id (FK)          │  │
│  │  ├─ year            │  │  ├─ page_num             │  │
│  │  └─ content_hash    │  │  └─ section_heading      │  │
│  │                     │  │                          │  │
│  │  images             │  │  image_embeddings        │  │
│  │  ├─ id              │  │  ├─ id                   │  │
│  │  ├─ doc_id (FK)     │  │  ├─ image_id             │  │
│  │  ├─ storage_url     │  │  ├─ caption_embedding    │  │
│  │  ├─ caption         │  │  └─ (vector)             │  │
│  │  ├─ figure_num      │  │                          │  │
│  │  └─ page_num        │  └──────────────────────────┘  │
│  └─────────────────────┘                                │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │           Supabase Storage Buckets              │    │
│  │                                                 │    │
│  │  /documents  → original PDFs                   │    │
│  │  /images     → extracted figures               │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘

Local Development:
  ChromaDB (in-memory/local file) → zero config, zero cost
  SQLite → metadata storage
  Local folder → PDFs and images
```

### Storage Sizing Estimates (Free Tier)

| Component | MVP (100 docs) | Scale (5,000 docs) |
|---|---|---|
| Vector DB (chunks) | ~10MB | ~500MB |
| Metadata DB | ~2MB | ~100MB |
| Image storage | ~200MB | ~10GB |
| **Supabase free limit** | **500MB DB, 1GB storage** | Upgrade needed at scale |

> [!TIP]
> Supabase Pro ($25/month) gives you 8GB DB + 100GB storage — enough for thousands of radiology documents.

---

## H. User Query Flow

End-to-end flow from question to answer, step by step:

```
1. USER TYPES QUERY
   "What are the MRI findings of multiple sclerosis?"
           │
           ▼
2. QUERY PREPROCESSING (< 100ms)
   • Lowercase + normalize
   • Medical abbreviation expansion (MS → multiple sclerosis)
   • Optional: generate 2-3 sub-questions via LLM for better recall
           │
           ▼
3. PARALLEL RETRIEVAL (< 500ms)
   ┌───────────────────────────────┐
   │ Text Search                   │
   │ → embed query                 │
   │ → cosine similarity vs chunks │
   │ → top 15 candidates           │
   └───────────────┬───────────────┘
   ┌───────────────▼───────────────┐
   │ Image Search                  │
   │ → cosine similarity vs        │
   │   caption embeddings          │
   │ → top 3 image candidates      │
   └───────────────────────────────┘
           │
           ▼
4. RE-RANKING (< 200ms)
   • Cohere Rerank OR cross-encoder local model
   • Select top 5 most relevant text chunks
   • Filter chunks below similarity threshold (0.65)
           │
           ▼
5. PROMPT CONSTRUCTION
   System prompt:
   "You are RadLearn, a radiology CME assistant.
    Answer only from the provided context.
    Cite each fact with [1], [2] etc.
    If answer not in context, say so clearly."
   
   User prompt:
   [CONTEXT: chunk1, chunk2, ..., chunk5]
   [QUESTION: user query]
           │
           ▼
6. LLM GENERATION (1–3 seconds)
   Gemini 1.5 Flash processes prompt
   Returns: structured answer with inline citations
           │
           ▼
7. RESPONSE FORMATTING
   • Parse inline citations → match to source metadata
   • Render answer in Streamlit
   • Display source cards with links
   • Display image cards (if images retrieved)
           │
           ▼
8. DISPLAYED TO USER
   ✓ Formatted answer
   ✓ Numbered source citations
   ✓ Clickable source links / PDF page links
   ✓ Relevant images with captions
   ✓ Confidence indicator (if low similarity)
```

---

## I. MVP vs. Future Scalable Architecture

### MVP Architecture (Week 1–4, Zero Cost)

**Goal**: Prove the concept works. Get feedback from 5–10 radiologists.

```
┌──────────────────────────────────────┐
│           MVP STACK                  │
│                                      │
│  UI: Streamlit Community Cloud       │
│  RAG: LlamaIndex (simple pipeline)   │
│  LLM: Gemini 1.5 Flash (free API)    │
│  Embeddings: text-embedding-004      │
│  Vector DB: Supabase pgvector        │
│  Storage: Supabase Storage           │
│  Ingestion: Python script (manual)   │
│  Auth: None (share link only)        │
│  Monitoring: None                    │
│                                      │
│  Knowledge base: 20–50 hand-picked  │
│  radiology documents / guidelines   │
│                                      │
│  Total cost: $0/month               │
└──────────────────────────────────────┘
```

**What the MVP does NOT have** (intentionally):
- ❌ User accounts / auth
- ❌ Feedback / rating system
- ❌ Admin document upload UI (use Python script)
- ❌ Advanced re-ranking
- ❌ Query analytics
- ❌ Multiple specialties

---

### Scalable Architecture (Month 3–12, ~$25–100/month)

```
┌──────────────────────────────────────────────────────┐
│                SCALED STACK                          │
│                                                      │
│  UI: Next.js on Vercel (free tier)                   │
│  OR: Streamlit with custom theme (still viable)      │
│                                                      │
│  RAG: LlamaIndex with advanced pipeline              │
│  LLM: Gemini 1.5 Pro (better quality, ~$7/1M tokens) │
│  Embeddings: text-embedding-004                      │
│  Vector DB: Supabase Pro ($25/mo) OR                 │
│             Weaviate Cloud (free 14k objects)        │
│                                                      │
│  Ingestion: Automated pipeline triggered by          │
│             document upload via UI                   │
│  Auth: Supabase Auth (free)                          │
│  Admin UI: Simple Streamlit admin page               │
│  Analytics: PostHog (free tier)                      │
│  Monitoring: LangSmith (free tier) for LLM traces    │
│                                                      │
│  Knowledge base: 500–5,000 documents                 │
│  Specialties: Neuro, Breast, Chest, MSK, Abdominal   │
│                                                      │
│  Total cost: ~$25–50/month                           │
└──────────────────────────────────────────────────────┘
```

### Architecture Evolution Timeline

```
PHASE 1 (Weeks 1–4)     PHASE 2 (Months 2–3)    PHASE 3 (Months 4–12)
─────────────────────   ─────────────────────   ─────────────────────
Streamlit + LlamaIndex  Add Supabase Auth       Next.js frontend
Manual ingestion script Add upload UI           Automated ingestion
20–50 curated docs      100–500 docs            1,000–5,000 docs
Gemini Flash free       Gemini Flash still      Gemini Pro for premium
No auth                 Basic auth              Role-based access
No analytics            Basic logging           Full analytics stack
Single specialty        3 specialties           All specialties
Share via link          Invite-only beta        Public launch
```

---

## J. Build Order — What to Build First

### 🟢 Build Now (Week 1–2): Core RAG Pipeline

| Priority | Component | Tool | Effort |
|---|---|---|---|
| 1 | Knowledge ingestion script | Python + PyMuPDF | 2–3 hours |
| 2 | Vector DB setup | Supabase pgvector | 1 hour |
| 3 | Basic RAG query chain | LlamaIndex | 2–3 hours |
| 4 | Streamlit chat UI | Streamlit | 2 hours |
| 5 | Gemini API integration | google-generativeai | 1 hour |
| 6 | Basic citation display | Python string formatting | 1–2 hours |

> **Total estimate: ~10–12 hours of guided work using AI coding tools**

---

### 🟡 Build Next (Week 3–4): Quality & Images

| Priority | Component | Tool | Effort |
|---|---|---|---|
| 7 | Image extraction + storage | PyMuPDF + Supabase | 3–4 hours |
| 8 | Image retrieval via captions | pgvector | 2 hours |
| 9 | Hybrid search (BM25 + vector) | LlamaIndex hybrid | 2–3 hours |
| 10 | Similarity threshold filter | Python | 1 hour |
| 11 | Specialty tag filtering | Supabase filter | 1 hour |

---

### 🔵 Build Later (Month 2–3): Scale & Polish

| Priority | Component | Tool | Effort |
|---|---|---|---|
| 12 | User auth | Supabase Auth | 3–4 hours |
| 13 | Document upload UI | Streamlit file uploader | 2–3 hours |
| 14 | Re-ranker | Cohere Rerank API | 2 hours |
| 15 | Query analytics | PostHog | 1–2 hours |
| 16 | LLM call monitoring | LangSmith | 1 hour |
| 17 | User feedback (thumbs up/down) | Supabase table | 2 hours |

---

### 🔴 Do NOT Build Until Paid Traction

| Component | Reason to Postpone |
|---|---|
| Custom fine-tuned LLM | $$$, premature, Gemini is sufficient |
| Multimodal image understanding | Complex, expensive, not needed for MVP |
| Real-time PACS integration | Requires hospital IT partnerships |
| Mobile app | Web-first is sufficient for CME |
| Custom embedding model | Pre-trained models are excellent |
| Self-hosted LLM (Llama) | Requires GPU server, complex to maintain |

---

## Paid vs. Free Component Summary

| Component | Free Option | Paid Option | When to Upgrade |
|---|---|---|---|
| LLM | Gemini 1.5 Flash (free tier) | Gemini 1.5 Pro ($7/1M tokens) | When quality complaints arise |
| Vector DB | Supabase free (500MB) | Supabase Pro ($25/mo) | At ~1,000 documents |
| Embeddings | text-embedding-004 (free) | — | Never (free is excellent) |
| Re-ranker | Local cross-encoder (free) | Cohere Rerank ($) | If retrieval quality is poor |
| Hosting | Streamlit Cloud (free) | Railway / Render (~$5/mo) | If you need custom domain |
| Auth | Supabase Auth (free) | — | Never (free tier is sufficient) |
| Monitoring | LangSmith free tier | LangSmith Plus ($39/mo) | At scale |
| Web scraping | Firecrawl free (500 pages/mo) | Firecrawl ($19/mo) | If ingesting many websites |

---

## Recommended Starter Knowledge Sources (Free, Citable)

| Source | Content Type | Access |
|---|---|---|
| [Radiopaedia.org](https://radiopaedia.org) | Case-based radiology encyclopedia | Free to read, link-to |
| [ACR Appropriateness Criteria](https://www.acr.org/Clinical-Resources/ACR-Appropriateness-Criteria) | Clinical guidelines | Free PDFs |
| [RSNA Case Collection](https://cases.rsna.org) | Teaching cases | Free |
| [LearningRadiology.com](https://learningradiology.com) | Teaching files | Free |
| [OpenMD Radiology](https://openmd.com) | Articles | Free |
| [PubMed Open Access](https://pubmed.ncbi.nlm.nih.gov) | Research articles | Free (filter by Open Access) |
| [StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK430685/) | NCBI textbook chapters | Free |
| [ESR iRefer Guidelines](https://www.irefer.org.uk) | European guidelines | Free |

---

## One-Page Summary: The Simplest Possible MVP

```
1. Download 20 free radiology PDFs (ACR guidelines, StatPearls chapters)
2. Run ingestion script → stores chunks + embeddings in Supabase
3. User opens Streamlit app, types radiology question
4. LlamaIndex retrieves top 5 relevant chunks
5. Gemini 1.5 Flash generates a cited answer
6. Streamlit displays answer + clickable source links
7. Deploy to Streamlit Community Cloud — shareable URL in 5 minutes
```

**That's it. That's your MVP.**  
No servers. No DevOps. No monthly bills. Fully working RAG chatbot for radiologists.

---

*Document version: 1.0 | Architecture designed for zero-budget solo founder MVP*
