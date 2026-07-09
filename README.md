# RadLearn — AI Radiology CME Assistant

Evidence-based radiology education powered by RAG (Retrieval-Augmented Generation).

---

## Phase 1: Foundation — Setup Instructions

### 1. Prerequisites

- Python 3.11 or higher
- A Google AI Studio API key (free at https://aistudio.google.com/app/apikey)

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in:
```
GOOGLE_API_KEY=your-key-here
ADMIN_PASSWORD=choose-a-password
```

### 4. Initialise the database

Run this once:
```bash
python scripts/setup_database.py
```

Expected output:
```
✅  Data directories ready at: /path/to/radlearn-chatbot/data
✅  002_tables.sql executed
✅  003_indexes.sql executed
✅  ChromaDB collection 'radlearn_chunks' ready
✅  ChromaDB collection 'radlearn_images' ready
✅  Default knowledge source: 'Manual Uploads' (id: xxxxxxxx...)
Setup complete!
```

### 5. Launch the app

```bash
streamlit run app.py
```

Open http://localhost:8501 — you should see all green ✅ checks.

---

## Project Structure

```
radlearn-chatbot/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── .env.example                    # Copy to .env
├── .gitignore
│
├── .streamlit/
│   ├── config.toml                 # Dark theme
│   └── secrets.toml                # For Streamlit Cloud deployment
│
├── radlearn/                       # Core package
│   ├── config.py                   # All constants and env vars
│   ├── database/
│   │   ├── client.py               # ChromaDB + SQLite singletons
│   │   ├── documents.py            # Documents + knowledge sources CRUD
│   │   ├── chunks.py               # ChromaDB vector ops + FTS5 keyword search
│   │   ├── images.py               # Image caption embeddings
│   │   ├── conversations.py        # Conversations + messages CRUD
│   │   └── citations.py            # Citations CRUD
│   ├── ingestion/
│   │   ├── embedder.py             # Google text-embedding-004 wrapper
│   │   └── chunker.py              # RecursiveCharacterTextSplitter
│   ├── retrieval/                  # (Phase 2)
│   ├── chat/                       # (Phase 2)
│   └── ui/                         # (Phase 3)
│
├── sql/
│   ├── 001_extensions.sql          # Empty (no extensions needed)
│   ├── 002_tables.sql              # All SQLite table definitions
│   ├── 003_indexes.sql             # All SQLite indexes
│   └── 004_functions.sql           # Empty (no RPC functions needed)
│
├── scripts/
│   └── setup_database.py           # One-time DB initialisation
│
└── data/                           # Created automatically — not committed to git
    ├── radlearn.db                 # SQLite database
    ├── chroma/                     # ChromaDB vector store
    ├── documents/                  # Uploaded PDFs and DOCX files
    └── images/                     # Extracted PDF figures
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ Yes | Google AI Studio key for Gemini + embeddings |
| `ADMIN_PASSWORD` | Optional | Admin dashboard password (default: `radlearn-admin`) |
| `DATA_DIR` | Optional | Path for local storage (default: `./data`) |

---

## Manual Testing — Phase 1

After setup, test each layer individually:

### Test 1: Database connections
```bash
streamlit run app.py
# Open http://localhost:8501
# Verify: all 4 checks show ✅
```

### Test 2: Embedder
```bash
python3 -c "
from radlearn.ingestion.embedder import embed_single
v = embed_single('Multiple sclerosis MRI findings')
print(f'Vector length: {len(v)}')  # Must print: Vector length: 768
print(f'First 3 values: {v[:3]}')
"
```

### Test 3: Chunker
```bash
python3 -c "
from radlearn.ingestion.chunker import chunk_text
sample = '''
INTRODUCTION

Multiple sclerosis (MS) is a chronic inflammatory demyelinating disease of the
central nervous system. It affects approximately 2.8 million people worldwide.
MRI is the most sensitive imaging modality for detecting MS plaques.

IMAGING FINDINGS

Periventricular lesions are the hallmark of MS on MRI. They appear as ovoid
hyperintense lesions on T2 and FLAIR sequences, oriented perpendicular to the
lateral ventricles — the so-called Dawson fingers sign.
''' * 5

chunks = chunk_text(
    text=sample,
    document_id='test-doc-id',
    doc_metadata={'doc_title': 'MS Test', 'doc_type': 'article', 'doc_specialty': 'neuro'}
)
print(f'Chunks created: {len(chunks)}')
for i, c in enumerate(chunks[:3]):
    print(f'  Chunk {i}: {c[\"token_count\"]} tokens, header_only={c[\"is_header_only\"]}')
"
```

### Test 4: Documents CRUD
```bash
python3 -c "
from radlearn.database.documents import create_document, get_document, list_documents, delete_document_record

doc = create_document({
    'title': 'Test MS Article',
    'doc_type': 'article',
    'specialty': 'neuro',
    'license_type': 'open_access',
})
print(f'Created: {doc[\"id\"][:8]}...')

fetched = get_document(doc['id'])
print(f'Fetched: {fetched[\"title\"]}')

all_docs = list_documents()
print(f'Total docs: {len(all_docs)}')

delete_document_record(doc['id'])
print('Deleted OK')
"
```

### Test 5: ChromaDB round-trip (requires GOOGLE_API_KEY)
```bash
python3 -c "
from radlearn.ingestion.embedder import embed_single
from radlearn.database.chunks import add_chunks_to_chromadb, semantic_search, count_chunks
import uuid

# Create a test chunk with a real embedding
text = 'Periventricular lesions are the hallmark of MS on MRI T2 FLAIR sequences.'
vector = embed_single(text)

chunk = {
    'id': str(uuid.uuid4()),
    'document_id': str(uuid.uuid4()),
    'text': text,
    'embedding': vector,
    'chunk_index': 0,
    'page_number': 1,
    'section_heading': 'MRI Findings',
    'token_count': len(text.split()),
    'is_header_only': False,
    'doc_title': 'MS Radiology Review',
    'doc_author': 'Test Author',
    'doc_year': 2023,
    'doc_type': 'article',
    'doc_specialty': 'neuro',
    'source_url': '',
    'file_path': '',
    'citation_format': '',
}

add_chunks_to_chromadb([chunk])
print(f'Total chunks in ChromaDB: {count_chunks()}')

query_vector = embed_single('What are MRI findings in multiple sclerosis?')
results = semantic_search(query_vector, top_k=3)
print(f'Semantic search returned: {len(results)} results')
if results:
    print(f'Top result similarity: {results[0][\"similarity_score\"]}')
    print(f'Top result text: {results[0][\"text\"][:80]}...')
"
```

---

## Technology Stack (Phase 1)

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| UI | Streamlit 1.32 |
| Vector store | ChromaDB 0.5 (local, persistent) |
| Relational DB | SQLite via SQLAlchemy 2.0 |
| Embeddings | Google text-embedding-004 (768-dim) |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Token counting | tiktoken cl100k_base |

All data stored locally in `data/`. Zero cloud dependencies except the Google AI API.
