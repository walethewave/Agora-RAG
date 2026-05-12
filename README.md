<![CDATA[<div align="center">

# Agora — AI Governance Document Q&A System

**Production-Grade Retrieval-Augmented Generation (RAG) for the ETO AGORA Corpus**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000?logo=pinecone)](https://pinecone.io)
[![Gemini](https://img.shields.io/badge/Gemini_2.5-Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-Chat_UI-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Upstash](https://img.shields.io/badge/Upstash-Redis-00E9A3?logo=redis&logoColor=white)](https://upstash.com)

*Upload governance documents, index them into Pinecone, and ask natural language questions — grounded answers with source citations, multi-turn conversation memory, and sub-query decomposition.*

</div>

---

## Table of Contents

- [Assessment Alignment](#assessment-alignment)
- [Dataset](#dataset)
- [Architecture Overview](#architecture-overview)
- [Key Design Decisions & Tradeoff Reasoning](#key-design-decisions--tradeoff-reasoning)
- [Interface Usage Guide](#interface-usage-guide)
- [File Structure & Responsibility Split](#file-structure--responsibility-split)
- [System Constants](#system-constants)
- [API Endpoints](#api-endpoints)
- [Prompt Engineering](#prompt-engineering)
- [Conversation Memory Flow](#conversation-memory-flow)
- [Installation & Setup](#installation--setup)
- [Testing](#testing)
- [Example Queries](#example-queries)
- [Technology Stack](#technology-stack)
- [Known Limitations](#known-limitations)
- [Next Steps](#next-steps)

---

## Assessment Alignment

This repository is built to address the **ML Engineer Assessment** expectations across three pillars:

| Criteria | How This System Addresses It |
|---|---|
| **Correctness** | A fully runnable system that ingests `.txt`/`.pdf` governance documents, chunks them with overlap, embeds via Gemini, indexes into Pinecone, and answers questions end-to-end. 43 unit tests validate chunking, prompts, models, sub-query parsing, and error handling. |
| **Retrieval Quality** | Sub-query decomposition splits complex questions into 1–5 focused queries for parallel retrieval. Top-4 chunks per sub-query, deduplicated by text content, ranked by cosine similarity. 1536D embeddings capture semantic nuance in legal/policy language. Conversation history (last 5 exchanges) is injected for multi-turn coherence. |
| **Tradeoff Reasoning** | Every architectural choice is documented with explicit tradeoffs — chunk size (500 vs 1200 tokens), embedding dimension (1536 vs 768), sub-query latency (+2–3s for accuracy), session TTL (30 min), memory window (5 pairs), and namespace isolation (trust-based vs cryptographic). See [Key Design Decisions](#key-design-decisions--tradeoff-reasoning). |

**Additional deliverables:**
- Grounded answers with source citations (filename + cosine similarity score)
- Admin panel for document upload, database inspection, and namespace management
- Persistent chat UI with session history sidebar
- Example queries with real retrieval results

---

## Dataset

**Source:** [ETO AGORA AI Governance Documents Data](https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data)

The AGORA corpus is a living collection of AI-relevant laws, regulations, standards, and governance documents from the United States and around the world. This system was built and tested using **10 `.txt` files** from the `agora/fulltext/` directory (files `1.txt` through `10.txt`), producing **36 chunks** across the `policy` namespace.

To extend the corpus, download additional `.txt` files from the dataset and upload them via the Admin Panel. They are added to the existing index without overwriting anything.

---

## Architecture Overview

```
User Question
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1 · Sub-Query Decomposition (Gemini 2.5 Flash)    │
│  ├─ Conversational ("hi", "thanks") → skip RAG entirely │
│  ├─ Multi-part question → split into 1–5 sub-queries    │
│  └─ Single question → pass through as-is                │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2 · Parallel Embedding + Pinecone Retrieval       │
│  ├─ Each sub-query → Gemini Embedding (1536D)           │
│  ├─ Query Pinecone (top-4 chunks per sub-query, cosine) │
│  └─ Deduplicate by text content (first 100 chars)       │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3 · Context Assembly                              │
│  ├─ Merge unique chunks, sort by relevance score        │
│  ├─ Build source citations (filename, score)            │
│  └─ Inject last 5 conversation exchanges from Upstash   │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4 · Answer Synthesis (Gemini 2.5 Flash)           │
│  ├─ System prompt: AI governance analyst persona        │
│  ├─ User template: history + context + question         │
│  └─ Generate grounded answer with inline citations      │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5 · Response + Memory Update                      │
│  ├─ Return answer + sources to caller                   │
│  └─ Save Q&A pair to Upstash Redis (30-min TTL, max 5)  │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions & Tradeoff Reasoning

### 1. Sub-Query Decomposition — Accuracy-First Approach

Every user question passes through a Gemini API call that classifies intent and optionally decomposes:

| Input Type | Behavior | Example |
|---|---|---|
| **Conversational** | Returns `["__conversational__"]` — RAG skipped, Gemini answers directly | "hi", "thanks" |
| **Single question** | Returns `["original question"]` — one retrieval pass | "What are the reporting requirements?" |
| **Multi-part** | Returns `["sub-query 1", "sub-query 2", ...]` — parallel retrieval, results merged | "What are the barriers to AI adoption and who enforces compliance?" |

**Why this matters:** If a user asks "What are the reporting requirements and who enforces them?", a single embedding query retrieves chunks about *either* reporting *or* enforcement, not both. Decomposition ensures both aspects are retrieved and answered.

**Latency tradeoff:**

| Mode | Latency | Accuracy |
|---|---|---|
| Without sub-query decomposition | **~4 seconds** | Misses multi-faceted questions |
| With sub-query decomposition | **~6–10 seconds** | Both aspects retrieved and answered |

The +2–3 second overhead (one Gemini API call for classification) is a **deliberate engineering tradeoff** — accuracy over speed. For a governance Q&A system where correctness matters more than response time, this is justified.

---

### 2. Embedding Dimension: 1536D

`models/gemini-embedding-001` supports configurable output dimensions. **1536 was chosen** because:

- **Semantic precision:** Legal and policy language contains subtle distinctions (PROHIBITS vs. RECOMMENDS vs. PERMITS). 1536D captures these nuances better than 768D.
- **Recommended quality/speed balance:** Google's recommended tradeoff for Gemini Embedding 2.
- **Pinecone compatibility:** The free tier supports up to 1536D without additional cost.
- **Future flexibility:** If experiments show 768D is sufficient for a specific corpus, change `EMBEDDING_DIMENSION` in `src/simplified_rag.py` and recreate the Pinecone index.

**Tradeoff:** Higher dimensions mean slightly larger index storage and marginally slower similarity search. For a corpus of 36 chunks (or even thousands), this is negligible.

---

### 3. Chunk Size: 500 Tokens (Tuned for Short Documents)

The AGORA `.txt` files are short governance documents — typically 500–1,000 tokens each. The default 1,200-token chunk target would collapse each file into a single chunk, making retrieval meaningless (every query returns the same chunk).

| Setting | Chunk Target | Overlap | Result for 10 AGORA files |
|---|---|---|---|
| Default | 1,200 tokens | 20% (240 tokens) | ~10 chunks (one per file — no retrieval granularity) |
| **Tuned** | **500 tokens** | **20% (100 tokens)** | **36 chunks (2–4 per file — meaningful retrieval)** |

**Tradeoff:** Smaller chunks mean less context per chunk. Important context may be split across chunks in very long documents. The 20% overlap mitigates this by carrying the tail of the previous chunk into the next.

> **Recommendation:** If you add larger documents (full legislation PDFs, 30+ pages), increase `CHUNK_TARGET_TOKENS` back to 1200.

---

### 4. Conversation Memory — Upstash Redis (30-Minute TTL, 5-Message Window)

**Why Upstash:** Multi-turn Q&A requires context. Without memory, every question is answered in isolation and the model cannot reference prior exchanges. Upstash Redis is serverless — zero infrastructure to manage, auto-scales, and provides sub-millisecond latency.

**Implementation** (`src/utils.py` — `ConversationMemory`):

| Parameter | Value | Rationale |
|---|---|---|
| **Storage key** | `session:{session_id}:history` | Redis list, one per session |
| **Window size** | Last 5 user-assistant pairs (10 messages) | Keeps the Gemini context window focused. More than 5 exchanges would dilute retrieval relevance with old context. |
| **TTL** | 30 minutes per session | Auto-expires inactive sessions. Governance Q&A sessions are typically short research bursts, not hours-long conversations. 30 min balances memory retention vs. Redis storage costs. |
| **Injection** | Plain text in `{history}` placeholder | Formatted as `"User: ...\nAgora: ..."` and injected into `user_template` |

**Two separate Redis stores (by design):**

| Key Pattern | Purpose | TTL | Used By |
|---|---|---|---|
| `session:{id}:history` | Short-term model context (last 5 pairs) | **30 min** | RAG backend (`ConversationMemory`) |
| `agora:{id}:messages` | Full chat UI history (all messages) | **7 days** | Streamlit frontend (session restoration) |

This separation means the model always gets a **clean 5-message window** for context, while the UI can restore the full conversation history when a user returns to a previous session.

---

### 5. Multi-Tenant Namespace Isolation

A single Pinecone index (`cyber`) serves multiple document sets via **namespaces**. Each `entity_id` maps to a Pinecone namespace:

- `entity_id=policy` → all AGORA governance documents
- `entity_id=legal` → a different document set
- Upload, query, and clear operations are fully namespace-isolated

**Tradeoff:** Namespace isolation is **trust-based, not cryptographic**. The backend does not validate `entity_id` against any database — it trusts whatever the caller sends. For strict multi-tenancy with sensitive data, use separate Pinecone indexes.

---

### 6. Deduplication by Text Content

When the same file is uploaded twice (different `document_id` UUIDs, same content), Pinecone stores duplicate vectors. The retrieval step deduplicates by the **first 100 characters of chunk text** rather than by vector ID, preventing the same content from appearing twice in the context window.

---

### 7. Embedding Retry Logic

The embedding function (`_embed_single`) retries up to 3 times with exponential backoff:

- **HTTP 429** (rate limit) → wait `5 × attempt` seconds, retry
- **Timeout** → wait `3 × attempt` seconds, retry
- **Any other failure** → raise immediately (no silent zero-vector fallback)

The previous implementation silently returned zero vectors on failure, causing Pinecone to reject uploads with *"Dense vectors must contain at least one non-zero value."* This is now fixed — failures raise explicitly so the background task reports `status: failed` with a clear error message.

---

### 8. pdfplumber over PyPDF2

The original PyPDF2 implementation extracted only ~14,000 tokens from a 30-page legal PDF (expected: ~40,000–60,000) due to poor handling of multi-column layouts. `pdfplumber` was substituted for significantly better extraction. For the AGORA dataset (plain `.txt` files), this distinction does not apply, but the system supports both `.txt` and `.pdf` uploads.

---

## Interface Usage Guide

The system provides three interfaces — a **Chat UI**, an **Admin Dashboard**, and a **REST API (Swagger)**.

### Chat UI — `http://localhost:8501`

| Feature | Behavior |
|---|---|
| **Auto-session on load** | A session is created automatically when the app opens — the chat input is immediately available |
| **New Chat** | Creates a new session only if the current session has ≥1 message. Clicking on an empty session is a no-op (prevents empty sessions accumulating) |
| **History sidebar** | Lists all past sessions with ≥1 message, sorted newest first, loaded from Upstash. Click to restore full conversation |
| **Source citations** | Each response shows an expandable "📄 Sources" section with filename and cosine similarity score |
| **Delete session** | 🗑 button removes the session from Upstash and the sidebar |
| **Session persistence** | Messages stored in Upstash under `agora:{session_id}:messages` with 7-day TTL |
| **Namespace** | Controlled via `?entity_id=policy` query param (defaults to `policy`) |

### Admin Dashboard — `http://localhost:8501` → ⚙️ Admin Panel

Three tabs:

| Tab | What It Does |
|---|---|
| **📤 Upload Document** | Upload `.txt` files (max 10 MB). Specify namespace (e.g. `policy`). Runs as background task — returns `task_id`. Re-uploading with the same namespace adds without overwriting. |
| **📊 Database Stats** | Leave namespace blank → global stats (total vectors, index fullness, dimension, all namespaces). Enter a namespace → stats for that namespace only. Shows per-namespace vector counts, index name, and dimension. |
| **🗑️ Clear Database** | Enter namespace + type `YES` to confirm. Deletes all vectors in that namespace only — other namespaces unaffected. |

### FastAPI Swagger — `http://localhost:8000/docs`

Interactive API documentation with request/response schemas for all endpoints. Useful for direct API testing and frontend integration reference.

---

## File Structure & Responsibility Split

```
.
├── app.py                    # FastAPI backend — all REST endpoints
├── chat.py                   # Streamlit chat frontend (Agora UI)
├── main.py                   # Entry point stub
├── pages/
│   └── admin.py              # Streamlit admin panel (upload, stats, clear)
├── src/
│   ├── __init__.py           # Package marker
│   ├── simplified_rag.py     # Core RAG: chunking, embedding, retrieval, orchestration
│   ├── chat_engine.py        # GeminiChatClient: sub-query decomposition + text generation
│   ├── utils.py              # ConversationMemory (Upstash Redis) + env helpers
│   ├── models.py             # Pydantic request/response models
│   └── prompts.yaml          # All prompt templates (system, user, sub-query)
├── tests/
│   └── test_rag_system.py    # 43 unit tests
├── .env                      # API keys (not committed)
├── .github/workflows/
│   └── proddeploy.yml        # CI/CD pipeline
├── requirements.txt          # Python dependencies
├── pyproject.toml             # uv project config
└── runtime.txt               # Python version (3.12)
```

| File | Responsibility |
|---|---|
| `app.py` | HTTP layer only — validates requests, delegates to `SimplifiedRAG`, returns JSON responses. Includes Mangum adapter for AWS Lambda deployment. |
| `src/simplified_rag.py` | PDF/TXT extraction, recursive chunking (500-token target, 20% overlap), Gemini embeddings (1536D), Pinecone upsert/query, RAG orchestration with parallel sub-query retrieval, deduplication, and context assembly. |
| `src/chat_engine.py` | `GeminiChatClient` — sub-query decomposition via Gemini with JSON parsing, regex fallback, and timeout handling. Also handles answer synthesis with configurable temperature and token limits. |
| `src/utils.py` | `ConversationMemory` class (Upstash Redis, 30-min TTL, 5-message window). `.env` loading and direct key reading helpers. |
| `src/models.py` | Pydantic models: `QuestionRequest`, `CreateSessionRequest`, `APIResponse` — the standard request/response envelope. |
| `src/prompts.yaml` | Single source of truth for all prompts — system prompt (AI governance analyst persona), user template (history + context + question), sub-query decomposition template. |
| `chat.py` | Streamlit chat UI with auto-session creation, persistent session history sidebar (Upstash-backed, 7-day TTL), source citation display, and session management (create, restore, delete). |
| `pages/admin.py` | Streamlit admin UI — three tabs for document upload, database stats (with per-namespace breakdown), and namespace clearing. |

---

## System Constants

Defined in `src/simplified_rag.py`:

```python
CHUNK_TARGET_TOKENS  = 500            # Tuned for short AGORA .txt files (~500-1000 tokens each)
CHUNK_OVERLAP_PCT    = 0.20           # 20% overlap = 100 tokens carried from previous chunk tail
CHUNK_OVERLAP_TOKENS = 100            # Derived: int(500 * 0.20)
EMBEDDING_DIMENSION  = 1536           # Gemini Embedding 2 — quality/speed tradeoff for legal text
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
GEMINI_TEXT_MODEL      = "models/gemini-2.5-flash"
GEMINI_API_BASE        = "https://generativelanguage.googleapis.com/v1beta"
```

Defined in `src/utils.py` — `ConversationMemory`:

```python
TTL          = 1800    # 30 minutes — auto-expire inactive sessions
MAX_MESSAGES = 5       # Last 5 Q&A pairs (10 messages total) injected as context
```

---

## API Endpoints

### Document Ingestion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/insert-doc-vector-db` | Upload `.txt` or `.pdf`, chunk, embed, upsert to Pinecone. Runs in background. Returns `task_id`. |
| `POST` | `/replace-document-vectors` | Delete vectors matching filename, re-index new file. Requires `confirm=YES`. |
| `POST` | `/reset-vector-db` | Wipe all vectors in a namespace. Requires `confirm=YES`. |

### Querying

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/create-session` | Generate a new `session_id`. No parameters required. |
| `POST` | `/ask-question` | RAG Q&A. Body: `{entity_id, question, session_id}`. Returns answer + sources. |

### Observability

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/stats` | Global stats: total vectors, index name, dimension, per-namespace vector counts. |
| `GET` | `/entities` | List all namespaces (entity_ids) in the index. |
| `GET` | `/task-status/{task_id}` | Poll background upload task. Returns `running`, `done`, or `failed`. |

### Removed Endpoints

- `/ask-question-stream` — SSE streaming was removed. The ~6s latency is dominated by Gemini API calls, not response transmission, so streaming added complexity without meaningful UX benefit.

> **Full API documentation** with request/response examples: see [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md)

---

## Prompt Engineering

Three templates in `src/prompts.yaml`, loaded at startup via `yaml.safe_load`:

### `system_prompt`

Defines the AI governance analyst persona with strict grounding rules:

- Answer **ONLY** from retrieved context — no external knowledge
- Identify jurisdiction and document type before stating obligations
- Quote exact figures, thresholds, and deadlines
- Distinguish between PROHIBITS / REQUIRES / RECOMMENDS / PERMITS
- 5 verification checkpoints applied before every answer
- If context is insufficient: *"The provided documents do not contain sufficient information to answer this question."*

### `user_template`

Injected per question with three placeholders:

- `{history}` — last 5 exchanges from Upstash Redis (or "No previous conversation.")
- `{context}` — top-ranked unique chunks from Pinecone retrieval
- `{question}` — the user's original question

### `sub_query_template`

Used by `GeminiChatClient.generate_sub_queries()`:

- Returns a JSON array of 1–5 strings
- Includes split/keep-together rules for governance topics
- Handles conversational input → `["__conversational__"]`

---

## Conversation Memory Flow

```
User sends message
        │
        ▼
chat.py saves to Upstash ──► agora:{session_id}:messages  (full UI history, 7-day TTL)
        │
        ▼
chat.py calls POST /ask-question with session_id
        │
        ▼
SimplifiedRAG.ask_questions() calls ConversationMemory.get_history(session_id)
  → reads from ──► session:{session_id}:history  (last 5 pairs, 30-min TTL)
  → formats as plain text: "User: ...\nAgora: ..."
        │
        ▼
History injected into user_template {history} placeholder
        │
        ▼
Gemini generates answer with conversation context
        │
        ▼
ConversationMemory.save(session_id, question, answer)
  → writes to ──► session:{session_id}:history
  → trims to last 5 pairs (LTRIM)
  → resets 30-min TTL (EXPIRE)
```

---

## Installation & Setup

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) package manager
- [Google Gemini API key](https://aistudio.google.com/app/apikey)
- [Pinecone API key](https://app.pinecone.io) + index named `cyber` (dimension: 1536, metric: cosine)
- [Upstash Redis URL](https://upstash.com)

### Install

```bash
uv sync
```

### Environment

Create `.env` in project root:

```env
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=cyber
REDIS_URL=rediss://default:your_token@your_host:6379
```

### Run

**Terminal 1 — Backend:**
```bash
.venv/bin/python -m uvicorn app:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
.venv/bin/streamlit run chat.py
```

| Interface | URL |
|---|---|
| Chat UI | `http://localhost:8501` |
| Admin Panel | `http://localhost:8501` → ⚙️ Admin Panel |
| API Swagger | `http://localhost:8000/docs` |

### Adding More Documents

1. Download additional `.txt` files from the [AGORA dataset](https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data)
2. Open the Admin Panel → 📤 Upload Document tab
3. Set `Index (Namespace)` to `policy` (or any namespace)
4. Upload the `.txt` file
5. Poll `/task-status/{task_id}` until `status: done`
6. New chunks are immediately queryable

---

## Testing

```bash
# Syntax validation
.venv/bin/python -m py_compile src/simplified_rag.py src/chat_engine.py src/utils.py src/models.py app.py chat.py pages/admin.py

# Full test suite (43 tests)
.venv/bin/python -m unittest discover tests -v
```

**Test coverage:**

| Area | What's Tested |
|---|---|
| Imports | All modules import without errors |
| Constants | Chunk size, overlap, dimension, model names, API base |
| Recursive chunking | Token limits, overlap between adjacent chunks, metadata keys, section heading detection, empty/short/whitespace text |
| Prompt loading | System prompt content, user template placeholders, sub-query template structure |
| PDF extraction | Valid PDF returns string, garbage bytes raise Exception |
| Pydantic models | Valid requests, empty question rejection, optional session_id, success/failure responses |
| Sub-query decomposition | Conversational marker, single question, multi-question split, Gemini timeout fallback, malformed JSON fallback |
| Error handling | Unicode text, whitespace-only input, repeated section headers |
| Configuration | `.env.example` exists with required keys, `prompts.yaml` valid, API key requirement enforced |

---

## Example Queries

```
Q: "What reports are required on AI integration within the intelligence community?"
→ Response type: RAG
→ Sources: 10.txt (scores: 0.842, 0.811, 0.807, 0.778)
→ Sub-queries: 1 (single focused question)

Q: "What is the Digital Development Infrastructure Plan?"
→ Response type: RAG
→ Sources: 1.txt (scores: 0.682, 0.664)
→ Sub-queries: 1

Q: "What are the barriers to AI adoption and who enforces compliance?"
→ Response type: RAG
→ Sources: multiple files (cross-document retrieval)
→ Sub-queries: 2 (decomposed into separate retrieval passes)

Q: "What working groups exist for AI governance?"
→ Response type: RAG
→ Sources: multiple files

Q: "Hi, what can you help me with?"
→ Response type: Conversational short-circuit
→ Sources: none (RAG skipped entirely)
→ Sub-queries: ["__conversational__"]
```

---

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| Backend | FastAPI + Uvicorn | Async, type-safe, auto-documented. Mangum adapter for AWS Lambda. |
| Embeddings | Gemini Embedding 2 (1536D) | Best semantic quality for policy/legal language at this dimension |
| Text Generation | Gemini 2.5 Flash | Fast, cost-effective, sufficient quality for grounded Q&A |
| Vector Store | Pinecone (cosine, serverless) | Managed, namespace isolation, no infrastructure to maintain |
| Session Memory (model) | Upstash Redis (30-min TTL) | Serverless, last-5-message window for Gemini context injection |
| Session History (UI) | Upstash Redis (7-day TTL) | Full conversation restoration in chat sidebar |
| Frontend | Streamlit | Rapid development, built-in chat components, no JS framework needed |
| Chunking | Tiktoken `cl100k_base` | Industry-standard token counting, matches OpenAI tokenization |
| PDF Parsing | pdfplumber | Handles complex multi-column layouts better than PyPDF2 |
| TXT Parsing | Python built-in (UTF-8 decode) | Zero-dependency for plain text files |

---

## Known Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | **Scanned PDFs** | pdfplumber cannot extract text from image-based PDFs. Use text-based PDFs or `.txt` files. |
| 2 | **Short documents** | Files under 500 tokens produce a single chunk — retrieval works but there is nothing to differentiate between. |
| 3 | **Gemini rate limits** | Free tier has per-minute limits. Large batch uploads (50+ chunks) may hit 429 errors — retry logic handles this with exponential backoff. |
| 4 | **30-minute session TTL** | The RAG backend's conversation context expires after 30 min of inactivity. The UI history (7-day TTL) persists, but the model loses context from sessions older than 30 min. |
| 5 | **Namespace isolation** | Trust-based, not cryptographic. Do not use for sensitive multi-tenant data without separate indexes. |
| 6 | **Sub-query limit** | Maximum 5 sub-queries per input — design choice to control API cost and latency. |

---

## Next Steps

1. **Hybrid search** — Add BM25 keyword search alongside semantic search for better recall on exact terms
2. **Reranking** — Add a cross-encoder reranker to improve top-k precision
3. **Larger corpus** — Upload all 600+ AGORA documents for full coverage
4. **Latency optimization** — Cache embeddings for repeated queries; skip sub-query decomposition for single questions locally
5. **Metadata filtering** — Use Pinecone metadata filters to scope queries by jurisdiction or document type
6. **Analytics** — Track query latency, source hit rates, and user satisfaction metrics

---

<div align="center">

**Last Updated:** May 2026 · **System Version:** 2.0 · **Dataset:** ETO AGORA (CC0 Public Domain) · **Corpus Tested:** 10 documents, 36 chunks, `policy` namespace

</div>
]]>
