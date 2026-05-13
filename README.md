<div align="center">

# Agora — AI Governance Document Q&A System

**Production-grade Retrieval-Augmented Generation (RAG) for the ETO AGORA Corpus**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000?logo=pinecone)](https://pinecone.io)
[![Gemini](https://img.shields.io/badge/Gemini_2.5-Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-Chat_UI-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Upstash](https://img.shields.io/badge/Upstash-Redis-00E9A3?logo=redis&logoColor=white)](https://upstash.com)
[![Tests](https://img.shields.io/badge/Tests-43_passing-brightgreen)](tests/)

*Ask natural language questions over AI governance documents — grounded answers with source citations, multi-turn memory, and sub-query decomposition.*

</div>

---

## What This Is

Agora is a RAG system built on the [ETO AGORA corpus](https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data) — a collection of AI-relevant laws, regulations, and governance documents from jurisdictions worldwide. It ingests `.txt` and `.pdf` files, chunks and embeds them into a Pinecone vector database, and answers questions using Gemini 2.5 Flash with retrieved context.

The focus is on **answer quality over UI polish**: every response is grounded in retrieved documents, source-attributed, and filtered through a strict governance analyst prompt that distinguishes between what a document *prohibits*, *requires*, *recommends*, or *permits*.

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Key Design Decisions](#key-design-decisions)
- [Testing the System](#testing-the-system)
- [Example Queries to Try](#example-queries-to-try)
- [API Reference](#api-reference)
- [File Structure](#file-structure)
- [System Constants](#system-constants)
- [Prompt Engineering](#prompt-engineering)
- [Conversation Memory](#conversation-memory)
- [Interface Guide](#interface-guide)
- [Known Limitations](#known-limitations)
- [What I Would Do Next](#what-i-would-do-next)
- [Technology Stack](#technology-stack)

---

## Quick Start

### 1. Prerequisites

Before you run anything, make sure you have:

| Requirement | Notes |
|---|---|
| Python 3.12+ | Check with `python --version` |
| [`uv`](https://github.com/astral-sh/uv) | Fast Python package manager — `pip install uv` |
| [Gemini API key](https://aistudio.google.com/app/apikey) | Free tier works for testing |
| [Pinecone API key + index](https://app.pinecone.io) | Create a serverless index named `cyber`, dimension `1536`, metric `cosine` |
| [Upstash Redis URL](https://upstash.com) | Free tier — create a database, copy the `rediss://` connection string |

### 2. Install

```bash
git clone <repo-url>
cd agora
uv sync
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=cyber
REDIS_URL=rediss://default:your_token@your_host:6379
```

> **Tip:** All four variables are required. The system will fail loudly if any are missing — by design.

### 4. Add Documents

Download `.txt` files from the [AGORA dataset on Kaggle](https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data). The system was tested on files `1.txt` through `10.txt` from the `agora/fulltext/` directory, which produce 36 chunks across the `policy` namespace.

You can add documents two ways:

**Via the Admin Panel (recommended):**
1. Open `http://localhost:8501`
2. Click ⚙️ Admin Panel in the sidebar
3. Go to 📤 Upload Document
4. Set namespace to `policy`, upload a `.txt` file
5. Poll the task status until `done`

**Via API:**
```bash
curl -X POST http://localhost:8000/insert-doc-vector-db \
  -F "file=@1.txt" \
  -F "entity_id=policy" \
  -F "document_id=doc-001"
```

### 5. Run

Open two terminals:

```bash
# Terminal 1 — Backend API
.venv/bin/python -m uvicorn app:app --reload --port 8000

# Terminal 2 — Chat Frontend
.venv/bin/streamlit run chat.py
```

| Interface | URL |
|---|---|
| Chat UI | `http://localhost:8501` |
| Admin Panel | `http://localhost:8501` → ⚙️ Admin Panel |
| API Swagger docs | `http://localhost:8000/docs` |

---

## How It Works

Every question goes through five steps:

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
│  └─ Deduplicate by first 100 chars of chunk text        │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3 · Context Assembly                              │
│  ├─ Sort unique chunks by cosine similarity score       │
│  ├─ Build source citations (filename + score)           │
│  └─ Inject last 5 conversation exchanges from Upstash   │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4 · Answer Synthesis (Gemini 2.5 Flash)           │
│  ├─ System prompt: strict AI governance analyst persona │
│  ├─ User template: history + context + question         │
│  └─ Grounded answer with inline source citations        │
└──────────────────────────┬──────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Step 5 · Response + Memory Update                      │
│  ├─ Return answer + sources to caller                   │
│  └─ Save Q&A pair to Upstash Redis (30-min TTL, max 5)  │
└─────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

These are the decisions I'd want to explain in a conversation — and the tradeoffs I made deliberately.

### Sub-Query Decomposition — Accuracy Over Speed

Every question passes through a Gemini call that classifies intent before retrieval:

| Input | Behavior | Example |
|---|---|---|
| Conversational | `["__conversational__"]` — RAG skipped | "hi", "thanks", "what can you do?" |
| Single question | `["original question"]` — one retrieval pass | "What are the reporting requirements?" |
| Multi-part | `["sub-query 1", "sub-query 2", ...]` — parallel retrieval | "What are the barriers to adoption and who enforces compliance?" |

Without decomposition, asking "What are the reporting requirements and who enforces them?" produces embeddings that average across both topics — retrieval returns chunks about *either* reporting *or* enforcement but rarely both with high scores. Decomposition ensures full coverage.

**The latency tradeoff:**

| Mode | Latency | Accuracy on multi-part questions |
|---|---|---|
| Without decomposition | ~4s | Misses one aspect of the question |
| With decomposition | ~6–10s | Both aspects retrieved and addressed |

The extra 2–3 seconds is a deliberate choice. For a governance Q&A system where correctness matters more than speed, I'd always make this call.

---

### Embedding Dimension: 1536D

`models/gemini-embedding-001` supports configurable output dimensions. I chose 1536 because:

- Policy language has subtle distinctions ("PROHIBITS" vs. "RECOMMENDS" vs. "PERMITS") that benefit from higher-dimensional representation.
- Google recommends 1536 as the quality/speed tradeoff for this model.
- Pinecone's free tier supports 1536D without additional cost.

If experiments on a specific corpus show 768D is sufficient, change `EMBEDDING_DIMENSION` in `src/simplified_rag.py` and recreate the index. Storage overhead at this corpus size (36–thousands of chunks) is negligible.

---

### Chunk Size: 500 Tokens (Tuned for Short Documents)

The AGORA `.txt` files are short — typically 500–1,000 tokens each. The default 1,200-token chunk target would collapse each file into a single chunk, making retrieval meaningless.

| Setting | Chunk target | Overlap | Result for 10 AGORA files |
|---|---|---|---|
| Default | 1,200 tokens | 20% (240 tokens) | ~10 chunks — one per file, no retrieval granularity |
| **Tuned** | **500 tokens** | **20% (100 tokens)** | **36 chunks — 2–4 per file, meaningful retrieval** |

The 20% overlap carries the tail of each chunk into the next, reducing the chance that a key sentence gets split at a boundary.

> If you add larger documents (30+ page PDFs), increase `CHUNK_TARGET_TOKENS` back to 1200.

---

### Conversation Memory: Upstash Redis (30-min TTL, 5-message window)

Multi-turn Q&A requires context. Without memory, every follow-up question ("what about the EU version of that?") is answered in isolation.

**Two separate Redis stores — by design:**

| Key pattern | Purpose | TTL | Used by |
|---|---|---|---|
| `session:{id}:history` | Model context (last 5 pairs) | 30 min | RAG backend |
| `agora:{id}:messages` | Full UI history (all messages) | 7 days | Streamlit frontend |

The model always gets a clean 5-message window for context. The UI can restore full conversation history when a user returns after days. These are different jobs and intentionally separate.

**Why a 5-message window:** More than 5 exchanges starts diluting retrieval relevance with stale context. For a governance research tool, users typically ask 3–5 related questions in a session, not 20.

---

### Namespace Isolation (Trust-Based)

A single Pinecone index serves multiple document sets via namespaces. Upload, query, and clear operations are fully isolated by `entity_id`. This is **trust-based, not cryptographic** — the backend doesn't validate `entity_id` against a database. For sensitive multi-tenant deployments, use separate indexes.

---

### Deduplication by Text Content

When the same file is uploaded twice (different UUIDs, same content), Pinecone stores duplicate vectors. The retrieval step deduplicates by the first 100 characters of chunk text, preventing the same content from inflating the context window with repetition.

---

### Explicit Failure on Zero Vectors

The original `_embed_single` implementation silently returned zero vectors on failure. Pinecone rejects these with *"Dense vectors must contain at least one non-zero value."* The current implementation fails explicitly with exponential backoff retry (3 attempts: 5s, 10s, 15s for rate limits; 3s, 6s, 9s for timeouts). Silent failures in an embedding pipeline are worse than loud ones.

---

### pdfplumber over PyPDF2

PyPDF2 extracted ~14,000 tokens from a 30-page legal PDF where 40,000–60,000 was expected — poor handling of multi-column layouts. `pdfplumber` resolved this. For the AGORA `.txt` corpus this doesn't matter, but the system supports both formats for completeness.

---

## Testing the System

### Run the Full Test Suite

```bash
# Validate syntax across all modules
.venv/bin/python -m py_compile \
  src/simplified_rag.py \
  src/chat_engine.py \
  src/utils.py \
  src/models.py \
  app.py \
  chat.py \
  pages/admin.py

# Run all 43 unit tests
.venv/bin/python -m unittest discover tests -v
```

Expected output: `43 tests, 0 failures, 0 errors`

### What the Tests Cover

| Area | What's validated |
|---|---|
| Imports | All modules import cleanly with no dependency errors |
| Constants | Chunk size, overlap, dimension, model names, API base URL |
| Chunking | Token limits respected, overlap between adjacent chunks, metadata keys present, section heading detection, edge cases (empty input, whitespace-only, very short text) |
| Prompts | System prompt content, user template has all three placeholders, sub-query template structure is correct |
| PDF extraction | Valid PDF returns non-empty string; garbage bytes raise `Exception` |
| Pydantic models | Valid requests pass; empty questions are rejected; `session_id` is optional; success/failure response shapes are correct |
| Sub-query decomposition | Conversational marker returned for greetings; single questions pass through; multi-part questions split; Gemini timeout handled gracefully; malformed JSON falls back to original question |
| Error handling | Unicode text processed correctly; whitespace-only input handled; repeated section headers don't break chunking |
| Configuration | `.env.example` exists with all required keys; `prompts.yaml` loads and is valid YAML; missing API key raises an error |

### Manual End-to-End Test

After uploading documents, verify the full pipeline:

```bash
# 1. Create a session
curl -X POST http://localhost:8000/create-session
# → {"session_id": "abc-123"}

# 2. Ask a question
curl -X POST http://localhost:8000/ask-question \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "policy", "question": "What are the reporting requirements for AI in the intelligence community?", "session_id": "abc-123"}'

# 3. Check index stats
curl http://localhost:8000/stats
# → Shows total vectors, namespaces, dimension

# 4. Poll a task (replace task_id with one from an upload response)
curl http://localhost:8000/task-status/{task_id}
# → {"status": "done"} or {"status": "running"}
```

---

## Example Queries to Try

These were tested against the 10-document AGORA corpus (`policy` namespace, 36 chunks). Paste them into the chat UI or hit the `/ask-question` endpoint directly.

### Single-document retrieval

```
What reports are required on AI integration within the intelligence community?
```
*What to expect:* Specific reporting obligations, timelines, and the responsible parties cited from `10.txt` (cosine scores: 0.84, 0.81, 0.81, 0.78).

```
What is the Digital Development Infrastructure Plan?
```
*What to expect:* A grounded summary from `1.txt` with citations. Good test for single-document precision.

```
What enforcement mechanisms exist for AI compliance violations?
```
*What to expect:* The system should distinguish between what's mandated vs. recommended, and cite the relevant authority clearly.

### Cross-document retrieval

```
What working groups or committees exist for AI governance?
```
*What to expect:* Cross-document retrieval pulling from multiple files — tests deduplication and ranking.

```
How do different jurisdictions approach AI transparency requirements?
```
*What to expect:* Multi-document synthesis with citations from several files. Good test for answer grounding across sources.

### Sub-query decomposition (these will split into 2 sub-queries)

```
What are the barriers to AI adoption and who is responsible for enforcing compliance?
```
*What to expect:* Two distinct retrieval passes — one for adoption barriers, one for enforcement — then synthesised into a single coherent answer.

```
What obligations do government agencies have around AI, and what penalties exist for non-compliance?
```
*What to expect:* Similar multi-pass retrieval. Watch the returned `sub_queries` field in the API response to confirm decomposition fired.

### Conversational short-circuit (RAG skipped entirely)

```
Hi, what can you help me with?
```
```
Thanks, that was helpful.
```
*What to expect:* Direct response with no source citations. The `sources` array will be empty. This confirms the conversational detection is working correctly.

### Edge cases worth testing

```
What does this document say about quantum computing?
```
*What to expect:* "The provided documents do not contain sufficient information to answer this question." — tests the grounding guardrail.

```
Summarise everything in the database.
```
*What to expect:* A broad summary with sources — tests retrieval breadth and the system's ability to handle open-ended queries.

---

## API Reference

### Document Ingestion

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/insert-doc-vector-db` | Upload `.txt` or `.pdf`, chunk, embed, and upsert to Pinecone. Runs as a background task. Returns `task_id`. |
| `POST` | `/replace-document-vectors` | Delete all vectors matching a filename, then re-index. Requires `confirm=YES`. |
| `POST` | `/reset-vector-db` | Wipe all vectors in a namespace. Requires `confirm=YES`. |

### Querying

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/create-session` | Generate a new `session_id`. No parameters. |
| `POST` | `/ask-question` | RAG Q&A. Body: `{entity_id, question, session_id}`. Returns answer + sources. |

### Observability

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/stats` | Global index stats: total vectors, dimension, per-namespace counts. |
| `GET` | `/entities` | List all namespaces in the index. |
| `GET` | `/task-status/{task_id}` | Poll a background upload task: `running`, `done`, or `failed`. |

Full schemas and interactive testing: `http://localhost:8000/docs`

---

## File Structure

```
.
├── app.py                    # FastAPI backend — all REST endpoints
├── chat.py                   # Streamlit chat UI
├── main.py                   # Entry point stub
├── pages/
│   └── admin.py              # Admin panel (upload, stats, namespace clear)
├── src/
│   ├── __init__.py
│   ├── simplified_rag.py     # Core RAG: chunking, embedding, retrieval, orchestration
│   ├── chat_engine.py        # GeminiChatClient: sub-query decomposition + generation
│   ├── utils.py              # ConversationMemory (Upstash Redis) + env helpers
│   ├── models.py             # Pydantic request/response models
│   └── prompts.yaml          # All prompt templates — single source of truth
├── tests/
│   └── test_rag_system.py    # 43 unit tests
├── .env                      # API keys (not committed)
├── .env.example              # Template — copy this to .env and fill in your keys
├── .github/workflows/
│   └── proddeploy.yml        # CI/CD pipeline
├── requirements.txt
├── pyproject.toml
└── runtime.txt               # Python 3.12
```

**Responsibility split:**

| File | What it owns |
|---|---|
| `app.py` | HTTP layer only — validates requests, delegates to `SimplifiedRAG`, returns JSON. Mangum adapter included for AWS Lambda. |
| `src/simplified_rag.py` | PDF/TXT extraction, recursive chunking, Gemini embeddings, Pinecone upsert/query, RAG orchestration, deduplication, context assembly. |
| `src/chat_engine.py` | Sub-query decomposition (Gemini call → JSON parsing → regex fallback → timeout handling). Answer synthesis with configurable temperature. |
| `src/utils.py` | `ConversationMemory` class (Upstash Redis, TTL management, window trimming). Env loading helpers. |
| `src/models.py` | Pydantic models: `QuestionRequest`, `CreateSessionRequest`, `APIResponse`. |
| `src/prompts.yaml` | System prompt, user template, and sub-query decomposition template — all in one place, loaded at startup. |
| `chat.py` | Chat UI: auto-session creation, session sidebar, source citation display, session restore and delete. |
| `pages/admin.py` | Three-tab admin: upload, stats, and namespace management. |

---

## System Constants

Defined in `src/simplified_rag.py`:

```python
CHUNK_TARGET_TOKENS    = 500       # Tuned for short AGORA .txt files
CHUNK_OVERLAP_PCT      = 0.20      # 20% overlap = ~100 token tail carry-over
CHUNK_OVERLAP_TOKENS   = 100       # int(500 * 0.20)
EMBEDDING_DIMENSION    = 1536      # Gemini Embedding 2 — quality/speed balance
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
GEMINI_TEXT_MODEL      = "models/gemini-2.5-flash"
GEMINI_API_BASE        = "https://generativelanguage.googleapis.com/v1beta"
```

Defined in `src/utils.py`:

```python
TTL          = 1800   # 30 minutes — auto-expire inactive sessions
MAX_MESSAGES = 5      # Last 5 Q&A pairs injected into model context
```

---

## Prompt Engineering

All templates live in `src/prompts.yaml` and are loaded once at startup via `yaml.safe_load`. They are not hardcoded anywhere in the Python files — changing a prompt does not require a code change or restart.

### `system_prompt` — The Governance Analyst Persona

Defines strict grounding rules:

- Answer **only** from retrieved context. No external knowledge, no hallucination.
- Identify jurisdiction and document type before stating any obligations.
- Quote exact figures, thresholds, and deadlines verbatim.
- Distinguish clearly between PROHIBITS / REQUIRES / RECOMMENDS / PERMITS.
- If context is insufficient: *"The provided documents do not contain sufficient information to answer this question."*

This persona was designed because governance documents make fine-grained normative distinctions that a generic assistant prompt will flatten. "The document discusses AI reporting" is not useful. "Section 4(b) *requires* agencies to submit quarterly reports within 30 days of quarter end" is.

### `user_template` — Per-Question Injection

Three placeholders filled at query time:

- `{history}` — last 5 exchanges from Redis (or "No previous conversation.")
- `{context}` — top-ranked unique chunks from Pinecone, sorted by score
- `{question}` — the user's original question

### `sub_query_template` — Decomposition Instructions

Instructs Gemini to return a JSON array of 1–5 strings. Includes explicit rules for when to split and when to keep together, plus the `__conversational__` sentinel for non-document queries.

---

## Conversation Memory

```
User sends message
        │
        ▼
chat.py → saves to Upstash ──► agora:{session_id}:messages  (full UI history, 7-day TTL)
        │
        ▼
chat.py → POST /ask-question with session_id
        │
        ▼
SimplifiedRAG.ask_questions()
  → ConversationMemory.get_history(session_id)
  → reads ──► session:{session_id}:history  (last 5 pairs, 30-min TTL)
  → formats: "User: ...\nAgora: ..."
        │
        ▼
History injected into {history} placeholder in user_template
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

## Interface Guide

### Chat UI — `http://localhost:8501`

| Feature | Behaviour |
|---|---|
| Auto-session | A session is created automatically when the app opens — no setup needed |
| New Chat | Creates a new session only if the current session has at least one message |
| History sidebar | All past sessions with messages, sorted newest first. Click any to restore. |
| Source citations | Expandable "📄 Sources" section under each answer — filename and cosine score |
| Delete session | 🗑 button removes session from Upstash and the sidebar |
| Namespace | Set via `?entity_id=policy` query param (defaults to `policy`) |

### Admin Panel — `http://localhost:8501` → ⚙️ Admin Panel

| Tab | What it does |
|---|---|
| 📤 Upload Document | Upload `.txt` files (max 10 MB). Returns `task_id`. Re-uploading adds without overwriting. |
| 📊 Database Stats | Leave namespace blank for global stats; enter a namespace for per-namespace counts. |
| 🗑️ Clear Database | Enter namespace + type `YES` to confirm. Deletes only that namespace — others are untouched. |

---

## Known Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | **Scanned PDFs** | `pdfplumber` cannot extract text from image-based PDFs. Use text-based PDFs or `.txt` files. |
| 2 | **Short documents** | Files under 500 tokens produce a single chunk. Retrieval works but there's nothing to differentiate between. |
| 3 | **Gemini rate limits** | Free tier has per-minute limits. Large uploads (50+ chunks) may hit 429 errors — exponential backoff handles retries, but large batches will be slow. |
| 4 | **30-minute model context TTL** | The RAG backend's conversation context expires after 30 minutes of inactivity. The UI history (7-day TTL) persists, but Gemini loses context from sessions older than 30 minutes. |
| 5 | **Trust-based namespace isolation** | `entity_id` is not validated server-side. Do not use this for sensitive multi-tenant data without separate Pinecone indexes. |
| 6 | **5 sub-query maximum** | Hard cap on decomposition — a deliberate choice to control API cost and latency. Extremely complex questions may not be fully decomposed. |
| 7 | **No streaming** | `/ask-question-stream` was removed. The ~6s latency is dominated by Gemini API calls, not response transmission. Streaming added complexity without meaningful UX benefit at this scale. |

---

## What I Would Do Next

Given more time, in rough priority order:

1. **Hybrid search** — BM25 keyword search alongside semantic search. Governance documents contain exact legal terms that semantic search alone can miss ("Section 4(b)(ii)", specific statute numbers). A weighted combination would improve recall significantly.

2. **Cross-encoder reranking** — Replace cosine similarity ranking with a cross-encoder (e.g. Cohere Rerank or a local `cross-encoder/ms-marco-MiniLM`) to improve top-k precision. The initial retrieval uses bi-encoder embeddings for speed; reranking on the top 20 results would sharpen the final 4.

3. **Metadata filtering** — Pinecone supports metadata filters. Adding `jurisdiction`, `document_type`, and `year` as chunk metadata would let users scope queries ("only EU regulations", "only documents from 2023+") without changing the retrieval architecture.

4. **Larger corpus** — The system was tested on 10 of the 600+ AGORA documents. Uploading the full corpus is straightforward (Admin Panel batch upload or a script calling `/insert-doc-vector-db` in a loop) and would produce a much richer retrieval surface.

5. **Query-level caching** — Cache embeddings for repeated queries. Pinecone queries with identical vectors are common in production (users often ask the same questions). A Redis cache keyed on the embedding vector would cut latency for repeat queries to near-zero.

6. **Evaluation harness** — Build a small ground-truth QA set (20–30 questions with known answers) and score the system on retrieval recall@k and answer accuracy. Right now, quality is assessed qualitatively. A quantitative eval loop would make it easier to validate whether changes (chunk size, decomposition threshold, top-k) actually help.

---

## Technology Stack

| Component | Technology | Why |
|---|---|---|
| Backend | FastAPI + Uvicorn | Async, type-safe, auto-documented. Mangum adapter for AWS Lambda if needed. |
| Embeddings | Gemini Embedding 2 (1536D) | Best semantic quality for policy/legal language at this dimension and cost point. |
| Text generation | Gemini 2.5 Flash | Fast and cost-effective. Sufficient quality for grounded Q&A where the heavy lifting is done by retrieval. |
| Vector store | Pinecone (cosine, serverless) | Managed, namespace isolation, no infrastructure. Free tier handles this corpus size comfortably. |
| Session memory (model) | Upstash Redis (30-min TTL) | Serverless. Sub-millisecond latency for the 5-message context window injection. |
| Session history (UI) | Upstash Redis (7-day TTL) | Same infrastructure, different job — full conversation restoration in sidebar. |
| Frontend | Streamlit | Rapid development, built-in chat components. The assessment prioritises retrieval quality over UI; Streamlit gets a working interface built quickly. |
| Token counting | Tiktoken `cl100k_base` | Industry-standard, consistent with how chunk sizes are actually measured. |
| PDF parsing | pdfplumber | Handles multi-column layouts. PyPDF2 was dropping ~65% of content on complex legal PDFs. |

---

## Assumptions Made

- The primary use case is researchers and policy analysts asking focused questions, not general-purpose chatbot usage. This justifies the strict grounding prompt and the accuracy-over-latency tradeoffs.
- Session TTLs (30 min for model context, 7 days for UI history) are appropriate for a research tool. These would be configurable parameters in a production deployment.
- The AGORA dataset files are short (500–1,000 tokens each). The 500-token chunk size was tuned for this. If you add longer documents, adjust `CHUNK_TARGET_TOKENS` accordingly.

---

<div align="center">

**Last updated:** May 2026 · **Version:** 2.0 · **Dataset:** ETO AGORA (CC0 Public Domain)

*Tested on 10 documents · 36 chunks · `policy` namespace*

</div>