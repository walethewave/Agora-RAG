# Agora — AI Governance Document Q&A System

A production-grade **Retrieval-Augmented Generation (RAG)** system for answering questions over the **ETO AGORA AI Governance Documents corpus** using **Google Gemini API** and **Pinecone** vector database.

The system is named **Agora** after the ETO AGORA dataset — a living collection of AI-relevant laws, regulations, standards, and governance documents from the United States and around the world.

---

## What This System Does

Agora lets you upload any collection of `.txt` governance documents, index them into Pinecone, and ask natural language questions over them. It returns grounded answers with source citations, maintains conversation history across sessions, and handles multi-part questions by decomposing them into sub-queries before retrieval.

---

## Assessment Alignment

This repository addresses the ML Engineer Assessment expectations:

- A runnable system that ingests documents and answers questions over them
- Grounded answers with source citations (filename + relevance score)
- A clear retrieval strategy with explicit tradeoff reasoning
- Conversation memory with persistent session history
- Admin panel for document management and database inspection
- Example queries with expected responses

---

## Dataset

**Source**: [ETO AGORA AI Governance Documents Data](https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data)

The corpus is a living collection of AI-relevant laws, regulations, standards, and other governance documents. This system was built and tested using 10 `.txt` files from the `agora/fulltext/` directory (files `1.txt` through `10.txt`), producing **36 chunks** across the `policy` namespace.

To extend the corpus, download additional `.txt` files from the dataset and upload them via the Admin Panel using the same `entity_id` (namespace). They will be added to the existing index without overwriting anything.

---

## Architecture Overview

```
User Question
      ↓
[Step 1] Sub-Query Decomposition (Gemini API)
      ├─ Conversational input ("hi", "thanks") → skip RAG, answer directly
      ├─ Multi-part question → split into 1–5 sub-queries
      └─ Single question → pass through as-is
      ↓
[Step 2] Parallel Embedding + Pinecone Retrieval
      ├─ Each sub-query → Gemini Embedding (1536D)
      ├─ Query Pinecone (top-4 chunks per sub-query, cosine similarity)
      └─ Deduplicate by text content, rank by score
      ↓
[Step 3] Context Assembly
      ├─ Merge unique chunks sorted by relevance score
      ├─ Build source citations (filename, score)
      └─ Inject last 5 conversation exchanges from Upstash Redis
      ↓
[Step 4] Answer Synthesis (Gemini 2.5 Flash)
      ├─ System prompt (AI governance analyst persona + rules)
      ├─ User template (history + context + question)
      └─ Generate grounded answer with citations
      ↓
[Step 5] Response + Memory Update
      ├─ Return answer + sources to caller
      └─ Save Q&A pair to Upstash Redis (TTL: 30 min, max 5 pairs)
```

---

## Key Design Decisions

### 1. Chunk Size: 500 Tokens (not 1200)

The AGORA `.txt` files are short governance documents — typically 500–1000 tokens each. Using a 1200-token chunk target would collapse each file into a single chunk, making retrieval meaningless (every query returns the same chunk regardless of relevance).

Reducing to **500 tokens** with **20% overlap (100 tokens)** produces 2–4 meaningful chunks per file, giving the retriever something to differentiate between. With 10 files this produced **36 chunks** — a reasonable retrieval pool.

**Tradeoff**: Smaller chunks mean less context per chunk. If a document is very long, important context may be split across chunks. The 20% overlap mitigates this by carrying the tail of the previous chunk into the next one.

If you add larger documents (e.g., full legislation PDFs), consider increasing `CHUNK_TARGET_TOKENS` back to 1200.

### 2. Embedding Dimension: 1536

`models/gemini-embedding-001` supports configurable output dimensions. **1536 was chosen** because:
- It is the recommended quality/speed tradeoff for Gemini Embedding 2
- It captures complex semantic relationships in legal/policy language better than lower dimensions (e.g., 768)
- Pinecone's free tier supports up to 1536 dimensions without additional cost
- If you want to run more experiments with lower cost, you can reduce to 768 by changing `EMBEDDING_DIMENSION` in `src/simplified_rag.py` and recreating the Pinecone index

**Note**: The embedding model was corrected from `models/embedding-001` (which returned 404) to `models/gemini-embedding-001` — the correct v1beta endpoint.

### 3. Sub-Query Decomposition

Every question passes through a Gemini API call that classifies it and optionally splits it:

- **Conversational** ("hi", "thanks") → returns `["__conversational__"]` → RAG is skipped entirely, Gemini answers directly
- **Single question** → returns `["original question"]` → one retrieval pass
- **Multi-part question** → returns `["sub-query 1", "sub-query 2", ...]` → parallel retrieval passes, results merged

**Why this matters**: If a user asks "What are the reporting requirements and who enforces them?", a single embedding query would retrieve chunks about either reporting OR enforcement, not both. Decomposition ensures both aspects are retrieved and answered.

**Latency impact**: Sub-query decomposition adds ~2–3 seconds (one Gemini API call). Total end-to-end latency is approximately **6–10 seconds**. Without sub-query decomposition it would be ~4 seconds. The accuracy improvement justifies the cost — this is a deliberate engineering tradeoff.

### 4. Conversation Memory (Upstash Redis)

**Why**: Multi-turn Q&A requires context. Without memory, every question is answered in isolation and the model cannot reference prior exchanges.

**Implementation** (`src/utils.py` — `ConversationMemory`):
- **Provider**: Upstash Redis (serverless, no infrastructure to manage)
- **Storage key**: `session:{session_id}:history` (Redis list)
- **Window**: Last 5 user-assistant pairs (10 messages total)
- **TTL**: 30 minutes per session (auto-expires inactive sessions)
- **Injection**: History is formatted as plain text and injected into the `{history}` placeholder in `user_template`

**Two separate Redis stores**:
1. `session:{session_id}:history` — short-term model context (last 5 exchanges, 30-min TTL) used by the RAG backend
2. `agora:{session_id}:messages` — full chat UI history (all messages, 7-day TTL) used by the Streamlit frontend for session restoration

This separation means the model always gets a clean 5-message window for context, while the UI can restore the full conversation history when a user returns to an old session.

### 5. Multi-Tenant Namespacing

A single Pinecone index (`cyber`) serves multiple document sets via **namespaces**. Each `entity_id` maps to a Pinecone namespace:

- `entity_id=policy` → all AGORA governance documents
- `entity_id=legal` → could hold a different document set
- Upload, query, and clear operations are fully namespace-isolated

**Tradeoff**: Namespace isolation is trust-based, not cryptographic. For strict multi-tenancy with sensitive data, use separate Pinecone indexes.

### 6. Deduplication by Text Content

When the same file is uploaded twice (different `document_id` UUIDs, same content), Pinecone stores duplicate vectors. The retrieval step deduplicates by the first 100 characters of chunk text rather than by vector ID, preventing the same content from appearing twice in the context window.

### 7. Embedding Retry Logic

The embedding function (`_embed_single`) retries up to 3 times with exponential backoff:
- HTTP 429 (rate limit) → wait `5 * attempt` seconds, retry
- Timeout → wait `3 * attempt` seconds, retry
- Any other failure → raise immediately (no silent zero-vector fallback)

The previous implementation silently returned zero vectors on failure, which caused Pinecone to reject uploads with "Dense vectors must contain at least one non-zero value." This is now fixed — failures raise explicitly so the background task reports `status: failed` with a clear error message.

### 8. pdfplumber over PyPDF2

The original PyPDF2 implementation only extracted ~14,000 tokens from a 30-page legal PDF (expected: ~40,000–60,000). This was because PyPDF2 cannot handle complex multi-column layouts and scanned pages.

`pdfplumber` was substituted — it handles complex PDF layouts significantly better. For the AGORA dataset (plain `.txt` files), this distinction does not apply, but the system supports both `.txt` and `.pdf` uploads.

---

## File Structure

```
.
├── app.py                    # FastAPI backend — all REST endpoints
├── chat.py                   # Streamlit chat frontend (Agora UI)
├── pages/
│   └── admin.py              # Streamlit admin panel
├── src/
│   ├── simplified_rag.py     # Core RAG: chunking, embedding, retrieval, orchestration
│   ├── chat_engine.py        # GeminiChatClient: sub-query decomposition + text generation
│   ├── utils.py              # ConversationMemory (Redis) + env helpers
│   ├── models.py             # Pydantic request/response models
│   └── prompts.yaml          # All prompt templates (system, user, sub-query)
├── tests/
│   └── test_rag_system.py    # 43 unit tests
├── .env                      # API keys (not committed)
├── .env.example              # Template for required keys
├── requirements.txt          # Python dependencies
└── pyproject.toml            # uv project config
```

### Responsibility Split

| File | Responsibility |
|------|---------------|
| `app.py` | HTTP layer only — validates requests, delegates to `SimplifiedRAG`, returns responses |
| `src/simplified_rag.py` | PDF/TXT extraction, recursive chunking, Gemini embeddings, Pinecone upload/query, RAG orchestration |
| `src/chat_engine.py` | `GeminiChatClient` — sub-query decomposition and answer synthesis via Gemini REST API |
| `src/utils.py` | `ConversationMemory` (Upstash Redis), `.env` loading helpers |
| `src/prompts.yaml` | Single source of truth for all prompts — system prompt, user template, sub-query template |
| `chat.py` | Streamlit chat UI with persistent session history sidebar backed by Upstash |
| `pages/admin.py` | Streamlit admin UI — upload documents, view stats, clear namespaces |

---

## Constants (src/simplified_rag.py)

```python
CHUNK_TARGET_TOKENS = 500           # Tuned for short AGORA .txt files (~500-1000 tokens each)
CHUNK_OVERLAP_PCT   = 0.20          # 20% overlap = 100 tokens carried from tail of previous chunk
CHUNK_OVERLAP_TOKENS = 100          # Derived: int(500 * 0.20)
EMBEDDING_DIMENSION = 1536          # Gemini Embedding 2 — quality/speed tradeoff
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
GEMINI_TEXT_MODEL      = "models/gemini-2.5-flash"
GEMINI_API_BASE        = "https://generativelanguage.googleapis.com/v1beta"
```

---

## API Endpoints

### Document Ingestion
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/insert-doc-vector-db` | Upload `.txt` or `.pdf`, chunk, embed, upsert to Pinecone. Runs in background. Returns `task_id`. |
| `POST` | `/replace-document-vectors` | Delete vectors matching filename, re-index new file. Requires `confirm=YES`. |
| `POST` | `/reset-vector-db` | Wipe all vectors in a namespace. Requires `confirm=YES`. |

### Querying
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/create-session` | Generate a new `session_id`. No parameters required. |
| `POST` | `/ask-question` | RAG Q&A. Body: `{entity_id, question, session_id}`. Returns answer + sources. |

### Observability
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/stats` | Global stats: total vectors, index name, dimension, per-namespace vector counts. |
| `GET` | `/entities` | List all namespaces (entity_ids) in the index. |
| `GET` | `/task-status/{task_id}` | Poll background upload task. Returns `running`, `done`, or `failed`. |

### Removed Endpoints
- `/ask-question-stream` — SSE streaming endpoint was removed. The system uses synchronous `/ask-question` only. Streaming added complexity without meaningful UX benefit given the ~6-second latency is dominated by Gemini API calls, not response transmission.

---

## Prompt Engineering (src/prompts.yaml)

Three templates, all loaded at startup via `yaml.safe_load`:

### system_prompt
Defines the AI governance analyst persona. Key rules:
- Answer ONLY from retrieved context — no external knowledge
- Identify jurisdiction and document type before stating any obligation
- Quote exact figures, thresholds, and deadlines
- Distinguish between PROHIBITS / REQUIRES / RECOMMENDS / PERMITS
- If context is insufficient, say exactly: "The provided documents do not contain sufficient information to answer this question."
- If context contains a partial answer, answer from it and state what is missing

### user_template
Injected per question. Placeholders:
- `{history}` — last 5 exchanges from Upstash Redis (or "No previous conversation.")
- `{context}` — top-ranked unique chunks from Pinecone retrieval
- `{question}` — the user's original question

### sub_query_template
Used by `GeminiChatClient.generate_sub_queries()`. Placeholder:
- `{user_query}` — the raw user input

Returns a JSON array. Examples:
- `"hi"` → `["__conversational__"]`
- `"What are the reporting requirements?"` → `["What are the reporting requirements?"]`
- `"What are the barriers to AI adoption and who enforces compliance?"` → `["What are the barriers to AI adoption?", "Who enforces AI compliance?"]`

---

## Admin Panel (pages/admin.py)

Three tabs accessible at `http://localhost:8501` → Admin Panel button:

### Upload Document
- Accepts `.txt` files (max 10 MB, enforced by FastAPI)
- Field: `Index (Namespace)` — maps to Pinecone namespace (e.g. `policy`)
- Upload runs as a background task — returns `task_id` immediately
- Poll `/task-status/{task_id}` to check `running` / `done` / `failed`
- Re-uploading with the same `entity_id` adds to the namespace without overwriting

### Database Stats
- Leave namespace blank → global stats (total vectors, index fullness, all namespaces)
- Enter a namespace → stats for that namespace only (vector count)
- Shows `index_name`, `dimension`, per-namespace breakdown

### Clear Database
- Enter namespace + type `YES` to confirm
- Calls `/reset-vector-db` — deletes all vectors in that namespace only
- Other namespaces are unaffected

---

## Chat Frontend (chat.py)

Built with Streamlit. Key behaviors:

- **Auto-session on load**: A session is created automatically when the app opens — the chat input is immediately available without clicking anything
- **New Chat**: Creates a new session only if the current session has at least one message. Clicking New Chat on an empty session is a no-op (prevents empty sessions accumulating in history)
- **History sidebar**: Lists all past sessions that have at least one message, sorted newest first, loaded from Upstash. Clicking a session restores the full conversation
- **Session persistence**: Each session's messages are stored in Upstash under `agora:{session_id}:messages` with a 7-day TTL
- **Source citations**: Each assistant response shows an expandable "📄 Sources" section with filename and cosine similarity score
- **Delete session**: 🗑 button next to each history item removes it from Upstash and the sidebar

---

## Conversation Memory Flow

```
User sends message
        ↓
chat.py saves to Upstash: agora:{session_id}:messages  (full UI history, 7-day TTL)
        ↓
chat.py calls POST /ask-question with session_id
        ↓
SimplifiedRAG.ask_questions() calls ConversationMemory.get_history(session_id)
  → reads from: session:{session_id}:history  (last 5 pairs, 30-min TTL)
  → formats as plain text: "User: ...\nAgora: ..."
        ↓
History injected into user_template {history} placeholder
        ↓
Gemini generates answer with conversation context
        ↓
ConversationMemory.save(session_id, question, answer)
  → writes to: session:{session_id}:history
  → trims to last 5 pairs
  → resets 30-min TTL
```

---

## Installation & Setup

### Prerequisites
- Python 3.12+
- `uv` package manager
- Google Gemini API key — [get one here](https://aistudio.google.com/app/apikey)
- Pinecone API key + index named `cyber` (dimension: 1536, metric: cosine) — [app.pinecone.io](https://app.pinecone.io)
- Upstash Redis URL — [upstash.com](https://upstash.com)

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

Open **http://localhost:8501** for the chat UI.
Open **http://localhost:8000/docs** for the FastAPI Swagger UI.

---

## Adding More Documents

1. Download additional `.txt` files from the [AGORA dataset](https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data)
2. Open the Admin Panel → Upload Document tab
3. Set `Index (Namespace)` to `policy` (or any namespace you want)
4. Upload the `.txt` file
5. Poll `/task-status/{task_id}` until `status: done`
6. The new chunks are immediately queryable

---

## Testing

```bash
# Syntax validation
.venv/bin/python -m py_compile src/simplified_rag.py src/chat_engine.py src/utils.py src/models.py app.py chat.py pages/admin.py

# Full test suite (43 tests)
.venv/bin/python -m unittest discover tests -v
```

Test coverage:
- Imports and initialization
- Recursive chunking (token limits, overlap, metadata, empty/short text)
- Prompt loading and placeholder validation
- PDF/TXT extraction
- Pydantic model validation
- Sub-query decomposition (conversational, single, multi-part, timeout, bad JSON)
- Error handling (unicode, whitespace, repeated headers)
- Configuration (.env.example, YAML validity, API key requirement)

---

## Known Limitations

1. **Scanned PDFs**: pdfplumber cannot extract text from image-based PDFs. Use text-based PDFs or `.txt` files
2. **Short documents**: Files under 500 tokens produce a single chunk — retrieval works but there is nothing to differentiate between
3. **Gemini rate limits**: Free tier has per-minute limits. Large batch uploads (50+ chunks) may hit 429 errors — the retry logic handles this with backoff
4. **30-minute session TTL**: The RAG backend's conversation context expires after 30 minutes of inactivity. The UI history (7-day TTL) persists, but the model will not have context from sessions older than 30 minutes
5. **Namespace isolation**: Trust-based, not cryptographic. Do not use for sensitive multi-tenant data without separate indexes
6. **Sub-query limit**: Maximum 5 sub-queries per input — design choice to control API cost and latency

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Backend | FastAPI + Uvicorn | Async, type-safe, auto-documented |
| Embeddings | Gemini Embedding 2 (1536D) | Best semantic quality for policy/legal language |
| Text Generation | Gemini 2.5 Flash | Fast, cost-effective, sufficient for Q&A |
| Vector Store | Pinecone (cosine, serverless) | Managed, namespace isolation, no infra |
| Session Memory (model) | Upstash Redis (30-min TTL) | Serverless, last-5-message window for Gemini context |
| Session History (UI) | Upstash Redis (7-day TTL) | Full conversation restoration in sidebar |
| Frontend | Streamlit | Rapid development, no JS framework needed |
| Chunking | Tiktoken cl100k_base | Industry-standard token counting |
| PDF Parsing | pdfplumber | Handles complex layouts better than PyPDF2 |
| TXT Parsing | Python built-in (UTF-8 decode) | No dependency needed |

---

## Example Queries

```
Q: "What reports are required on AI integration within the intelligence community?"
→ Sources: 10.txt (scores: 0.842, 0.811, 0.807, 0.778)

Q: "What is the Digital Development Infrastructure Plan?"
→ Sources: 1.txt (scores: 0.682, 0.664)

Q: "What working groups exist for AI governance?"
→ Sources: multiple files (cross-document retrieval)

Q: "Hi, what can you help me with?"
→ Conversational short-circuit — no RAG, direct Gemini response
```

---

## Next Steps

1. **Hybrid search**: Add BM25 keyword search alongside semantic search for better recall on exact terms
2. **Reranking**: Add a cross-encoder reranker to improve top-k precision
3. **Larger corpus**: Upload all 600+ AGORA documents for full coverage
4. **Latency optimization**: Cache embeddings for repeated queries; skip sub-query decomposition for single questions locally
5. **Metadata filtering**: Use Pinecone metadata filters to scope queries by jurisdiction or document type
6. **Analytics**: Track query latency, source hit rates, user satisfaction

---

**Last Updated**: May 2026
**System Version**: 2.0 (Agora — AI Governance Q&A)
**Dataset**: ETO AGORA (CC0 Public Domain)
**Corpus tested**: 10 documents, 36 chunks, `policy` namespace
# Agora-Rag-
# Agora-RAG
