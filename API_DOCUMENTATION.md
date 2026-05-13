# Agora RAG — API Documentation

> REST API reference for the Agora AI Governance Document Q&A System.

---

## Base Information

| Property | Value |
|---|---|
| **Base URL (Local Dev)** | `http://localhost:8000` |
| **Swagger UI** | `http://localhost:8000/docs` |
| **Content Types** | `application/json` or `multipart/form-data` (per endpoint) |
| **API Version** | `3.0.0` |

> **Note:** The Streamlit frontend (`chat.py`) calls `SimplifiedRAG` directly in-process — it does **not** require the FastAPI backend to be running. The REST API is for external integrations, testing via Swagger, or AWS Lambda deployment (via Mangum adapter).

---

## How `entity_id` Works

`entity_id` is used as a **Pinecone namespace** to isolate document sets:

- `entity_id=policy` → AGORA governance documents
- `entity_id=legal` → a separate document collection
- Upload, query, and clear operations are fully namespace-isolated
- The backend does **not** validate `entity_id` — it trusts whatever the caller sends

---

## Standard Response Format

All endpoints return a consistent JSON envelope:

```json
{
  "responseCode": "00",
  "responseMessage": "Human readable message",
  "data": {}
}
```

| Field | Type | Description |
|---|---|---|
| `responseCode` | `string` | `"00"` = success; `"01"` = error |
| `responseMessage` | `string` | Human-readable description of the result |
| `data` | `object` | Response payload (may be `null` on errors) |

---

## Endpoints

### Health Check

**`GET /`**

```json
{
  "responseCode": "00",
  "responseMessage": "Simplified RAG API is running successfully"
}
```

---

### Create a Chat Session

**`POST /create-session`**

Generates a new `session_id` for conversation memory. No request body required.

**Response**

```json
{
  "responseCode": "00",
  "responseMessage": "Session created successfully",
  "data": {
    "session_id": "123e4567-e89b-12d3-a456-426614174000"
  }
}
```

---

### Ask a Question

**`POST /ask-question`**

RAG Q&A with sub-query decomposition, scoped to a namespace.

**Headers:** `Content-Type: application/json`

**Request Body**

```json
{
  "entity_id": "policy",
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "question": "What are the reporting requirements for AI integration?"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `entity_id` | `string` | **Yes** | Namespace to search |
| `question` | `string` | **Yes** | The user's question |
| `session_id` | `string` | No | Session ID for conversation memory (from `/create-session`) |

**Response**

```json
{
  "responseCode": "00",
  "responseMessage": "Question answered successfully",
  "data": {
    "answer": "According to the documents, the following reports are required...",
    "sources": [
      {
        "filename": "10.txt",
        "category": "",
        "section": "",
        "relevance_score": 0.842
      }
    ]
  }
}
```

---

### Upload a Document

**`POST /insert-doc-vector-db`**

Upload a `.txt` or `.pdf` file, chunk it, embed with Gemini, and upsert to Pinecone. Runs as a background task.

**Headers:** `Content-Type: multipart/form-data`

**Form Data**

| Field | Type | Required | Description |
|---|---|---|---|
| `entity_id` | `string` | **Yes** | Namespace to upload into |
| `file` | `file` | **Yes** | `.txt` or `.pdf` file (max 10 MB) |

**Response**

```json
{
  "responseCode": "00",
  "responseMessage": "Background task started to process '5.txt'.",
  "data": {
    "entity_id": "policy",
    "task_id": "task-uuid-here",
    "file_size_mb": 0.02
  }
}
```

Poll `/task-status/{task_id}` to track progress.

---

### Check Task Status

**`GET /task-status/{task_id}`**

Poll the status of a background upload/replace task.

**Response**

```json
{
  "responseCode": "00",
  "responseMessage": "Task status retrieved",
  "data": {
    "task_id": "task-uuid-here",
    "status": "done",
    "message": "Successfully added 5.txt"
  }
}
```

| `status` Value | Meaning |
|---|---|
| `running` | Task is still in progress |
| `done` | Task completed successfully |
| `failed` | Task encountered an error |

---

### Replace a Document

**`POST /replace-document-vectors`**

Deletes existing vectors matching the filename in a namespace, then re-indexes the new file.

**Headers:** `Content-Type: multipart/form-data`

**Form Data**

| Field | Type | Required | Description |
|---|---|---|---|
| `entity_id` | `string` | **Yes** | Namespace the document belongs to |
| `confirm` | `string` | **Yes** | Must be exactly `"YES"` |
| `file` | `file` | **Yes** | New `.txt` or `.pdf` file (max 10 MB) |

**Response**

```json
{
  "responseCode": "00",
  "responseMessage": "Document vector replacement started",
  "data": {
    "entity_id": "policy",
    "task_id": "task-uuid-here"
  }
}
```

---

### Wipe a Namespace

**`POST /reset-vector-db`**

Permanently deletes **all vectors** in the specified namespace. Does not affect other namespaces.

**Headers:** `Content-Type: multipart/form-data`

**Form Data**

| Field | Type | Required | Description |
|---|---|---|---|
| `entity_id` | `string` | **Yes** | Namespace to wipe |
| `confirm` | `string` | **Yes** | Must be exactly `"YES"` |

**Response**

```json
{
  "responseCode": "00",
  "responseMessage": "Namespace 'policy' reset successfully",
  "data": {}
}
```

---

### Get Database Stats

**`GET /stats`**

Returns global index statistics with per-namespace vector counts.

**No parameters required.**

**Response**

```json
{
  "responseCode": "00",
  "responseMessage": "Database statistics fetched successfully",
  "data": {
    "total_vectors": 36,
    "index_name": "cyber",
    "dimension": 1536,
    "entity_ids": {
      "policy": {
        "vector_count": 36
      }
    }
  }
}
```

---

### List All Namespaces

**`GET /entities`**

Lists all active namespaces (entity_ids) in the Pinecone index.

**No parameters required.**

**Response**

```json
{
  "responseCode": "00",
  "responseMessage": "Found 1 entities",
  "data": {
    "entities": ["policy"],
    "count": 1
  }
}
```

---

## Quick Start Workflow

### Chat Integration

1. Call `POST /create-session` → store the returned `session_id`
2. Call `POST /ask-question` with `entity_id`, `session_id`, and the user's question
3. Display `data.answer` and optionally show `data.sources`

### Document Upload

1. Call `POST /insert-doc-vector-db` with `entity_id` and the file
2. Poll `GET /task-status/{task_id}` every 3 seconds until `status` is `done` or `failed`
3. New chunks are immediately queryable via `/ask-question`

---

## Error Responses

All errors use `responseCode: "01"`:

```json
{
  "responseCode": "01",
  "responseMessage": "RAG system not initialized"
}
```

Common errors:

| Scenario | `responseMessage` |
|---|---|
| System not ready | `"RAG system not initialized"` |
| Invalid file type | `"Only PDF and TXT files are supported"` |
| File too large | `"File too large (12.50 MB). Max allowed is 10 MB."` |
| Missing confirmation | `"Must confirm with 'YES' to reset the vector database"` |
| Task not found | `"Task ID not found"` |
