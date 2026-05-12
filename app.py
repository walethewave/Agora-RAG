"""
FastAPI Backend for Simplified RAG System (Multi-Tenant)
=========================================================
Endpoints:
1. POST /insert-doc-vector-db    - Upload TXT/PDF & add to tenant namespace
2. POST /replace-document-vectors - Replace vectors for a specific document
3. POST /reset-vector-db          - Wipe vectors in a specific namespace
4. POST /create-session           - Generate a new session ID
5. POST /ask-question             - Ask questions with RAG (pass session_id for memory)
6. GET  /stats                    - Database statistics (per-tenant or global)
7. GET  /entities                 - List all known namespaces
8. GET  /task-status/{task_id}    - Poll background upload task
"""

# --- Standard Library Imports ---
import os
import uuid
import logging
from typing import Optional

# --- Third-Party Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from mangum import Mangum

# --- Local Application Imports ---
from src.simplified_rag import SimplifiedRAG
from src.models import (
    QuestionRequest, CreateSessionRequest,
    APIResponse,
)

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Simplified RAG API (Multi-Tenant)",
    description="Multi-tenant RAG API — each tenant is isolated via Pinecone namespaces",
    version="3.0.0"
)

# Mangum adapter for AWS Lambda
handler = Mangum(app)

# --- Global State ---
rag_system: Optional[SimplifiedRAG] = None
MAX_FILE_SIZE_MB = 10
tasks = {}


def generate_task_id() -> str:
    return str(uuid.uuid4())


@app.on_event("startup")
async def startup_event():
    """Initialize the SimplifiedRAG system on application startup."""
    global rag_system
    try:
        rag_system = SimplifiedRAG()
        logger.info("RAG system initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        rag_system = None


@app.get("/", response_model=APIResponse)
async def root():
    """API Health Check Endpoint."""
    return {
        "responseCode": "00",
        "responseMessage": "Simplified RAG API is running successfully",
    }


@app.post("/insert-doc-vector-db", response_model=APIResponse)
async def insert_doc_vector_db(
    background_tasks: BackgroundTasks,
    entity_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload a TXT or PDF file and index it into Pinecone.

    - `entity_id` — namespace label for this document set. Re-use the same entity_id to add more documents to the same namespace.
    - `file` — TXT or PDF file to upload (max 10 MB).
    - Chunks recursively (1200 tokens, 20% overlap), embeds with Gemini, and upserts to Pinecone.
    - Runs in the background — poll /task-status/{task_id} to check progress.
    """
    try:
        if not rag_system:
            logger.error("POST /insert-doc-vector-db failed: RAG system not initialized.")
            return {
                "responseCode": "01",
                "responseMessage": "RAG system not initialized"
            }

        # Validate file type
        if not file.filename or not file.filename.lower().endswith(('.pdf', '.txt')):
            return {
                "responseCode": "01",
                "responseMessage": "Only PDF and TXT files are supported"
            }

        # Read file bytes and validate size
        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return {
                "responseCode": "01",
                "responseMessage": f"File too large ({file_size_mb:.2f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB."
            }

        task_id = generate_task_id()
        tasks[task_id] = {"status": "running", "message": f"Processing {file.filename}"}

        def background_update():
            try:
                logger.info(f"Starting background add for '{file.filename}' in namespace: {entity_id}")
                result = rag_system.add_to_existing_collection(
                    file_bytes=file_bytes,
                    filename=file.filename,
                    namespace=entity_id,
                )
                if result.get('success'):
                    tasks[task_id]["status"] = "done"
                    tasks[task_id]["message"] = f"Successfully added {file.filename}"
                    tasks[task_id]["result"] = result
                else:
                    tasks[task_id]["status"] = "failed"
                    tasks[task_id]["message"] = result.get('error', 'Unknown error')
            except Exception as e:
                logger.error(f"Background add failed for '{file.filename}': {e}")
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["message"] = f"Failed: {str(e)}"

        background_tasks.add_task(background_update)

        return {
            "responseCode": "00",
            "responseMessage": f"Background task started to process '{file.filename}'.",
            "data": {
                "entity_id": entity_id,
                "task_id": task_id,
                "file_size_mb": round(file_size_mb, 2),
            }
        }

    except Exception as e:
        logger.error(f"Unexpected error in /insert-doc-vector-db: {e}")
        return {
            "responseCode": "01",
            "responseMessage": f"Failed: {str(e)}"
        }


@app.post("/replace-document-vectors", response_model=APIResponse)
async def replace_document_vectors_endpoint(
    background_tasks: BackgroundTasks,
    entity_id: str = Form(...),
    confirm: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Replace all vectors for a specific document within a namespace.
    Deletes old vectors matching the filename, then re-indexes the new file.
    Requires confirm="YES".
    """
    try:
        if not rag_system:
            return {
                "responseCode": "01",
                "responseMessage": "RAG system not initialized"
            }

        if confirm.upper() != "YES":
            return {
                "responseCode": "01",
                "responseMessage": "Must confirm with 'YES' to replace document vectors"
            }

        if not file.filename or not file.filename.lower().endswith(('.pdf', '.txt')):
            return {
                "responseCode": "01",
                "responseMessage": "Only PDF and TXT files are supported"
            }

        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return {
                "responseCode": "01",
                "responseMessage": f"File too large ({file_size_mb:.2f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB."
            }

        task_id = generate_task_id()
        tasks[task_id] = {"status": "running", "message": f"Replacing vectors for {file.filename}"}

        def background_replace():
            try:
                logger.info(f"Starting background replace for '{file.filename}' in namespace: {entity_id}")
                result = rag_system.replace_specific_document_vectors(
                    file_bytes=file_bytes,
                    filename=file.filename,
                    namespace=entity_id,
                )
                if result.get('success'):
                    tasks[task_id]["status"] = "done"
                    tasks[task_id]["message"] = f"Successfully replaced vectors for {file.filename}"
                    tasks[task_id]["result"] = result
                else:
                    tasks[task_id]["status"] = "failed"
                    tasks[task_id]["message"] = result.get('error', 'Unknown error')
            except Exception as e:
                logger.error(f"Background replace failed for '{file.filename}': {e}")
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["message"] = f"Failed: {str(e)}"

        background_tasks.add_task(background_replace)

        return {
            "responseCode": "00",
            "responseMessage": "Document vector replacement started",
            "data": {
                "entity_id": entity_id,
                "task_id": task_id,
            }
        }

    except Exception as e:
        logger.error(f"Replace document vectors error: {str(e)}")
        return {
            "responseCode": "01",
            "responseMessage": f"Failed: {str(e)}"
        }


@app.post("/reset-vector-db", response_model=APIResponse)
async def reset_vector_db(
    entity_id: str = Form(...),
    confirm: str = Form(...),
):
    """
    Reset (wipe) all vectors in a specific tenant namespace.
    Does NOT affect other tenants.
    Requires confirm="YES".
    """
    try:
        if not rag_system:
            return {
                "responseCode": "01",
                "responseMessage": "RAG system not initialized"
            }

        if confirm.upper() != "YES":
            return {
                "responseCode": "01",
                "responseMessage": "Must confirm with 'YES' to reset the vector database"
            }

        result = rag_system.reset_vector_database(namespace=entity_id)

        return {
            "responseCode": "00",
            "responseMessage": f"Namespace '{entity_id}' reset successfully",
            "data": result
        }

    except Exception as e:
        logger.error(f"Reset vector database error: {str(e)}")
        return {
            "responseCode": "01",
            "responseMessage": f"Failed: {str(e)}"
        }


@app.get("/stats", response_model=APIResponse)
async def get_stats():
    """
    Retrieve vector database statistics.
    Returns a breakdown of all namespaces (entity_ids) automatically.
    No parameters required.
    """
    try:
        if not rag_system:
            return {
                "responseCode": "01",
                "responseMessage": "RAG system not initialized"
            }

        # Always pull global stats — Pinecone returns per-namespace counts automatically
        raw_stats = rag_system.index.describe_index_stats()
        logger.info(f"Fetched stats successfully: {raw_stats}")

        namespaces = raw_stats.get("namespaces", {})
        entity_ids = {
            ns: {"vector_count": info.get("vector_count", 0)}
            for ns, info in namespaces.items()
        }

        return {
            "responseCode": "00",
            "responseMessage": "Database statistics fetched successfully",
            "data": {
                "total_vectors": raw_stats.get("total_vector_count", 0),
                "index_name": rag_system.index_name,
                "dimension": raw_stats.get("dimension", 512),
                "entity_ids": entity_ids,
            }
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "responseCode": "01",
            "responseMessage": f"Failed: {str(e)}"
        }


@app.get("/entities", response_model=APIResponse)
async def list_entities():
    """
    List all known namespaces (entity_ids) from the Pinecone index.
    Useful for recovery — if a client forgets their entity_id.
    """
    try:
        if not rag_system:
            return {
                "responseCode": "01",
                "responseMessage": "RAG system not initialized"
            }

        stats = rag_system.index.describe_index_stats()
        namespaces = list(stats.get("namespaces", {}).keys())
        logger.info(f"Found {len(namespaces)} namespaces: {namespaces}")

        return {
            "responseCode": "00",
            "responseMessage": f"Found {len(namespaces)} entities",
            "data": {
                "entities": namespaces,
                "count": len(namespaces),
            }
        }

    except Exception as e:
        logger.error(f"Error listing entities: {e}")
        return {
            "responseCode": "01",
            "responseMessage": f"Failed: {str(e)}"
        }


@app.get("/task-status/{task_id}", response_model=APIResponse)
async def task_status(task_id: str):
    """Check status of a background task."""
    task = tasks.get(task_id)
    if not task:
        return {
            "responseCode": "01",
            "responseMessage": "Task ID not found"
        }
    return {
        "responseCode": "00",
        "responseMessage": "Task status retrieved",
        "data": {
            "task_id": task_id,
            "status": task["status"],
            "message": task["message"],
        }
    }


@app.post("/create-session", response_model=APIResponse)
async def create_session():
    """Generate a new session ID for conversation history."""
    session_id = str(uuid.uuid4())
    return {
        "responseCode": "00",
        "responseMessage": "Session created successfully",
        "data": {"session_id": session_id}
    }


@app.post("/ask-question", response_model=APIResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question with RAG retrieval, scoped to the tenant's namespace.
    Pass session_id from /create-session to maintain conversation history.
    Uses sub-query decomposition internally for better results.
    """
    try:
        if not rag_system:
            return {
                "responseCode": "01",
                "responseMessage": "RAG system not initialized"
            }

        logger.info(f"Processing question: '{request.question[:50]}...' for entity: {request.entity_id}")
        result = rag_system.ask_questions(
            question=request.question,
            session_id=request.session_id,
            namespace=request.entity_id,
        )

        if result.get("success"):
            logger.info("Successfully answered question.")
            # Return only what the frontend needs
            sources = [
                {
                    "filename": s.get("filename", ""),
                    "category": s.get("category", ""),
                    "section": s.get("section", ""),
                    "relevance_score": s.get("relevance_score", 0),
                }
                for s in result.get("sources", [])
            ]
            return {
                "responseCode": "00",
                "responseMessage": "Question answered successfully",
                "data": {
                    "answer": result.get("answer"),
                    "sources": sources,
                }
            }
        else:
            error_message = result.get("error", "Unknown error")
            logger.warning(f"Question processing failed: {error_message}")
            return {
                "responseCode": "01",
                "responseMessage": f"Question processing failed: {error_message}",
            }

    except Exception as e:
        logger.error(f"Unexpected error in /ask-question: {str(e)}")
        return {
            "responseCode": "01",
            "responseMessage": f"Unexpected error: {str(e)}"
        }


