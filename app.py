"""
FastAPI Backend for Simplified RAG System
=========================================
REST API endpoints for the core RAG functions:
1. POST /insert-doc-vector-db - Upload PDF & add to existing collection
2. POST /replace-document-vectors - Replace vectors for a specific document
3. POST /reset-vector-db - Wipe entire database
4. POST /ask-question - Ask questions with RAG
5. GET  /stats - Database statistics

Direct PDF upload - no S3 dependency.
"""

# --- Standard Library Imports ---
import os
import uuid
import json
import logging
from typing import Optional

# --- Third-Party Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from mangum import Mangum

# --- Local Application Imports ---
from src.simplified_rag import SimplifiedRAG
from src.models import QuestionRequest, response

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Simplified RAG API",
    description="Core functions for RAG document processing and Q&A (direct PDF upload)",
    version="2.0.0"
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


@app.get("/", response_model=response)
async def root():
    """API Health Check Endpoint."""
    return {
        "responseCode": "00",
        "responseMessage": "Simplified RAG API is running successfully",
    }


@app.post("/insert-doc-vector-db", response_model=response)
async def insert_doc_vector_db(
    background_tasks: BackgroundTasks,
    doc_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Upload a PDF and add its Q&A pairs to the vector database.

    - Accepts a PDF file directly (no S3).
    - Parses Q&A pairs from the PDF.
    - Generates embeddings and upserts to Pinecone.
    - Runs in the background.
    """
    try:
        if not rag_system:
            logger.error("POST /insert-doc-vector-db failed: RAG system not initialized.")
            return {
                "responseCode": "01",
                "responseMessage": "RAG system not initialized"
            }

        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            return {
                "responseCode": "01",
                "responseMessage": "Only PDF files are supported"
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
        tasks[task_id] = {"status": "running", "message": f"Processing {doc_id}"}

        def background_update():
            try:
                logger.info(f"Starting background add for doc_id: {doc_id}")
                result = rag_system.add_to_existing_collection(
                    file_bytes=file_bytes,
                    filename=doc_id
                )
                if result.get('success'):
                    tasks[task_id]["status"] = "done"
                    tasks[task_id]["message"] = f"Successfully added {doc_id}"
                    tasks[task_id]["result"] = result
                else:
                    tasks[task_id]["status"] = "failed"
                    tasks[task_id]["message"] = result.get('error', 'Unknown error')
                logger.info(f"Background add completed for doc_id: {doc_id}")
            except Exception as e:
                logger.error(f"Background add failed for doc_id {doc_id}: {e}")
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["message"] = f"Failed: {str(e)}"

        background_tasks.add_task(background_update)

        return {
            "responseCode": "00",
            "responseMessage": f"Background task started to process '{doc_id}'.",
            "data": {
                "doc_id": doc_id,
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


@app.post("/replace-document-vectors", response_model=response)
async def replace_document_vectors_endpoint(
    background_tasks: BackgroundTasks,
    doc_id: str = Form(...),
    confirm: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Replace vectors for a specific document.
    Upload new PDF, delete old vectors matching doc_id, and re-index.
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

        if not file.filename or not file.filename.lower().endswith('.pdf'):
            return {
                "responseCode": "01",
                "responseMessage": "Only PDF files are supported"
            }

        file_bytes = await file.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return {
                "responseCode": "01",
                "responseMessage": f"File too large ({file_size_mb:.2f} MB). Max allowed is {MAX_FILE_SIZE_MB} MB."
            }

        task_id = generate_task_id()
        tasks[task_id] = {"status": "running", "message": f"Replacing vectors for {doc_id}"}

        def background_replace():
            try:
                logger.info(f"Starting background replace for {doc_id}")
                result = rag_system.replace_specific_document_vectors(
                    file_bytes=file_bytes,
                    filename=doc_id
                )
                if result.get('success'):
                    tasks[task_id]["status"] = "done"
                    tasks[task_id]["message"] = f"Successfully replaced vectors for {doc_id}"
                    tasks[task_id]["result"] = result
                else:
                    tasks[task_id]["status"] = "failed"
                    tasks[task_id]["message"] = result.get('error', 'Unknown error')
            except Exception as e:
                logger.error(f"Background replace failed for {doc_id}: {e}")
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["message"] = f"Failed: {str(e)}"

        background_tasks.add_task(background_replace)

        return {
            "responseCode": "00",
            "responseMessage": "Document vector replacement started",
            "data": {
                "doc_id": doc_id,
                "task_id": task_id,
            }
        }

    except Exception as e:
        logger.error(f"Replace document vectors error: {str(e)}")
        return {
            "responseCode": "01",
            "responseMessage": f"Failed: {str(e)}"
        }


@app.post("/reset-vector-db", response_model=response)
async def reset_vector_db(
    confirm: str = Form(...)
):
    """
    Reset (wipe) the entire vector database.
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

        result = rag_system.reset_vector_database()

        return {
            "responseCode": "00",
            "responseMessage": "Vector database reset successfully",
            "data": result
        }

    except Exception as e:
        logger.error(f"Reset vector database error: {str(e)}")
        return {
            "responseCode": "01",
            "responseMessage": f"Failed: {str(e)}"
        }


@app.get("/stats", response_model=response)
async def get_stats():
    """Retrieve vector database statistics and document list."""
    try:
        if not rag_system:
            return {
                "responseCode": "01",
                "responseMessage": "RAG system not initialized"
            }

        stats = rag_system.get_database_stats()
        documents = rag_system.list_all_documents()
        logger.info(f"Fetched stats successfully: {stats}")

        return {
            "responseCode": "00",
            "responseMessage": "Database statistics fetched successfully",
            "data": {
                "stats": stats,
                "document_count": len(documents),
                "documents": documents
            }
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "responseCode": "01",
            "responseMessage": f"Failed: {str(e)}"
        }


@app.get("/task-status/{task_id}", response_model=response)
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


@app.post("/ask-question-stream")
async def ask_question_stream(request: QuestionRequest):
    """
    Stream an answer token by token using Server-Sent Events (SSE).
    The client receives `data: {"text": "..."}\n\n` chunks, then `data: [DONE]\n\n`.
    """
    if not rag_system:
        async def error_gen():
            yield f"data: {json.dumps({'text': 'RAG system not initialized'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")

    def generate():
        try:
            logger.info(f"Streaming question: '{request.question[:50]}...'")
            for text_chunk in rag_system.ask_questions_stream(question=request.question):
                if text_chunk:
                    yield f"data: {json.dumps({'text': text_chunk})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'text': f'⚠️ Error: {str(e)}'})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.post("/ask-question", response_model=response)
async def ask_question(request: QuestionRequest):
    """
    Ask a question with RAG retrieval.
    Uses sub-query decomposition internally for better results.
    """
    try:
        if not rag_system:
            return {
                "responseCode": "01",
                "responseMessage": "RAG system not initialized"
            }

        logger.info(f"Processing question: '{request.question[:50]}...'")
        result = rag_system.ask_questions(question=request.question)

        if result.get("success"):
            logger.info("Successfully answered question.")
            return {
                "responseCode": "00",
                "responseMessage": "Question answered successfully",
                "data": result
            }
        else:
            error_message = result.get("error", "Unknown error")
            logger.warning(f"Question processing failed: {error_message}")
            return {
                "responseCode": "01",
                "responseMessage": f"Question processing failed: {error_message}",
                "data": result
            }

    except Exception as e:
        logger.error(f"Unexpected error in /ask-question: {str(e)}")
        return {
            "responseCode": "01",
            "responseMessage": f"Unexpected error: {str(e)}"
        }
