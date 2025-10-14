"""
FastAPI Backend for Simplified RAG System
=========================================
REST API endpoints for the 4 core RAG functions:
1. POST /process-document - Complete PDF processing
2. POST /add-document - Add to existing collection  
3. POST /replace-database - Replace entire database
4. POST /ask-question - Ask questions with RAG

Perfect for backend integration!
"""

import os
import boto3
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from botocore.exceptions import NoCredentialsError, ClientError
from fastapi.responses import JSONResponse
from typing import Optional
from mangum import Mangum

from src.simplified_rag import SimplifiedRAG
from src.models import QuestionRequest

# Initialize FastAPI app
app = FastAPI(
    title="Simplified RAG API",
    description="4 core functions for RAG document processing and Q&A",
    version="1.0.0"
)

handler = Mangum(app)

# Initializations
rag_system = None
MAX_FILE_SIZE_MB = 2  # 2 MB limit
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "simplified-rag-app")
S3_REGION = os.getenv("AWS_REGION", "us-east-1")
s3_client = boto3.client("s3", region_name=S3_REGION)


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    try:
        rag_system = SimplifiedRAG()
        print("✅ RAG system initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")



@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Simplified RAG API is running!",
        "functions": [
            "POST /process-document - Complete PDF processing",
            "POST /add-document - Add to existing collection", 
            "POST /replace-database - Replace entire database",
            "POST /ask-question - Ask questions with RAG",
            "GET /stats - Get database statistics"
        ]
    }


@app.post("/upload_file")
async def upload_document(
    file_name: str = Form(...),
    file: UploadFile = File(...)
):
    # Ensure file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read file bytes to check size
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        print(2)
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_size_mb:.2f} MB). Max allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    try:
        # Reset file pointer for upload
        file.file.seek(0)

        # S3 key and upload
        s3_key = f"{file_name.lower().replace(' ', '_')}.pdf"
        s3_client.upload_fileobj(
            file.file,
            S3_BUCKET,
            s3_key,
            ExtraArgs={"ContentType": file.content_type}
        )

        file_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        return {"message": "Upload successful", "file_url": file_url}

    except NoCredentialsError:
        raise HTTPException(status_code=401, detail="AWS credentials not available")
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        stats = rag_system.get_database_stats()
        documents = rag_system.list_all_documents()
        
        return {
            "success": True,
            "stats": stats,
            "document_count": len(documents),
            "documents": documents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-database")
async def update_database(
    background_tasks: BackgroundTasks,
    document_name: str = Form(...),
    chunk_size: Optional[int] = Form(200)
):
    """
    FUNCTION 2: Add Document to Existing Collection
    Adds new document without removing existing ones
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    def background_add():
        rag_system.function_2_add_to_existing_collection(
            document_name=document_name,
            chunk_size=chunk_size
        )

    background_tasks.add_task(background_add)

    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "message": f"Background task started to add '{document_name}' to collection."
        }
    )


@app.post("/replace-database")
async def replace_database(
    background_tasks: BackgroundTasks,
    document_name: str = Form(...),
    chunk_size: Optional[int] = Form(200),
    confirm: str = Form(...)
):
    """
    FUNCTION 3: Replace Entire Database
    ⚠️ WARNING: Deletes ALL existing documents and uploads this new one
    Requires confirm="YES" to proceed
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    if confirm.lower() != "yes":
        raise HTTPException(
            status_code=400, 
            detail="Must confirm with 'YES' to replace entire database"
        )

    def background_replace():
        rag_system.function_3_replace_entire_database(
            document_name=document_name,
            chunk_size=chunk_size
        )

    background_tasks.add_task(background_replace)

    return JSONResponse(
        status_code=202,
        content={
            "success": True,
            "message": f"Background database replacement started for '{document_name}'."
        }
    )



@app.post("/ask-question")
async def ask_question(request: QuestionRequest):
    """
    FUNCTION 4: Ask Questions with RAG Retrieval
    Query the knowledge base and get AI-generated answers with sources
    """
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        result = rag_system.function_4_ask_questions(
            question=request.question
        )
        
        if result['success']:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Question answered successfully",
                    "data": result
                }
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": f"Question processing failed: {result['error']}",
                    "data": result
                }
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Development server runner
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)