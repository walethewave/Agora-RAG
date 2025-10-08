from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import os
import uuid
from dotenv import load_dotenv

from schema.models import MessageRequest, MessageResponse
from src.simple_rag_claude import SimpleRAGSystem

# Load environment variables
load_dotenv()
# -------------------- Settings --------------------
doc_db = os.getenv("DOCUMENT_DATABASE")
if not doc_db:
    raise RuntimeError("Environment variable 'DOCUMENT_DATABASE' is not set.")
UPLOAD_DIR = Path(doc_db)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Initialize system
rag = SimpleRAGSystem()

# -------------------- FastAPI App --------------------
app = FastAPI(
    title="Document & Chatbot API",
    version="1.0.0",
    description="API for PDF upload, vector database update, and chatbot interaction."
)


# -------------------- Endpoints --------------------
@app.post("/upload-pdf", summary="Upload a PDF file")
async def upload_pdf(tag_name: str = Form(...), file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    try:
        unique_filename = f"{tag_name}.pdf"
        file_path = UPLOAD_DIR / unique_filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": unique_filename, "message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.post("/update-vector-db", summary="Update vector database")
def update_vector_db(tag_name: str = Form(...),):
    """
    Update Vector Database
    """
    
    try:
        PDF_PATH = f"./{doc_db}/{tag_name}.pdf"
        doc_id = rag.function_1_pdf_to_pinecone(PDF_PATH, "My Test Document")
        return {"message": f"Vector database Created for Document ID: {doc_id[:8]}..." }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update vector DB: {str(e)}")


@app.post("/chat", response_model=MessageResponse, summary="Send a message to the bot")
def chat(message_req: MessageRequest):
    try:
        user_msg = message_req.user_message
        
        bot_msg = rag.generate_answer(user_msg)
        # If bot_msg is a dict, extract the string message; otherwise, use as is
        if isinstance(bot_msg, dict) and 'answer' in bot_msg:
            bot_message_str = bot_msg['answer']
        else:
            bot_message_str = str(bot_msg)
        return MessageResponse(bot_message=bot_message_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
