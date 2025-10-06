import os
import uuid
import hashlib
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

import PyPDF2
import boto3
from pinecone import Pinecone
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleRAGSystem:
    def __init__(self):
        """Initialize the RAG system with Pinecone and AWS Bedrock (Claude) connections."""
        # Initialize AWS Bedrock client using session pattern
        session = boto3.Session()
        self.bedrock_client = session.client(
            service_name='bedrock-runtime',
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        
        # Model IDs
        self.embedding_model = "amazon.titan-embed-text-v2:0"  # Titan v2 embeddings
        self.chat_model = "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Claude 3.5 Sonnet
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "rag-documents")
        
        # Try to get existing index or create one
        try:
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            print(f"Index {self.index_name} not found. Please create it first with dimension 1024 (for Titan v2 embeddings).")
            raise e
        
        # Initialize tokenizer for chunk size calculation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Configuration
        self.chunk_size = 200  # tokens
        self.overlap_percentage = 0.1  # 10% overlap
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using AWS Bedrock Titan."""
        try:
            embeddings = []
            for text in texts:
                # Prepare the request body for Titan embeddings
                body = json.dumps({
                    "inputText": text
                })
                
                # Call Bedrock
                response = self.bedrock_client.invoke_model(
                    modelId=self.embedding_model,
                    body=body,
                    contentType='application/json',
                    accept='application/json'
                )
                
                # Parse response
                response_body = json.loads(response['body'].read())
                embedding = response_body['embedding']
                embeddings.append(embedding)
            
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise e
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            raise e
    
    def _create_chunks(self, text: str, document_id: str, filename: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata.
        Each chunk will be ~200 tokens with 10% overlap.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        overlap_size = int(self.chunk_size * self.overlap_percentage)
        step_size = self.chunk_size - overlap_size
        
        for i in range(0, len(tokens), step_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Determine page number (rough estimation based on text position)
            page_num = self._estimate_page_number(text, chunk_text)
            
            chunk_id = str(uuid.uuid4())
            chunk_metadata = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "page_number": page_num,
                "filename": filename,
                "chunk_index": len(chunks),
                "created_at": datetime.now().isoformat(),
                "token_count": len(chunk_tokens)
            }
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _estimate_page_number(self, full_text: str, chunk_text: str) -> int:
        """Estimate page number based on chunk position in the full text."""
        try:
            chunk_start = full_text.find(chunk_text[:50])  # Use first 50 chars to find position
            pages_before = full_text[:chunk_start].count("--- Page")
            return max(1, pages_before)
        except:
            return 1
    
    def function_1_pdf_to_pinecone(self, pdf_path: str, document_name: Optional[str] = None) -> str:
        """
        Function 1: Takes a PDF, converts to vectors, and uploads to Pinecone.
        
        Args:
            pdf_path (str): Path to the PDF file
            document_name (str, optional): Custom name for the document
            
        Returns:
            str: Document ID that can be used for future operations
        """
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            filename = document_name or os.path.basename(pdf_path)
            
            print(f"Document ID: {document_id}")
            print(f"Filename: {filename}")
            
            # Step 1: Extract text from PDF
            print("Extracting text from PDF...")
            text = self._extract_text_from_pdf(pdf_path)
            
            # Step 2: Create chunks with metadata
            print("Creating chunks with metadata...")
            chunks = self._create_chunks(text, document_id, filename)
            print(f"Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            print("Generating embeddings...")
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self._get_embeddings(chunk_texts)
            
            # Step 4: Upload to Pinecone
            print("Uploading to Pinecone...")
            vectors_to_upsert = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors_to_upsert.append({
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": {
                        **chunk["metadata"],
                        "text": chunk["text"]  # Store text in metadata for retrieval
                    }
                })
            
            # Batch upsert to Pinecone
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            print(f"Successfully uploaded {len(vectors_to_upsert)} vectors to Pinecone")
            print(f"Document ID: {document_id}")
            
            return document_id
            
        except Exception as e:
            print(f"Error in function_1_pdf_to_pinecone: {e}")
            raise e
    
    def function_2_upload_additional_document(self, pdf_path: str, document_name: Optional[str] = None) -> str:
        """
        Function 2: Upload additional documents to the existing vector database.
        This function can be called multiple times to add more documents.
        
        Args:
            pdf_path (str): Path to the new PDF file
            document_name (str, optional): Custom name for the document
            
        Returns:
            str: Document ID of the newly uploaded document
        """
        try:
            print(f"Uploading additional document: {pdf_path}")
            
            # This function is essentially the same as function_1 but with different messaging
            document_id = self.function_1_pdf_to_pinecone(pdf_path, document_name)
            
            print(f"Additional document uploaded successfully with ID: {document_id}")
            return document_id
            
        except Exception as e:
            print(f"Error in function_2_upload_additional_document: {e}")
            raise e
    
    def function_3_upload_and_chunk_document(self, pdf_path: str, document_name: Optional[str] = None, custom_chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Function 3: Upload document and return detailed chunking information.
        This function provides more control and transparency over the chunking process.
        
        Args:
            pdf_path (str): Path to the PDF file
            document_name (str, optional): Custom name for the document
            custom_chunk_size (int, optional): Custom chunk size in tokens
            
        Returns:
            Dict: Detailed information about the document and its chunks
        """
        try:
            print(f"Uploading and chunking document: {pdf_path}")
            
            # Temporarily set custom chunk size if provided
            original_chunk_size = self.chunk_size
            if custom_chunk_size:
                self.chunk_size = custom_chunk_size
                print(f"Using custom chunk size: {custom_chunk_size} tokens")
            
            # Generate unique document ID
            document_id = str(uuid.uuid4())
            filename = document_name or os.path.basename(pdf_path)
            
            # Extract text
            text = self._extract_text_from_pdf(pdf_path)
            
            # Create chunks
            chunks = self._create_chunks(text, document_id, filename)
            
            # Generate embeddings
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self._get_embeddings(chunk_texts)
            
            # Upload to Pinecone
            vectors_to_upsert = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors_to_upsert.append({
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": {
                        **chunk["metadata"],
                        "text": chunk["text"]
                    }
                })
            
            # Batch upsert
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            # Restore original chunk size
            self.chunk_size = original_chunk_size
            
            # Return detailed information
            result = {
                "document_id": document_id,
                "filename": filename,
                "total_chunks": len(chunks),
                "total_tokens": sum(len(self.tokenizer.encode(chunk["text"])) for chunk in chunks),
                "chunk_size_used": custom_chunk_size or original_chunk_size,
                "overlap_percentage": self.overlap_percentage,
                "chunks_info": [
                    {
                        "chunk_id": chunk["id"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                        "page_number": chunk["metadata"]["page_number"],
                        "token_count": chunk["metadata"]["token_count"],
                        "preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
                    }
                    for chunk in chunks
                ]
            }
            
            print(f"Document processed successfully:")
            print(f"- Document ID: {document_id}")
            print(f"- Total chunks: {len(chunks)}")
            print(f"- Chunk size: {custom_chunk_size or original_chunk_size} tokens")
            
            return result
            
        except Exception as e:
            print(f"Error in function_3_upload_and_chunk_document: {e}")
            raise e
    
    def function_4_update_document(self, document_id: str, new_pdf_path: str, document_name: Optional[str] = None) -> str:
        """
        Function 4: Update an existing document by deleting old vectors and uploading new ones.
        Uses document_id metadata to identify and delete the old document.
        
        Args:
            document_id (str): The document ID to update
            new_pdf_path (str): Path to the new PDF file
            document_name (str, optional): New name for the document
            
        Returns:
            str: The same document ID (for consistency)
        """
        try:
            print(f"Updating document with ID: {document_id}")
            
            # Step 1: Delete all vectors with this document_id
            print("Deleting old document vectors...")
            
            # Query to find all vectors with this document_id
            query_results = self.index.query(
                vector=[0.0] * 1024,  # Dummy vector for metadata filtering (Titan embeddings are 1024 dim)
                filter={"document_id": document_id},
                top_k=10000,  # Large number to get all matches
                include_metadata=True
            )
            
            if query_results.matches:
                # Delete all matching vectors
                vector_ids_to_delete = [match.id for match in query_results.matches]
                
                # Delete in batches
                batch_size = 1000
                for i in range(0, len(vector_ids_to_delete), batch_size):
                    batch_ids = vector_ids_to_delete[i:i + batch_size]
                    self.index.delete(ids=batch_ids)
                
                print(f"Deleted {len(vector_ids_to_delete)} old vectors")
            else:
                print(f"No existing vectors found for document_id: {document_id}")
            
            # Step 2: Process and upload the new document with the same document_id
            print("Uploading new document content...")
            
            filename = document_name or os.path.basename(new_pdf_path)
            
            # Extract text from new PDF
            text = self._extract_text_from_pdf(new_pdf_path)
            
            # Create new chunks with the same document_id
            chunks = self._create_chunks(text, document_id, filename)
            
            # Generate embeddings
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self._get_embeddings(chunk_texts)
            
            # Upload new vectors
            vectors_to_upsert = []
            for chunk, embedding in zip(chunks, embeddings):
                vectors_to_upsert.append({
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": {
                        **chunk["metadata"],
                        "text": chunk["text"],
                        "updated_at": datetime.now().isoformat()  # Add update timestamp
                    }
                })
            
            # Batch upsert new vectors
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            print(f"Successfully updated document {document_id}")
            print(f"- New filename: {filename}")
            print(f"- New chunk count: {len(chunks)}")
            
            return document_id
            
        except Exception as e:
            print(f"Error in function_4_update_document: {e}")
            raise e
    
    def query_documents(self, question: str, top_k: int = 5, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the document database to find relevant information.
        
        Args:
            question (str): The question to ask
            top_k (int): Number of relevant chunks to retrieve
            document_id (str, optional): Limit search to specific document
            
        Returns:
            Dict: Query results with relevant chunks and metadata
        """
        try:
            print(f"Querying: {question}")
            
            # Generate embedding for the question
            question_embedding = self._get_embeddings([question])[0]
            
            # Build filter if document_id is specified
            filter_dict = {"document_id": document_id} if document_id else None
            
            # Query Pinecone
            results = self.index.query(
                vector=question_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            relevant_chunks = []
            for match in results.matches:
                chunk_info = {
                    "chunk_id": match.id,
                    "score": match.score,
                    "text": match.metadata.get("text", ""),
                    "document_id": match.metadata.get("document_id", ""),
                    "filename": match.metadata.get("filename", ""),
                    "page_number": match.metadata.get("page_number", 1),
                    "chunk_index": match.metadata.get("chunk_index", 0)
                }
                relevant_chunks.append(chunk_info)
            
            return {
                "question": question,
                "relevant_chunks": relevant_chunks,
                "total_results": len(relevant_chunks)
            }
            
        except Exception as e:
            print(f"Error in query_documents: {e}")
            raise e
    
    def generate_answer(self, question: str, top_k: int = 5, document_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an answer to a question based on the document database using Claude.
        
        Args:
            question (str): The question to answer
            top_k (int): Number of relevant chunks to use for context
            document_id (str, optional): Limit search to specific document
            
        Returns:
            Dict: Answer with sources and relevant chunks
        """
        try:
            # Get relevant chunks
            query_results = self.query_documents(question, top_k, document_id)
            
            if not query_results["relevant_chunks"]:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": [],
                    "relevant_chunks": []
                }
            
            # Prepare context from relevant chunks
            context_parts = []
            sources = []
            
            for chunk in query_results["relevant_chunks"]:
                context_parts.append(f"From {chunk['filename']} (Page {chunk['page_number']}):\n{chunk['text']}")
                
                source_info = {
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "document_id": chunk["document_id"],
                    "relevance_score": chunk["score"]
                }
                sources.append(source_info)
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using Claude via AWS Bedrock
            prompt = f"""Human: You are a helpful assistant that answers questions based on provided document context. 
Use only the information from the provided context to answer questions. 
If the context doesn't contain enough information to answer the question, say so clearly.
Be concise and accurate in your responses.

Context from documents:
{context}

Question: {question}

Please provide a clear and accurate answer based only on the information in the context above.
            Assistant: """
            
            # Prepare request body for Claude
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            # Call Claude
            response = self.bedrock_client.invoke_model(
                modelId=self.chat_model,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            answer = response_body['content'][0]['text']
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "relevant_chunks": query_results["relevant_chunks"]
            }
            
        except Exception as e:
            print(f"Error in generate_answer: {e}")
            raise e
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the vector database.
        
        Returns:
            List: Information about all stored documents
        """
        try:
            # Query with a dummy vector to get all documents
            results = self.index.query(
                vector=[0.0] * 1024,
                top_k=10000,
                include_metadata=True
            )
            
            # Group by document_id
            documents = {}
            for match in results.matches:
                doc_id = match.metadata.get("document_id")
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": match.metadata.get("filename", "Unknown"),
                        "chunk_count": 0,
                        "created_at": match.metadata.get("created_at"),
                        "updated_at": match.metadata.get("updated_at")
                    }
                documents[doc_id]["chunk_count"] += 1
            
            return list(documents.values())
            
        except Exception as e:
            print(f"Error in list_documents: {e}")
            raise e


# If this file is run directly, execute a test
if __name__ == "__main__":
    # PDF path - UPDATE this to your actual PDF file path
    PDF_PATH = "Hands On Machine Learning with Scikit Learn and TensorFlow (1).pdf"
    
    print("🚀 Running Simple RAG System Test")
    print(f"📄 PDF Path: {PDF_PATH}")
    
    try:
        # Initialize the system
        rag = SimpleRAGSystem()
        
        # Test Function 1
        print("\n📤 Testing Function 1: PDF to Pinecone")
        doc_id = rag.function_1_pdf_to_pinecone(PDF_PATH, "Test Document")
        print(f"✅ Document uploaded with ID: {doc_id[:8]}...")
        
        # Test querying
        print("\n❓ Testing question answering")
        answer = rag.generate_answer("What is this document about?")
        print(f"💡 Answer: {answer['answer']}")
        
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure:")
        print("1. Update PDF_PATH to your actual file")
        print("2. Your .env file has correct API keys")
        print("3. Pinecone index exists with dimension 1024")
