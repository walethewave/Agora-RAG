"""
Simplified RAG System - 4 Core Functions
========================================
1. Process complete document (PDF → Chunks → Vectors → Pinecone)
2. Add document to existing collection
3. Replace entire database with new document
4. Ask questions with RAG retrieval

Perfect for backend developers - clean, simple API
"""

import os
import uuid
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
import boto3
import tiktoken
from io import BytesIO
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from fastapi import HTTPException


logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
aws_region = "us-east-1" 
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "simplified-rag-app")
s3_client = boto3.client('s3', region_name=aws_region)

class SimplifiedRAG:
    """Simplified RAG system with 4 core functions for backend integration"""
    
    def __init__(self):
        """Initialize the RAG system with AWS Bedrock and Pinecone"""
        # AWS Bedrock setup
        self.bedrock = boto3.client(
            'bedrock-runtime',
            region_name='us-east-1',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Pinecone setup
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = "rag-documents"
        
        # Create index if it doesn't exist
        if self.index_name not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=self.index_name,
                dimension=1024,  # Titan v2 embedding dimension
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        self.index = pc.Index(self.index_name)
        
        # Model configurations
        self.embedding_model = "amazon.titan-embed-text-v2:0"
        self.chat_model = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        
        # Tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        print("✅ Simplified RAG system initialized successfully!")
    
    def _extract_pdf_text(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract text from PDF with page metadata"""
        try:
            reader = PdfReader(BytesIO(file_bytes))
            pages = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text().strip()
                if text:
                    pages.append({
                        'page_number': page_num,
                        'text': text,
                        'char_count': len(text)
                    })
            
            return pages
            
        except Exception as e:
            raise Exception(f"Failed to extract PDF text: {str(e)}")


    def _get_s3_file_content(self, response, S3_BUCKET: str) -> bytes | None:
        """Retrieve the first PDF file from S3 and return its bytes."""
        try:
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith(".pdf"):
                    logger.info(f"Fetching PDF from S3: {key}")
                    file_obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                    return file_obj["Body"].read()  # return bytes immediately

            logger.warning("No PDF files found in S3 response.")
            return None

        except Exception as e:
            logger.error(f"Error retrieving PDF from S3: {e}")
            return None
    
    def _create_chunks(self, pages: List[Dict], chunk_size: int = 200, overlap_percent: float = 0.1) -> List[Dict[str, Any]]:
        """Create overlapping chunks from pages with metadata"""
        chunks = []
        overlap_tokens = int(chunk_size * overlap_percent)
        
        for page in pages:
            text = page['text']
            tokens = self.tokenizer.encode(text)
            
            # Create chunks for this page
            for i in range(0, len(tokens), chunk_size - overlap_tokens):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                chunks.append({
                    'text': chunk_text,
                    'page_number': page['page_number'],
                    'token_count': len(chunk_tokens),
                    'char_count': len(chunk_text),
                    'chunk_index': len(chunks)
                })
        
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using AWS Bedrock Titan"""
        embeddings = []
        
        for text in texts:
            try:
                import json
                request_body = json.dumps({"inputText": text})
                
                response = self.bedrock.invoke_model(
                    modelId=self.embedding_model,
                    body=request_body,
                    contentType='application/json'
                )
                
                import json
                result = json.loads(response['body'].read())
                embeddings.append(result['embedding'])
                
            except Exception as e:
                print(f"⚠️ Warning: Failed to generate embedding for text: {str(e)}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 1024)
        
        return embeddings
    
    def _upload_to_pinecone(self, chunks: List[Dict], embeddings: List[List[float]], 
                           document_id: str, filename: str) -> Dict[str, Any]:
        """Upload chunks and embeddings to Pinecone with rich metadata"""
        vectors = []
        timestamp = datetime.now().isoformat()
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{document_id}_chunk_{i}"
            
            metadata = {
                'document_id': document_id,
                'filename': filename,
                'page_number': chunk['page_number'],
                'chunk_index': chunk['chunk_index'],
                'text': chunk['text'][:1000],  # Limit text in metadata
                'token_count': chunk['token_count'],
                'char_count': chunk['char_count'],
                'created_at': timestamp,
                'chunk_type': 'text'
            }
            
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        # Upload in batches
        batch_size = 100
        total_uploaded = 0
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
            total_uploaded += len(batch)
        
        return {
            'vectors_uploaded': total_uploaded,
            'document_id': document_id,
            'timestamp': timestamp
        }
    
    # =================
    # CORE FUNCTIONS
    # =================
    
    def _function_1_process_complete_document(self, filename: str, 
                                           chunk_size: int = 200) -> Dict[str, Any]:
        """
        FUNCTION 1: Complete PDF Processing Pipeline
        
        Takes a PDF, processes it completely: PDF → Chunks → Embeddings → Pinecone
        Perfect for backend developers - one call does everything.
        
        Args:
            pdf_path: Path to PDF file
            document_name: Optional custom name (defaults to filename)
            chunk_size: Chunk size in tokens (default 200)
            
        Returns:
            Dict with processing results and metadata for backend tracking
        """
        start_time = time.time()
        
        try:
            # Generate document ID and name
            document_id = str(uuid.uuid4())
            
            print(f"🚀 Processing document: {filename}")
            
            # Step 1: Extract PDF text
            print("📄 Extracting file bytes...")
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET,
                Prefix=f"{filename.lower().replace(' ', '_')}.pdf"
            )
            
            if 'Contents' not in response:
                raise HTTPException(status_code=404, detail="No files found for the specified company")
            
            file_bytes = self._get_s3_file_content(response, S3_BUCKET)
            pages = self._extract_pdf_text(file_bytes) #type: ignore
            total_pages = len(pages)
            
            # Step 2: Create chunks
            print("✂️ Creating chunks...")
            chunks = self._create_chunks(pages, chunk_size)
            total_chunks = len(chunks)
            
            # Step 3: Generate embeddings
            print("🧠 Generating embeddings...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self._generate_embeddings(chunk_texts)
            
            # Step 4: Upload to Pinecone
            print("📤 Uploading to Pinecone...")
            upload_result = self._upload_to_pinecone(chunks, embeddings, document_id, filename)
            
            # Calculate processing time and statistics
            processing_time = time.time() - start_time
            avg_chunk_length = sum(chunk['char_count'] for chunk in chunks) / len(chunks)
            total_tokens = sum(chunk['token_count'] for chunk in chunks)
            
            result = {
                'success': True,
                'document_id': document_id,
                'filename': filename,
                'processing_time_seconds': round(processing_time, 2),
                'total_pages': total_pages,
                'total_chunks': total_chunks,
                'total_tokens': total_tokens,
                'chunk_size_used': chunk_size,
                'avg_chunk_length': round(avg_chunk_length, 1),
                'pinecone_vectors_uploaded': upload_result['vectors_uploaded'],
                'created_at': upload_result['timestamp'],
                'metadata': {
                    'embedding_model': self.embedding_model,
                    'index_name': self.index_name
                }
            }
            
            print(f"✅ SUCCESS! Document processed completely in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'document_id': None,
                'processing_time_seconds': time.time() - start_time
            }
    
    def function_2_add_to_existing_collection(self, document_name: str, 
                                            chunk_size: int = 200) -> Dict[str, Any]:
        """
        FUNCTION 2: Add Document to Existing Collection
        
        Adds a new document to the existing Pinecone database without removing anything.
        Perfect for expanding your knowledge base.
        
        Args:
            pdf_path: Path to PDF file
            document_name: Optional custom name
            chunk_size: Chunk size in tokens
            
        Returns:
            Dict with processing results
        """
        print("➕ Adding document to existing collection...")
        
        # Get current document count
        stats = self.index.describe_index_stats()
        initial_vector_count = stats['total_vector_count']
        
        # Process the document (same as function 1)
        result = self._function_1_process_complete_document( document_name, chunk_size)
        
        if result['success']:
            # Update result with collection info
            new_stats = self.index.describe_index_stats()
            result['collection_info'] = {
                'total_vectors_before': initial_vector_count,
                'total_vectors_after': new_stats['total_vector_count'],
                'vectors_added': result['pinecone_vectors_uploaded']
            }
            
            print(f"✅ Document added! Collection now has {new_stats['total_vector_count']} total vectors")
        
        return result
    
    def function_3_replace_entire_database(self, document_name: str, 
                                         chunk_size: int = 200) -> Dict[str, Any]:
        """
        FUNCTION 3: Replace Entire Database
        
        Deletes ALL existing documents and uploads this new one.
        Use with caution - this wipes everything!
        
        Args:
            pdf_path: Path to PDF file
            document_name: Optional custom name
            chunk_size: Chunk size in tokens
            
        Returns:
            Dict with processing results
        """
        print("🔄 Replacing entire database...")
        
        try:
            # Get current stats
            initial_stats = self.index.describe_index_stats()
            initial_count = initial_stats['total_vector_count']
            
            print(f"⚠️ Deleting {initial_count} existing vectors...")
            
            # Delete all existing vectors
            self.index.delete(delete_all=True)
            
            print("🗑️ Database cleared!")
            
            # Process new document
            result = self._function_1_process_complete_document( document_name, chunk_size)
            
            if result['success']:
                result['database_replacement_info'] = {
                    'vectors_deleted': initial_count,
                    'new_vectors_uploaded': result['pinecone_vectors_uploaded'],
                    'replacement_completed': True
                }
                
                print(f"✅ Database replaced! Old: {initial_count} vectors → New: {result['pinecone_vectors_uploaded']} vectors")
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'database_replacement_info': {
                    'vectors_deleted': 0,
                    'new_vectors_uploaded': 0,
                    'replacement_completed': False
                }
            }
    
    def function_4_ask_questions(self, question: str) -> Dict[str, Any]:
        """
        FUNCTION 4: Ask Questions with RAG Retrieval
        
        Query the knowledge base and get AI-generated answers with sources.
        Uses static top_k=5 for consistent retrieval.
        
        Args:
            question: The question to ask
            
        Returns:
            Dict with answer, sources, and metadata
        """
        top_k = 5  # Static value for consistent retrieval
        start_time = time.time()
        
        try:
            print(f"🤔 Processing question: {question}")
            
            # Step 1: Generate question embedding
            question_embedding = self._generate_embeddings([question])[0]
            
            # Step 2: Search Pinecone for relevant chunks
            search_results = self.index.query(
                vector=question_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            if not search_results['matches']:
                return {
                    'success': False,
                    'error': 'No relevant documents found in the knowledge base',
                    'answer': None,
                    'sources': [],
                    'query_time_seconds': time.time() - start_time
                }
            
            # Step 3: Prepare context for Claude
            context_chunks = []
            sources = []
            
            for match in search_results['matches']:
                metadata = match['metadata']
                context_chunks.append(metadata['text'])
                sources.append({
                    'document_id': metadata['document_id'],
                    'filename': metadata['filename'],
                    'page_number': metadata['page_number'],
                    'relevance_score': round(match['score'], 3),
                    'chunk_index': metadata['chunk_index']
                })
            
            # Step 4: Generate answer with Claude
            context_text = "\n\n".join(context_chunks)
            
            prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context_text}

Question: {question}

Answer:"""

            import json
            request_body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            })
            
            response = self.bedrock.invoke_model(
                modelId=self.chat_model,
                body=request_body,
                contentType='application/json'
            )
            
            import json
            result = json.loads(response['body'].read())
            answer = result['content'][0]['text'] if 'content' in result else result.get('completion', 'No answer generated')
            
            query_time = time.time() - start_time
            
            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'question': question,
                'query_time_seconds': round(query_time, 2),
                'chunks_retrieved': len(context_chunks),
                'metadata': {
                    'embedding_model': self.embedding_model,
                    'chat_model': self.chat_model,
                    'top_k_used': top_k
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'answer': None,
                'sources': [],
                'query_time_seconds': time.time() - start_time
            }
    
    # =================
    # UTILITY FUNCTIONS
    # =================
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats['total_vector_count'],
                'index_fullness': stats.get('index_fullness', 0),
                'dimension': 1024,
                'index_name': self.index_name
            }
        except Exception as e:
            return {'error': str(e)}
    
    def list_all_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database with metadata"""
        try:
            # Query to get some sample vectors and extract document info
            sample_results = self.index.query(
                vector=[0.0] * 1024,  # Dummy vector
                top_k=1000,  # Get many results to find all documents
                include_metadata=True
            )
            
            # Group by document_id
            documents = {}
            for match in sample_results['matches']:
                metadata = match['metadata']
                doc_id = metadata['document_id']
                
                if doc_id not in documents:
                    documents[doc_id] = {
                        'document_id': doc_id,
                        'filename': metadata['filename'],
                        'created_at': metadata['created_at'],
                        'chunk_count': 0
                    }
                documents[doc_id]['chunk_count'] += 1
            
            return list(documents.values())
            
        except Exception as e:
            return []


# =================
# INTERACTIVE DEMO
# =================

# def run_interactive_demo():
#     """Interactive demo for testing the 4 core functions"""
#     print("🚀 Simplified RAG System - Interactive Demo")
#     print("=" * 50)
    
#     try:
#         rag = SimplifiedRAG()
        
#         while True:
#             print(f"\n📊 Database Stats: {rag.get_database_stats()['total_vectors']} vectors")
            
#             print("\n🔧 SELECT FUNCTION:")
#             print("1. 🚀 Process Complete Document (PDF → Pinecone)")
#             print("2. ➕ Add Document to Existing Collection")
#             print("3. 🔄 Replace Entire Database")
#             print("4. ❓ Ask Questions")
#             print("5. 📋 List All Documents")
#             print("6. 🚪 Exit")
#             print("=" * 50)
            
#             choice = input("👉 Select function (1-6): ").strip()
            
#             if choice == "1":
#                 print("\n🚀 FUNCTION 1: Process Complete Document")
#                 pdf_path = input("📄 Enter PDF file path: ").strip()
#                 doc_name = input("📝 Document name (optional): ").strip() or None
                
#                 result = rag.function_1_process_complete_document(pdf_path, doc_name)
                
#                 if result['success']:
#                     print(f"\n✅ SUCCESS!")
#                     print(f"📄 Document ID: {result['document_id']}")
#                     print(f"⏱️ Processing Time: {result['processing_time_seconds']}s")
#                     print(f"📊 Total Chunks: {result['total_chunks']}")
#                     print(f"📤 Vectors Uploaded: {result['pinecone_vectors_uploaded']}")
#                 else:
#                     print(f"\n❌ ERROR: {result['error']}")
            
#             elif choice == "2":
#                 print("\n➕ FUNCTION 2: Add to Existing Collection")
#                 pdf_path = input("📄 Enter PDF file path: ").strip()
#                 doc_name = input("📝 Document name (optional): ").strip() or None
                
#                 result = rag.function_2_add_to_existing_collection(pdf_path, doc_name)
                
#                 if result['success']:
#                     print(f"\n✅ Document Added!")
#                     print(f"📄 Document ID: {result['document_id']}")
#                     print(f"📊 Vectors Added: {result['pinecone_vectors_uploaded']}")
#                     if 'collection_info' in result:
#                         info = result['collection_info']
#                         print(f"📈 Total Vectors: {info['total_vectors_before']} → {info['total_vectors_after']}")
#                 else:
#                     print(f"\n❌ ERROR: {result['error']}")
            
#             elif choice == "3":
#                 print("\n🔄 FUNCTION 3: Replace Entire Database")
#                 print("⚠️ WARNING: This will DELETE ALL existing documents!")
#                 confirm = input("Type 'YES' to confirm: ").strip()
                
#                 if confirm == "YES":
#                     pdf_path = input("📄 Enter PDF file path: ").strip()
#                     doc_name = input("📝 Document name (optional): ").strip() or None
                    
#                     result = rag.function_3_replace_entire_database(pdf_path, doc_name)
                    
#                     if result['success']:
#                         print(f"\n✅ Database Replaced!")
#                         print(f"📄 New Document ID: {result['document_id']}")
#                         info = result['database_replacement_info']
#                         print(f"🗑️ Deleted: {info['vectors_deleted']} vectors")
#                         print(f"📤 Uploaded: {info['new_vectors_uploaded']} vectors")
#                     else:
#                         print(f"\n❌ ERROR: {result['error']}")
#                 else:
#                     print("❌ Operation cancelled")
            
#             elif choice == "4":
#                 print("\n❓ FUNCTION 4: Ask Questions")
                
#                 while True:
#                     question = input("\n🤔 Your question (or 'back' to return): ").strip()
#                     if question.lower() == 'back':
#                         break
#                     if not question:
#                         continue
                    
#                     result = rag.function_4_ask_questions(question)
                    
#                     if result['success']:
#                         print(f"\n💡 Answer:")
#                         print(result['answer'])
                        
#                         if result['sources']:
#                             print(f"\n📚 Sources ({len(result['sources'])}):")
#                             for i, source in enumerate(result['sources'][:3], 1):
#                                 print(f"   {i}. {source['filename']} (Page {source['page_number']}) - Score: {source['relevance_score']}")
                        
#                         print(f"\n⏱️ Query Time: {result['query_time_seconds']}s")
#                     else:
#                         print(f"\n❌ ERROR: {result['error']}")
            
#             elif choice == "5":
#                 print("\n📋 ALL DOCUMENTS")
#                 docs = rag.list_all_documents()
                
#                 if docs:
#                     for doc in docs:
#                         print(f"\n📄 {doc['filename']}")
#                         print(f"   ID: {doc['document_id']}")
#                         print(f"   Chunks: {doc['chunk_count']}")
#                         print(f"   Created: {doc['created_at']}")
#                 else:
#                     print("📭 No documents found")
            
#             elif choice == "6":
#                 print("\n👋 Goodbye! Your documents are safely stored in Pinecone.")
#                 break
            
#             else:
#                 print("❌ Invalid choice! Please select 1-6.")
    
#     except Exception as e:
#         print(f"❌ System Error: {e}")
#         print("\n💡 Make sure:")
#         print("1. Your .env file has all required API keys")
#         print("2. AWS credentials are configured")
#         print("3. Pinecone API key is valid")


# if __name__ == "__main__":
#     run_interactive_demo()