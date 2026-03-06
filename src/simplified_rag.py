"""
Simplified RAG System - Core Functions
=======================================
1. Process uploaded PDF (PDF bytes -> Q&A Chunks -> Vectors -> Pinecone)
2. Add document to existing collection
3. Replace vectors for a specific document
4. Ask questions with RAG retrieval (sub-query decomposition)

Direct PDF upload via FastAPI - no S3 dependency.
Chunking strategy: one chunk per Q&A pair.
"""

import os
import re
import uuid
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Optional, Any
import boto3
import tiktoken
from io import BytesIO
from pinecone import Pinecone, ServerlessSpec
from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv
from fastapi import HTTPException


logger = logging.getLogger(__name__)

# Load environment variables from .env file (searches current dir and all parents)
load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

class SimplifiedRAG:
    """Simplified RAG system with 4 core functions for backend integration"""
        
    def __init__(self):
        """Initialize the RAG system with AWS Bedrock and Pinecone"""
        try:
            # AWS Bedrock setup
            self.bedrock = boto3.client(
                'bedrock-runtime',
                region_name='us-east-1',
            )

            # Pinecone setup
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            self.index_name = os.getenv('PINECONE_INDEX_NAME')
            
            # Create index if it doesn't exist
            if self.index_name not in [index.name for index in pc.list_indexes()]:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=512,  # Titan v2 embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            
            # Connect to the Pinecone index
            self.index = pc.Index(self.index_name)
            
            # Model configurations
            self.embedding_model = "amazon.titan-embed-text-v2:0"  # Embedding model (512-dim)
            self.chat_model = "amazon.nova-lite-v1:0"  # LLM for answer synthesis
            self.fast_model = "amazon.nova-lite-v1:0"  # Fast LLM for sub-queries
            
            # Tokenizer for chunking (matches Claude models)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            logger.info("Simplified RAG system initialized successfully!")
        except Exception as e:
            logger.error(f"Failed to initialize SimplifiedRAG: {e}", exc_info=True)
            raise

    # =========================
    # PDF & CHUNKING
    # =========================

    def _extract_pdf_text(self, file_bytes: bytes) -> str:
        """Extract all text from a PDF and return as a single string."""
        try:
            reader = PdfReader(BytesIO(file_bytes))
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            logger.info(f"Extracted text from {len(reader.pages)} PDF pages.")
            return full_text.strip()
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}", exc_info=True)
            raise Exception(f"Failed to extract PDF text: {str(e)}")

    def _create_qa_chunks(self, full_text: str) -> List[Dict[str, Any]]:
        """
        Parse PDF text into one chunk per Q&A pair.

        Expected format:
            [CATEGORY: SomeName]
            Q: <question>
            A: <answer>

        Normalizes the extracted text first so that Q:, A:, [CATEGORY:,
        and SECTION markers always appear at the start of a line, regardless
        of how PyPDF2 joined them during extraction.
        """
        chunks: List[Dict[str, Any]] = []
        section_pattern = re.compile(r'SECTION\s+\d+\s*[\u2014\u2013\-]\s*(.+)')
        category_pattern = re.compile(r'\[CATEGORY:\s*(.+?)\]')

        # --- Normalization: force key markers onto their own lines ---
        # Handles PDFs where PyPDF2 joins "...text Q: next question" on one line
        full_text = re.sub(r'(?<!\n)(Q:\s)', r'\n\1', full_text)
        full_text = re.sub(r'(?<!\n)(A:\s)', r'\n\1', full_text)
        full_text = re.sub(r'(?<!\n)(\[CATEGORY:)', r'\n\1', full_text)
        full_text = re.sub(r'(?<!\n)(SECTION\s+\d+)', r'\n\1', full_text)

        current_section = "General"
        current_category = "General"

        lines = full_text.split('\n')
        i = 0
        chunk_index = 0

        while i < len(lines):
            line = lines[i].strip()

            section_match = section_pattern.match(line) 
            if section_match:
                current_section = section_match.group(1).strip()
                i += 1
                continue

            category_match = category_pattern.match(line)
            if category_match:
                current_category = category_match.group(1).strip()
                i += 1
                continue

            if line.startswith('Q:'):
                question_text = line[2:].strip()
                i += 1
                while i < len(lines):
                    l = lines[i].strip()
                    if l.startswith('A:') or l.startswith('Q:') or l.startswith('[CATEGORY') or section_pattern.match(l):
                        break
                    question_text += ' ' + l
                    i += 1

                answer_text = ""
                if i < len(lines) and lines[i].strip().startswith('A:'):
                    answer_text = lines[i].strip()[2:].strip()
                    i += 1
                    while i < len(lines):
                        l = lines[i].strip()
                        if l.startswith('Q:') or l.startswith('[CATEGORY') or section_pattern.match(l):
                            break
                        if l == '':
                            j = i + 1
                            while j < len(lines) and lines[j].strip() == '':
                                j += 1
                            if j >= len(lines) or lines[j].strip().startswith('Q:') or lines[j].strip().startswith('[CATEGORY') or section_pattern.match(lines[j].strip()):
                                break
                        answer_text += ' ' + l
                        i += 1

                chunk_text = f"Q: {question_text.strip()}\nA: {answer_text.strip()}"
                tokens = self.tokenizer.encode(chunk_text)

                chunks.append({
                    'text': chunk_text,
                    'question': question_text.strip(),
                    'answer': answer_text.strip(),
                    'section': current_section,
                    'category': current_category,
                    'token_count': len(tokens),
                    'char_count': len(chunk_text),
                    'chunk_index': chunk_index,
                })
                chunk_index += 1
            else:
                i += 1

        logger.info(f"Created {len(chunks)} Q&A chunks from PDF text.")
        return chunks


    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using AWS Bedrock Titan"""
        embeddings = []
        
        # Iterate over each text chunk
        for text in texts:
            try:
                # Format the request body for Bedrock Titan (512-dim output)
                request_body = json.dumps({"inputText": text, "dimensions": 512, "normalize": True})
                
                # Invoke the Bedrock model
                response = self.bedrock.invoke_model(
                    modelId=self.embedding_model,
                    body=request_body,
                    contentType='application/json'
                )
                
                # Parse the response
                result = json.loads(response['body'].read())
                # Append the resulting embedding vector
                embeddings.append(result['embedding'])
                
            except Exception as e:
                # Log a warning and append a zero vector as a fallback
                logger.warning(f"Failed to generate embedding for text chunk: {str(e)}")
                # Use zero vector as fallback to avoid dimension mismatch
                embeddings.append([0.0] * 512)
        
        logger.info(f"Generated {len(embeddings)} embeddings.")
        return embeddings
    
    def _upload_to_pinecone(self, chunks: List[Dict], embeddings: List[List[float]], 
                            document_id: str, filename: str) -> Dict[str, Any]:
        """Upload chunks and embeddings to Pinecone with rich metadata"""
        try:
            vectors = []
            timestamp = datetime.now().isoformat()  # Get current timestamp
            
            # Prepare vectors for upload
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_id}_chunk_{i}"  # Create a unique ID for each chunk
                
                # Create a rich metadata object
                metadata = {
                    'document_id': document_id,
                    'filename': filename,
                    'section': chunk.get('section', ''),
                    'category': chunk.get('category', ''),
                    'question': chunk.get('question', ''),
                    'answer': chunk.get('answer', ''),
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'],
                    'token_count': chunk['token_count'],
                    'char_count': chunk['char_count'],
                    'created_at': timestamp,
                    'chunk_type': 'qa_pair',
                }
                
                # Append the final vector object
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upload in batches for efficiency and reliability
            batch_size = 100
            total_uploaded = 0
            
            logger.info(f"Uploading {len(vectors)} vectors in batches of {batch_size}...")
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)  # Upsert batch to Pinecone
                total_uploaded += len(batch)
            
            logger.info(f"Successfully uploaded {total_uploaded} vectors to Pinecone.")
            
            # Return a summary of the upload
            return {
                'vectors_uploaded': total_uploaded,
                'document_id': document_id,
                'timestamp': timestamp
            }
        except Exception as e:
            logger.error(f"Failed to upload vectors to Pinecone: {e}", exc_info=True)
            raise Exception(f"Pinecone upload failed: {str(e)}") # Propagate error
    
    # =========================
    # CORE FUNCTIONS
    # =========================

    def process_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Complete PDF Processing Pipeline (from bytes).
        PDF bytes -> Q&A Chunks -> Embeddings -> Pinecone
        """
        start_time = time.time()

        try:
            document_id = str(uuid.uuid4())
            logger.info(f"Processing document: {filename} (ID: {document_id})")

            # Step 1: Extract text
            full_text = self._extract_pdf_text(file_bytes)
            if not full_text:
                raise Exception("PDF contained no extractable text.")

            # Step 2: Create Q&A chunks
            logger.info("Parsing Q&A chunks...")
            chunks = self._create_qa_chunks(full_text)
            total_chunks = len(chunks)
            if total_chunks == 0:
                raise Exception("No Q&A pairs found in the PDF. Ensure the format uses 'Q:' and 'A:' prefixes.")

            # Step 3: Generate embeddings
            logger.info("Generating embeddings...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self._generate_embeddings(chunk_texts)

            # Step 4: Upload to Pinecone
            logger.info("Uploading to Pinecone...")
            upload_result = self._upload_to_pinecone(chunks, embeddings, document_id, filename)

            processing_time = time.time() - start_time
            total_tokens = sum(chunk['token_count'] for chunk in chunks)

            result = {
                'success': True,
                'document_id': document_id,
                'filename': filename,
                'processing_time_seconds': round(processing_time, 2),
                'total_qa_pairs': total_chunks,
                'total_tokens': total_tokens,
                'pinecone_vectors_uploaded': upload_result['vectors_uploaded'],
                'created_at': upload_result['timestamp'],
                'metadata': {
                    'embedding_model': self.embedding_model,
                    'index_name': self.index_name,
                    'chunking_strategy': 'qa_pair',
                }
            }

            logger.info(f"Document processed in {processing_time:.2f}s - {total_chunks} Q&A pairs uploaded.")
            return result

        except Exception as e:
            logger.error(f"FAILED to process document {filename}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'document_id': None,
                'processing_time_seconds': round(time.time() - start_time, 2)
            }


    def add_to_existing_collection(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Add a document to the existing Pinecone collection without removing anything."""
        logger.info(f"Adding document '{filename}' to existing collection...")

        try:
            stats = self.index.describe_index_stats()
            initial_vector_count = stats['total_vector_count']
            logger.info(f"Collection has {initial_vector_count} vectors before adding.")

            result = self.process_document(file_bytes, filename)

            if result['success']:
                new_stats = self.index.describe_index_stats()
                result['collection_info'] = {
                    'total_vectors_before': initial_vector_count,
                    'total_vectors_after': new_stats['total_vector_count'],
                    'vectors_added': result.get('pinecone_vectors_uploaded', 0)
                }
                logger.info(f"Document added! Collection now has {new_stats['total_vector_count']} total vectors.")
            else:
                logger.error(f"Failed to add document '{filename}': {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"FAILED to add document {filename} to collection: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


    def replace_specific_document_vectors(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Replace all vectors for a specific document (delete old, upload new)."""
        logger.info(f"Replacing vectors for document: {filename}")

        try:
            self.index.delete(filter={"filename": {"$eq": filename}})
            logger.info(f"Deleted existing vectors for {filename}.")

            result = self.process_document(file_bytes, filename)

            if result.get('success'):
                result['document_replacement_info'] = {
                    'new_vectors_uploaded': result.get('pinecone_vectors_uploaded', 0),
                    'replacement_completed': True
                }
                logger.info(f"Replacement complete for {filename}.")
            else:
                logger.error(f"Replacement failed for {filename}: {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"Failed to replace vectors for {filename}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'document_replacement_info': {
                    'new_vectors_uploaded': 0,
                    'replacement_completed': False
                }
            }


    def reset_vector_database(self) -> Dict[str, Any]:
        """
        Empty Entire Database
        
        Deletes ALL existing documents.
        Use with caution - this wipes everything!
            
        Returns:
            Dict with processing results
        """
        logger.info(f"Deleting entire database")
        
        try:
            # Get current stats before deleting
            initial_stats = self.index.describe_index_stats()
            initial_count = initial_stats['total_vector_count']
            
            logger.warning(f"Deleting {initial_count} existing vectors...")
            
            # Delete all existing vectors
            self.index.delete(delete_all=True)
            
            logger.info("Database cleared!")
                       
            return {
                'success':True,
                'vectors_deleted': initial_count,
                'reset_completed': True
            }
            
        except Exception as e:
            logger.error(f"FAILED to rest database: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'database_replacement_info': {
                    'vectors_deleted': initial_count, # Report how many were deleted before fail
                    'new_vectors_uploaded': 0,
                    'replacement_completed': False
                }
            }
    

    def _generate_sub_queries(self, question: str) -> List[str]:
        """
        Use Claude to silently break a user question into 2 focused sub-queries
        for better retrieval coverage. Never exposed to the user.
        """
        t0 = time.time()
        try:
            prompt = (
                "Given the following user question, generate exactly 2 focused sub-queries "
                "that would help retrieve relevant FAQ entries from a knowledge base about "
                "FIRS e-Invoicing and Qucoon/Qorpy.\n\n"
                "The sub-queries should:\n"
                "- Cover different aspects of the question\n"
                "- Be specific enough to match FAQ entries\n"
                "- Be phrased as questions or search terms\n\n"
                f"User question: {question}\n\n"
                'Return ONLY a JSON array of 2 strings, nothing else. Example:\n'
                '["sub-query 1 text", "sub-query 2 text"]'
            )

            request_body = json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 200}
            })

            logger.info("[TIMING] Calling Bedrock for sub-query generation...")
            response = self.bedrock.invoke_model(
                modelId=self.fast_model,
                body=request_body,
                contentType='application/json'
            )

            result = json.loads(response['body'].read())
            text = result['output']['message']['content'][0]['text'].strip()
            # Strip markdown code fences if present (e.g. ```json ... ```)
            json_match = re.search(r'\[.*?\]', text, re.DOTALL)
            if json_match:
                text = json_match.group(0)
            sub_queries = json.loads(text)
            if isinstance(sub_queries, list) and len(sub_queries) >= 2:
                logger.info(f"[TIMING] Sub-query generation done in {time.time()-t0:.2f}s")
                return sub_queries[:2]

        except Exception as e:
            logger.warning(f"[TIMING] Sub-query generation failed after {time.time()-t0:.2f}s: {e}")

        # Fallback: use original question for both slots
        return [question, question]

    def ask_questions(self, question: str) -> Dict[str, Any]:
        """
        Ask a question with RAG retrieval using sub-query decomposition.
        """
        top_k = 5
        start_time = time.time()

        try:
            logger.info(f"[TIMING] ========== START ask_questions ==========")
            logger.info(f"[TIMING] Question: {question[:100]}")

            # Step 1: Sub-query generation
            t1 = time.time()
            sub_queries = self._generate_sub_queries(question)
            logger.info(f"[TIMING] Step 1 - Sub-query generation: {time.time()-t1:.2f}s")

            # Step 2: Embeddings + Pinecone retrieval — run both sub-queries in parallel
            def retrieve(idx: int, sq: str):
                t2 = time.time()
                embedding = self._generate_embeddings([sq])[0]
                logger.info(f"[TIMING] Step 2.{idx+1}a - Embedding sub-query {idx+1}: {time.time()-t2:.2f}s")
                t3 = time.time()
                results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
                logger.info(f"[TIMING] Step 2.{idx+1}b - Pinecone query {idx+1}: {time.time()-t3:.2f}s | hits: {len(results['matches'])}")
                return results['matches']

            t_retrieval = time.time()
            all_matches: Dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(retrieve, idx, sq): idx for idx, sq in enumerate(sub_queries)}
                for future in as_completed(futures):
                    for match in future.result():
                        if match['id'] not in all_matches:
                            all_matches[match['id']] = match

            unique_matches = sorted(all_matches.values(), key=lambda m: m['score'], reverse=True)
            logger.info(f"[TIMING] Step 2 total - Parallel retrieval complete in {time.time()-t_retrieval:.2f}s | {len(unique_matches)} unique chunks")

            # Step 3: Build context and sources
            context_chunks = []
            sources = []
            for match in unique_matches:
                metadata = match['metadata']
                context_chunks.append(metadata['text'])
                sources.append({
                    'document_id': metadata.get('document_id', ''),
                    'filename': metadata.get('filename', ''),
                    'category': metadata.get('category', ''),
                    'section': metadata.get('section', ''),
                    'relevance_score': round(match['score'], 3),
                    'chunk_index': metadata.get('chunk_index', 0),
                })

            context_text = "\n\n".join(context_chunks)

            # Step 4: Answer synthesis
            prompt = f"""You are the Qorpy FAQ assistant — an expert on FIRS e-Invoicing, Qucoon, and the Qorpy platform. Answer the user's question clearly and directly using only the context provided.

Context:
{context_text}

User's Question: {question}

Formatting rules (follow strictly):
- Write in plain, confident prose — no filler phrases like "Great question!" or "Based on the context"
- Keep answers short and scannable. Use bullet points or numbered lists when listing multiple items
- Use **bold** for key terms, names, or important values
- If the answer has a single clear fact, give it in 1-2 sentences — do not pad
- If information is truly missing from the context, say so in one line: "I don't have details on that — contact support@qucoon.com"
- Never start your reply with "I" as the first word
- Do not repeat the question back to the user

Answer:"""

            request_body = json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 800}
            })

            t4 = time.time()
            logger.info("[TIMING] Step 4 - Calling Bedrock for answer synthesis...")
            response = self.bedrock.invoke_model(
                modelId=self.chat_model,
                body=request_body,
                contentType='application/json'
            )

            result = json.loads(response['body'].read())
            answer = result['output']['message']['content'][0]['text']
            logger.info(f"[TIMING] Step 4 - Answer synthesis done: {time.time()-t4:.2f}s")

            total_time = time.time() - start_time
            logger.info(f"[TIMING] ========== TOTAL: {total_time:.2f}s ==========")

            return {
                'success': True,
                'answer': answer,
                'sources': sources,
                'question': question,
                'query_time_seconds': round(total_time, 2),
                'chunks_retrieved': len(context_chunks),
                'metadata': {
                    'embedding_model': self.embedding_model,
                    'chat_model': self.chat_model,
                    'top_k_used': top_k,
                    'sub_queries_used': 2,
                }
            }

        except Exception as e:
            logger.error(f"FAILED to answer question '{question[:50]}...': {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'answer': None,
                'sources': [],
                'query_time_seconds': round(time.time() - start_time, 2)
            }
    

    def ask_questions_stream(self, question: str):
        """
        Ask a question with RAG retrieval, streaming the answer token by token.
        Sub-query decomposition + retrieval happen upfront, then the answer streams.
        Yields raw text chunks from Bedrock's streaming API.
        """
        top_k = 5
        start_time = time.time()

        try:
            logger.info(f"[TIMING] ========== START ask_questions_stream ==========")

            # Step 1: Sub-query generation
            t1 = time.time()
            sub_queries = self._generate_sub_queries(question)
            logger.info(f"[TIMING] Step 1 - Sub-query generation: {time.time()-t1:.2f}s")

            # Step 2: Parallel retrieval
            def retrieve(idx: int, sq: str):
                embedding = self._generate_embeddings([sq])[0]
                results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
                return results['matches']

            t_retrieval = time.time()
            all_matches: Dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {executor.submit(retrieve, idx, sq): idx for idx, sq in enumerate(sub_queries)}
                for future in as_completed(futures):
                    for match in future.result():
                        if match['id'] not in all_matches:
                            all_matches[match['id']] = match

            unique_matches = sorted(all_matches.values(), key=lambda m: m['score'], reverse=True)
            logger.info(f"[TIMING] Step 2 - Parallel retrieval: {time.time()-t_retrieval:.2f}s | {len(unique_matches)} unique chunks")

            # Step 3: Build context
            context_text = "\n\n".join(m['metadata']['text'] for m in unique_matches)

            # Step 4: Stream answer synthesis
            prompt = f"""You are the Qorpy FAQ assistant — an expert on FIRS e-Invoicing, Qucoon, and the Qorpy platform. Answer the user's question clearly and directly using only the context provided.

Context:
{context_text}

User's Question: {question}

Formatting rules (follow strictly):
- Write in plain, confident prose — no filler phrases like "Great question!" or "Based on the context"
- Keep answers short and scannable. Use bullet points or numbered lists when listing multiple items
- Use **bold** for key terms, names, or important values
- If the answer has a single clear fact, give it in 1-2 sentences — do not pad
- If information is truly missing from the context, say so in one line: "I don't have details on that — contact support@qucoon.com"
- Never start your reply with "I" as the first word
- Do not repeat the question back to the user

Answer:"""

            request_body = json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 800}
            })

            t4 = time.time()
            logger.info("[TIMING] Step 4 - Starting streaming synthesis...")
            response = self.bedrock.invoke_model_with_response_stream(
                modelId=self.chat_model,
                body=request_body,
                contentType='application/json'
            )

            stream = response.get('body')
            if stream:
                for event in stream:
                    chunk = event.get('chunk')
                    if chunk:
                        chunk_data = json.loads(chunk.get('bytes').decode())
                        # Nova Lite streaming format
                        if 'contentBlockDelta' in chunk_data:
                            delta = chunk_data['contentBlockDelta'].get('delta', {})
                            if 'text' in delta:
                                yield delta['text']

            logger.info(f"[TIMING] Step 4 - Streaming done: {time.time()-t4:.2f}s")
            logger.info(f"[TIMING] ========== STREAM TOTAL: {time.time()-start_time:.2f}s ==========")

        except Exception as e:
            logger.error(f"FAILED to stream answer for '{question[:50]}': {e}", exc_info=True)
            yield f"\n\n⚠️ Something went wrong. Please try again."

    # =================
    # UTILITY FUNCTIONS
    # =================
    def get_database_stats(self) -> Dict[str, Any]:
        """Get current database statistics"""
        try:
            # Get stats directly from Pinecone index
            stats = self.index.describe_index_stats()
            logger.info(f"Retrieved DB stats: {stats}")
            return {
                'total_vectors': stats['total_vector_count'],
                'index_fullness': stats.get('index_fullness', 0), # Serverless may not have this
                'dimension': stats.get('dimension', 512), # Get dimension if available
                'index_name': self.index_name
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}", exc_info=True)
            return {'error': str(e)}
    
    def list_all_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database with metadata"""
        try:
            logger.info("Listing all documents... (uses dummy query)")
            # Query with a dummy vector to get a sample of vectors
            # This is a workaround as Pinecone doesn't have a "list all" metadata API
            sample_results = self.index.query(
                vector=[0.0] * 512,  # Dummy vector
                top_k=1000,  # Get many results to find all documents
                include_metadata=True
            )
            
            # Group by document_id to aggregate document info
            documents = {}
            for match in sample_results['matches']:
                metadata = match['metadata']
                doc_id = metadata.get('document_id', 'unknown')
                
                # If this is the first time seeing this doc_id, initialize it
                if doc_id not in documents:
                    documents[doc_id] = {
                        'document_id': doc_id,
                        'filename': metadata.get('filename', 'unknown'),
                        'created_at': metadata.get('created_at', 'unknown'),
                        'chunk_count': 0
                    }
                # Increment the chunk count for this document
                documents[doc_id]['chunk_count'] += 1
            
            logger.info(f"Found {len(documents)} unique documents.")
            # Return the aggregated list of documents
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Failed to list all documents: {e}", exc_info=True)
            return [] # Return empty list on failure