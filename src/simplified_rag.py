"""
Simplified RAG System - Google Gemini Integration
==================================================
1. Process uploaded PDF (PDF bytes -> Recursive Chunks -> Gemini Embeddings -> Pinecone)
2. Add document to existing collection
3. Replace vectors for a specific document
4. Ask questions with RAG retrieval (sub-query decomposition via Gemini)

Direct PDF upload via FastAPI - no S3 dependency.
Chunking strategy: Recursive with 1200-token target and 20% overlap (semantic-aware).
Embedding model: Google Gemini Embedding 2 (1536-dimensional).
LLM: Google Gemini 2.5 Flash for answer synthesis.
"""

import os
import re
import uuid
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
import tiktoken
from io import BytesIO
from pinecone import Pinecone, ServerlessSpec
import pdfplumber
from fastapi import HTTPException
import pathlib
import yaml
from src.chat_engine import GeminiChatClient
from src.utils import ConversationMemory, load_project_env, read_env_value

logger = logging.getLogger(__name__)

# ===========================
# CONSTANTS
# ===========================
CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_PCT = 0.20
CHUNK_OVERLAP_TOKENS = int(CHUNK_TARGET_TOKENS * CHUNK_OVERLAP_PCT)
EMBEDDING_DIMENSION = 1536
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-001"
GEMINI_TEXT_MODEL = "models/gemini-2.5-flash"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Load .env once at module level via utils
_env_file = load_project_env()


class SimplifiedRAG:
    """Simplified RAG system using Google Gemini API and Pinecone for retrieval."""
        
    def __init__(self):
        """Initialize the RAG system with Google Gemini API and Pinecone."""
        try:
            # Google Gemini API setup
            self.gemini_api_key = os.getenv('GEMINI_API_KEY')
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            logger.info("[CONFIG] Google Gemini API initialized")

            # Pinecone setup
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            self.index_name = read_env_value("PINECONE_INDEX_NAME", _env_file) or os.getenv('PINECONE_INDEX_NAME')
            logger.info(f"[CONFIG] Pinecone index: {self.index_name!r}")
            
            # Create index if it doesn't exist (1536-dimensional for Gemini Embedding 2)
            if self.index_name not in [index.name for index in pc.list_indexes()]:
                logger.info(f"Creating new Pinecone index: {self.index_name} with dimension {EMBEDDING_DIMENSION}")
                pc.create_index(
                    name=self.index_name,
                    dimension=EMBEDDING_DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            
            # Connect to the Pinecone index
            self.index = pc.Index(self.index_name)
            
            # Model configurations
            self.embedding_model = GEMINI_EMBEDDING_MODEL
            self.text_model = GEMINI_TEXT_MODEL
            self.chat_client = GeminiChatClient(
                api_key=self.gemini_api_key,
                api_base=GEMINI_API_BASE,
                text_model=self.text_model,
            )
            logger.info(f"[CONFIG] Models: embedding={self.embedding_model}, text={self.text_model}")

            # Load prompt templates from YAML file
            _prompt_file = pathlib.Path(__file__).resolve().parent / "prompts.yaml"
            if not _prompt_file.exists():
                raise FileNotFoundError(f"Prompts file not found: {_prompt_file}")
            
            with open(_prompt_file, "r") as f:
                prompts_config = yaml.safe_load(f)

            self.system_prompt = prompts_config.get("system_prompt", "").strip()
            self.user_template = prompts_config.get("user_template", "").strip()
            self.sub_query_template = prompts_config.get("sub_query_template", "").strip()
            logger.info("[CONFIG] Loaded prompt templates from src/prompts.yaml")

            # Redis conversation memory (Upstash)
            _redis_url = read_env_value("REDIS_URL", _env_file) or os.getenv("REDIS_URL")
            if _redis_url:
                self.memory = ConversationMemory(_redis_url)
            else:
                self.memory = None
                logger.warning("[CONFIG] REDIS_URL not set — conversation history disabled")

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
        """Extract all text from a PDF using pdfplumber (handles complex layouts)."""
        try:
            full_text = ""
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                page_count = len(pdf.pages)
            logger.info(f"Extracted text from {page_count} PDF pages ({len(full_text)} chars).")
            return full_text.strip()
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}", exc_info=True)
            raise Exception(f"Failed to extract PDF text: {str(e)}")

    def _extract_text(self, file_bytes: bytes, filename: str) -> str:
        """Extract text from PDF or TXT file."""
        if filename.lower().endswith('.txt'):
            try:
                text = file_bytes.decode('utf-8')
                logger.info(f"Extracted {len(text)} chars from TXT file.")
                return text.strip()
            except Exception as e:
                raise Exception(f"Failed to read TXT file: {str(e)}")
        return self._extract_pdf_text(file_bytes)

    def _create_qa_chunks(self, full_text: str) -> List[Dict[str, Any]]:
        """
        Recursive text chunker — works on any prose document (legal acts, FAQs, reports).

        Strategy:
        1. Try to split on section boundaries first (preserves legal structure)
        2. Fall back to paragraph breaks, then sentences, then words
        3. Each chunk targets CHUNK_TARGET_TOKENS (1200) with CHUNK_OVERLAP_TOKENS (240)
           carried forward from the tail of the previous chunk
        """
        # Separators tried in order — most semantic to least
        SEPARATORS = [
            r'\n(?=\d+\.\s)',          # numbered section: "6. Unlawful access"
            r'\n(?=PART\s+[IVXLC]+)',  # PART I, PART II …
            r'\n\n+',                  # paragraph break
            r'\n',                     # single newline
            r'(?<=[.!?])\s+',          # sentence boundary
        ]

        def _split(text: str) -> List[str]:
            """Split text using the first separator that produces >1 piece."""
            for sep in SEPARATORS:
                parts = [p.strip() for p in re.split(sep, text) if p.strip()]
                if len(parts) > 1:
                    return parts
            # Last resort: split on whitespace
            return text.split()

        def _token_len(text: str) -> int:
            return len(self.tokenizer.encode(text))

        # --- detect section heading for metadata ---
        section_pattern = re.compile(
            r'^(?:PART\s+[IVXLC]+.*|\d+\.\s+.{3,80})$', re.MULTILINE
        )

        def _section_for(text: str) -> str:
            m = section_pattern.search(text)
            return m.group(0).strip()[:80] if m else "General"

        # --- recursive merge: combine small pieces into target-sized chunks ---
        def _merge(pieces: List[str]) -> List[str]:
            """
            Walk through pieces, accumulating tokens until we hit the target.
            When a single piece already exceeds the target, recurse into it.
            """
            result: List[str] = []
            current = ""

            for piece in pieces:
                if _token_len(piece) > CHUNK_TARGET_TOKENS:
                    # Piece is too big on its own — recurse
                    if current:
                        result.append(current)
                        current = ""
                    result.extend(_merge(_split(piece)))
                    continue

                candidate = (current + "\n" + piece).strip() if current else piece
                if _token_len(candidate) <= CHUNK_TARGET_TOKENS:
                    current = candidate
                else:
                    if current:
                        result.append(current)
                    current = piece

            if current:
                result.append(current)
            return result

        raw_pieces = _split(full_text)
        merged = _merge(raw_pieces)

        # --- apply 20% overlap between adjacent chunks ---
        chunks: List[Dict[str, Any]] = []
        for idx, text in enumerate(merged):
            # carry the last CHUNK_OVERLAP_TOKENS tokens from previous chunk
            if idx > 0:
                prev_words = merged[idx - 1].split()
                overlap = ""
                for word in reversed(prev_words):
                    candidate = word + " " + overlap
                    if _token_len(candidate) > CHUNK_OVERLAP_TOKENS:
                        break
                    overlap = candidate
                if overlap.strip():
                    combined = (overlap.strip() + "\n" + text).strip()
                    # if adding overlap pushes over the limit, trim the tail of text
                    if _token_len(combined) > CHUNK_TARGET_TOKENS:
                        words = combined.split()
                        while words and _token_len(" ".join(words)) > CHUNK_TARGET_TOKENS:
                            words.pop()
                        combined = " ".join(words)
                    text = combined

            token_count = _token_len(text)
            chunks.append({
                'text': text,
                'section': _section_for(text),
                'category': 'General',
                'token_count': token_count,
                'char_count': len(text),
                'chunk_index': idx,
                # keep these keys so Pinecone metadata stays consistent
                'question': '',
                'answer': '',
            })

        logger.info(
            f"Created {len(chunks)} recursive chunks "
            f"(target={CHUNK_TARGET_TOKENS} tokens, overlap={CHUNK_OVERLAP_PCT*100:.0f}%)"
        )
        return chunks


    def _embed_single(self, text: str, idx: int, total: int) -> List[float]:
        """Embed one text with 3 retries. Raises on all failures."""
        url = f"{GEMINI_API_BASE}/{GEMINI_EMBEDDING_MODEL}:embedContent?key={self.gemini_api_key}"
        payload = {
            "model": GEMINI_EMBEDDING_MODEL,
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": EMBEDDING_DIMENSION,
        }
        for attempt in range(1, 4):
            try:
                response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
                if response.status_code == 429:
                    wait = 5 * attempt
                    logger.warning(f"[EMBEDDING {idx}/{total}] Rate limited — waiting {wait}s (attempt {attempt})")
                    time.sleep(wait)
                    continue
                if response.status_code != 200:
                    raise Exception(f"Gemini API error {response.status_code}: {response.text[:200]}")
                embedding = response.json().get('embedding', {}).get('values', [])
                if len(embedding) != EMBEDDING_DIMENSION:
                    raise Exception(f"Wrong dimension: {len(embedding)}")
                return embedding
            except requests.exceptions.Timeout:
                logger.warning(f"[EMBEDDING {idx}/{total}] Timeout (attempt {attempt})")
                if attempt == 3:
                    raise Exception(f"Embedding timed out after 3 attempts")
                time.sleep(3 * attempt)
        raise Exception(f"Embedding failed after 3 attempts")

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed all texts, raising immediately if any chunk fails."""
        embeddings = []
        for i, text in enumerate(texts):
            embedding = self._embed_single(text, i + 1, len(texts))
            embeddings.append(embedding)
            logger.info(f"[EMBEDDING {i+1}/{len(texts)}] OK ({EMBEDDING_DIMENSION}D)")
        logger.info(f"Generated {len(embeddings)} embeddings ({EMBEDDING_DIMENSION}D)")
        return embeddings
    
    def _upload_to_pinecone(self, chunks: List[Dict], embeddings: List[List[float]], 
                            document_id: str, filename: str, namespace: str = "") -> Dict[str, Any]:
        """
        Upload chunks and 1536-dimensional embeddings to Pinecone with rich metadata.
        
        Args:
            chunks: List of chunk dictionaries with text, section, category, etc.
            embeddings: List of 1536-dimensional embedding vectors
            document_id: Unique identifier for the document
            filename: Original filename
            namespace: Pinecone namespace for multi-tenancy
            
        Returns:
            Dictionary with upload statistics
            
        Raises:
            Exception: If Pinecone upload fails
        """
        try:
            vectors = []
            timestamp = datetime.now().isoformat()
            
            # Prepare vectors for upload with metadata
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_id = f"{document_id}_chunk_{i}"
                
                # Rich metadata for retrieval and context
                metadata = {
                    'document_id': document_id,
                    'filename': filename,
                    'section': chunk.get('section', 'General'),
                    'category': chunk.get('category', 'General'),
                    'question': chunk.get('question', ''),
                    'answer': chunk.get('answer', ''),
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['text'],
                    'token_count': chunk['token_count'],
                    'char_count': chunk['char_count'],
                    'created_at': timestamp,
                    'chunk_type': 'qa_recursive',  # New type to indicate recursive chunking
                    'embedding_dimension': EMBEDDING_DIMENSION,
                }
                
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upload in batches for efficiency
            batch_size = 100
            total_uploaded = 0
            
            logger.info(f"Uploading {len(vectors)} vectors in batches of {batch_size}...")
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                try:
                    self.index.upsert(vectors=batch, namespace=namespace)
                    total_uploaded += len(batch)
                except Exception as e:
                    logger.error(f"Batch upload failed (vectors {i}-{i+len(batch)}): {e}")
                    raise
            
            logger.info(f"Successfully uploaded {total_uploaded} vectors to Pinecone (dim={EMBEDDING_DIMENSION})")
            
            return {
                'vectors_uploaded': total_uploaded,
                'document_id': document_id,
                'timestamp': timestamp
            }
        except Exception as e:
            error_msg = f"Pinecone upload failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg)
    
    # =========================
    # CORE FUNCTIONS
    # =========================

    def process_document(self, file_bytes: bytes, filename: str, namespace: str = "") -> Dict[str, Any]:
        """
        Complete PDF Processing Pipeline (recursive chunking + Gemini embeddings + Pinecone upload).
        
        PDF bytes → Text Extraction → Recursive Chunks → Gemini Embeddings → Pinecone
        
        Args:
            file_bytes: Raw PDF file bytes
            filename: Original filename for metadata
            namespace: Pinecone namespace for multi-tenancy
            
        Returns:
            Dictionary with processing statistics and results
        """
        start_time = time.time()

        try:
            document_id = str(uuid.uuid4())
            logger.info(f"Processing document: {filename} (ID: {document_id})")

            # Step 1: Extract text from PDF or TXT
            full_text = self._extract_text(file_bytes, filename)
            if not full_text:
                raise Exception("PDF contained no extractable text.")

            # Step 2: Create recursive chunks with overlap
            logger.info(f"Creating recursive chunks (target={CHUNK_TARGET_TOKENS} tokens, overlap={CHUNK_OVERLAP_PCT*100}%)")
            chunks = self._create_qa_chunks(full_text)
            total_chunks = len(chunks)
            if total_chunks == 0:
                raise Exception("No text content found in the PDF. Ensure the file contains extractable text.")

            # Step 3: Generate Gemini embeddings (1536-dimensional)
            logger.info("Generating Gemini embeddings...")
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self._generate_embeddings(chunk_texts)

            # Step 4: Upload to Pinecone
            logger.info("Uploading to Pinecone...")
            upload_result = self._upload_to_pinecone(chunks, embeddings, document_id, filename, namespace=namespace)

            processing_time = time.time() - start_time
            total_tokens = sum(chunk['token_count'] for chunk in chunks)

            result = {
                'success': True,
                'document_id': document_id,
                'filename': filename,
                'processing_time_seconds': round(processing_time, 2),
                'total_chunks': total_chunks,
                'total_tokens': total_tokens,
                'pinecone_vectors_uploaded': upload_result['vectors_uploaded'],
                'created_at': upload_result['timestamp'],
                'metadata': {
                    'embedding_model': self.embedding_model,
                    'embedding_dimension': EMBEDDING_DIMENSION,
                    'chunking_strategy': 'recursive_with_overlap',
                    'chunk_target_tokens': CHUNK_TARGET_TOKENS,
                    'chunk_overlap_pct': CHUNK_OVERLAP_PCT,
                    'index_name': self.index_name,
                }
            }

            logger.info(f"Document processed in {processing_time:.2f}s - {total_chunks} recursive chunks uploaded.")
            return result

        except Exception as e:
            logger.error(f"FAILED to process document {filename}: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'document_id': None,
                'processing_time_seconds': round(time.time() - start_time, 2)
            }


    def add_to_existing_collection(self, file_bytes: bytes, filename: str, namespace: str = "") -> Dict[str, Any]:
        """Add a document to the existing Pinecone collection without removing anything."""
        logger.info(f"Adding document '{filename}' to existing collection...")

        try:
            stats = self.index.describe_index_stats()
            ns_stats = stats.get('namespaces', {}).get(namespace, {})
            initial_vector_count = ns_stats.get('vector_count', 0)
            logger.info(f"Namespace '{namespace}' has {initial_vector_count} vectors before adding.")

            result = self.process_document(file_bytes, filename, namespace=namespace)

            if result['success']:
                new_stats = self.index.describe_index_stats()
                new_ns_stats = new_stats.get('namespaces', {}).get(namespace, {})
                result['collection_info'] = {
                    'total_vectors_before': initial_vector_count,
                    'total_vectors_after': new_ns_stats.get('vector_count', 0),
                    'vectors_added': result.get('pinecone_vectors_uploaded', 0)
                }
                logger.info(f"Document added! Collection now has {new_stats['total_vector_count']} total vectors.")
            else:
                logger.error(f"Failed to add document '{filename}': {result.get('error')}")

            return result

        except Exception as e:
            logger.error(f"FAILED to add document {filename} to collection: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


    def replace_specific_document_vectors(self, file_bytes: bytes, filename: str, namespace: str = "") -> Dict[str, Any]:
        """Replace all vectors for a specific document (delete old, upload new)."""
        logger.info(f"Replacing vectors for document: {filename}")

        try:
            self.index.delete(filter={"filename": {"$eq": filename}}, namespace=namespace)
            logger.info(f"Deleted existing vectors for {filename}.")

            result = self.process_document(file_bytes, filename, namespace=namespace)

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


    def reset_vector_database(self, namespace: str = "") -> Dict[str, Any]:
        """
        Delete all vectors in a specific namespace (tenant).
        Does NOT wipe other tenants' data.
            
        Returns:
            Dict with processing results
        """
        logger.info(f"Deleting all vectors in namespace '{namespace}'")
        
        try:
            # Get current stats before deleting
            initial_stats = self.index.describe_index_stats()
            ns_stats = initial_stats.get('namespaces', {}).get(namespace, {})
            initial_count = ns_stats.get('vector_count', 0)
            
            logger.warning(f"Deleting {initial_count} existing vectors in namespace '{namespace}'...")
            
            # Delete all vectors in the specified namespace only
            try:
                self.index.delete(delete_all=True, namespace=namespace)
            except Exception as e:
                # If Pinecone throws "Namespace not found", it's already empty!
                if "Namespace not found" in str(e) or "404" in str(e):
                    logger.info(f"Namespace {namespace} is already empty (404).")
                else:
                    raise e
            
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
    
    def ask_questions(self, question: str, session_id: Optional[str] = None, namespace: str = "") -> Dict[str, Any]:
        """
        Ask a question with RAG retrieval using Google Gemini API and sub-query decomposition.
        
        Strategy:
        1. Sub-query decomposition: Split question into 1-5 semantically distinct queries
        2. Parallel retrieval: Embed each sub-query and fetch top-2 chunks per query via cosine similarity
        3. Context merging: De-duplicate and rank retrieved chunks by relevance
        4. Answer synthesis: Use Gemini 2.5 Flash with system prompt + history + context + 5 checkpoints
        5. Redis caching: Store Q&A pairs for future context (last 5 exchanges, 30-min TTL)
        
        Args:
            question: User's original question
            session_id: Optional session ID for conversation history
            namespace: Pinecone namespace for multi-tenancy
            
        Returns:
            Dictionary with answer, sources, timing, and metadata
        """
        top_k = 4  # Retrieve top-4 chunks per sub-query (cosine similarity metric)
        start_time = time.time()

        try:
            logger.info(f"[TIMING] ========== START ask_questions ==========")
            logger.info(f"[TIMING] Question: {question[:100]}")

            # Step 1: Intent classification + sub-query generation (Gemini API)
            t1 = time.time()
            sub_queries = self.chat_client.generate_sub_queries(question, self.sub_query_template)
            logger.info(f"[TIMING] Step 1 - Sub-query generation: {time.time()-t1:.2f}s")
            logger.info(f"[SUB-QUERIES] {len(sub_queries)} query(s): {sub_queries}")

            # Conversational short-circuit — skip RAG, answer directly from Gemini
            if sub_queries == ["__conversational__"]:
                history = self.memory.get_history(session_id) if (self.memory and session_id) else "No previous conversation."
                conv_message = (
                    f"Conversation History (last 5 exchanges):\n{history}\n\n"
                    f"User's message: {question}\n\n"
                    f"Respond naturally and conversationally."
                )
                answer = self.chat_client.generate_text(
                    system_prompt=self.system_prompt,
                    user_message=conv_message,
                    max_tokens=200
                )
                if self.memory and session_id:
                    self.memory.save(session_id, question, answer)
                return {
                    'success': True,
                    'answer': answer,
                    'sources': [],
                    'question': question,
                    'query_time_seconds': round(time.time() - start_time, 2),
                    'chunks_retrieved': 0,
                    'sub_queries_used': 0,
                    'response_type': 'conversational'
                }

            # Step 2: Parallel Gemini embeddings + Pinecone retrieval
            def retrieve(idx: int, sq: str):
                t2 = time.time()
                embedding = self._generate_embeddings([sq])[0]
                logger.info(f"[TIMING] Step 2.{idx+1}a - Embedding sub-query {idx+1}: {time.time()-t2:.2f}s")
                t3 = time.time()
                results = self.index.query(vector=embedding, top_k=top_k, include_metadata=True, namespace=namespace)
                logger.info(f"[TIMING] Step 2.{idx+1}b - Pinecone query {idx+1}: {time.time()-t3:.2f}s | hits: {len(results['matches'])}")
                return results['matches']

            t_retrieval = time.time()
            all_matches: Dict[str, Any] = {}
            with ThreadPoolExecutor(max_workers=len(sub_queries)) as executor:
                futures = {executor.submit(retrieve, idx, sq): idx for idx, sq in enumerate(sub_queries)}
                for future in as_completed(futures):
                    for match in future.result():
                        # Deduplicate by text content to avoid duplicate uploads showing as separate chunks
                        text_key = match['metadata'].get('text', '')[:100]
                        if text_key not in all_matches:
                            all_matches[text_key] = match

            unique_matches = sorted(all_matches.values(), key=lambda m: m['score'], reverse=True)
            logger.info(f"[TIMING] Step 2 total - Parallel retrieval in {time.time()-t_retrieval:.2f}s | {len(unique_matches)} unique chunks")
            for i, m in enumerate(unique_matches[:5]):  # Log top 5
                logger.info(f"[RETRIEVED {i+1}] score={m['score']:.3f} | cat={m['metadata'].get('category','')} | sec={m['metadata'].get('section','')}")

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

            # Fallback: no context retrieved
            if not unique_matches:
                logger.warning("[RAG] No chunks retrieved — returning graceful failure")
                return {
                    'success': False,
                    'error': "No relevant context found in knowledge base",
                    'answer': "I don't have information on that topic. Please try rephrasing your question.",
                    'sources': [],
                    'question': question,
                    'query_time_seconds': round(time.time() - start_time, 2),
                    'chunks_retrieved': 0,
                    'sub_queries_used': len(sub_queries),
                }

            # Step 4: Answer synthesis via Gemini 2.5 Flash
            history = self.memory.get_history(session_id) if (self.memory and session_id) else "No previous conversation."
            if history != "No previous conversation.":
                logger.info(f"[REDIS] Injecting {len(context_chunks)} context chunks + history for session {session_id}")

            user_message = self.user_template.format(
                history=history,
                context=context_text,
                question=question,
            )

            t4 = time.time()
            logger.info("[TIMING] Step 4 - Calling Gemini for answer synthesis...")
            answer = self.chat_client.generate_text(
                system_prompt=self.system_prompt,
                user_message=user_message,
                max_tokens=1000
            )
            logger.info(f"[TIMING] Step 4 - Answer synthesis done: {time.time()-t4:.2f}s")

            # Save Q&A to Redis history
            if self.memory and session_id:
                self.memory.save(session_id, question, answer)

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
                    'text_model': self.text_model,
                    'embedding_dimension': EMBEDDING_DIMENSION,
                    'chunking_strategy': 'recursive_with_overlap',
                    'top_k_per_query': top_k,
                    'sub_queries_used': len(sub_queries),
                },
                'response_type': 'rag'
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
    
    def ask_questions_stream(self, question: str, session_id: str | None = None, namespace: str = ""):
        """Yield answer text word-by-word for SSE streaming (wraps ask_questions)."""
        result = self.ask_questions(question=question, session_id=session_id, namespace=namespace)
        answer = result.get("answer") or ""
        for word in answer.split(" "):
            yield word + " "

    # UTILITY FUNCTIONS
    # =================
    def get_database_stats(self, namespace: str | None = None) -> Dict[str, Any]:
        """
        Get database statistics for Pinecone index.
        
        Args:
            namespace: Optional namespace to get stats for specific tenant
            
        Returns:
            Dictionary with vector counts, dimensions, and namespace info
        """
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Retrieved DB stats: {stats}")

            if namespace is not None:
                ns_stats = stats.get('namespaces', {}).get(namespace, {})
                return {
                    'namespace': namespace,
                    'total_vectors': ns_stats.get('vector_count', 0),
                    'index_name': self.index_name,
                    'embedding_dimension': EMBEDDING_DIMENSION,
                }

            # No namespace specified — return global overview
            return {
                'total_vectors': stats['total_vector_count'],
                'index_fullness': stats.get('index_fullness', 0),
                'embedding_dimension': EMBEDDING_DIMENSION,
                'index_name': self.index_name,
                'namespaces': {ns: info for ns, info in stats.get('namespaces', {}).items()},
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}", exc_info=True)
            return {'error': str(e)}
    
    def list_all_documents(self, namespace: str = "") -> List[Dict[str, Any]]:
        """
        List all documents in a specific namespace.
        Uses a dummy zero-vector query to retrieve metadata without semantic matching.
        
        Args:
            namespace: Pinecone namespace to query
            
        Returns:
            List of documents with document_id, filename, created_at, chunk_count
        """
        try:
            logger.info(f"Listing documents in namespace '{namespace}'...")
            # Use a zero vector (1536 dimensions) to retrieve all chunks without filtering
            sample_results = self.index.query(
                vector=[0.0] * EMBEDDING_DIMENSION,
                top_k=1000,
                include_metadata=True,
                namespace=namespace,
            )
            
            # Aggregate chunks by document_id
            documents = {}
            for match in sample_results['matches']:
                metadata = match['metadata']
                doc_id = metadata.get('document_id', 'unknown')
                
                if doc_id not in documents:
                    documents[doc_id] = {
                        'document_id': doc_id,
                        'filename': metadata.get('filename', 'unknown'),
                        'created_at': metadata.get('created_at', 'unknown'),
                        'chunk_count': 0,
                        'embedding_dimension': EMBEDDING_DIMENSION,
                    }
                documents[doc_id]['chunk_count'] += 1
            
            logger.info(f"Found {len(documents)} unique documents in namespace '{namespace}'.")
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}", exc_info=True)
            return []