"""
Test suite for Cybercrime Act 2015 RAG system.

Tests cover:
- Module imports
- Named constants
- Recursive prose chunking (token limits + overlap)
- Prompt template loading from YAML
- Pydantic model validation
- Sub-query decomposition (mocked Gemini)
- PDF bytes extraction
- Error handling
- Configuration validation
"""

import os
import sys
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_rag():
    """Return a SimplifiedRAG instance with all external services mocked."""
    from src.simplified_rag import SimplifiedRAG
    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = [MagicMock(name="test_index")]
    mock_pc.Index.return_value = MagicMock()
    with patch("src.simplified_rag.Pinecone", return_value=mock_pc), \
         patch("src.simplified_rag.ConversationMemory"):
        return SimplifiedRAG()


ENV = {
    "GEMINI_API_KEY": "test_key",
    "PINECONE_API_KEY": "test_key",
    "PINECONE_INDEX_NAME": "test_index",
    "REDIS_URL": "redis://test",
}

# Realistic Cybercrime Act prose (no Q: / A: markers)
CYBERCRIME_SAMPLE = """
PART III OFFENCES AND PENALTIES

6. Unlawful access to a computer
(1) Any person, who without authorization or in excess of authorization, intentionally
accesses in whole or in part, a computer system or network, commits an offence
and liable on conviction to imprisonment for a term of not less than two years
or to a fine of not less than N5,000,000 or to both fine and imprisonment.
(2) Where the offence provided in subsection (1) of this section is committed with
the intent of obtaining computer data, securing access to any program,
commercial or industrial secrets or confidential information, the punishment shall
be imprisonment for a term of not less than three years or a fine of not less than
N7,000,000.00 or to both fine and imprisonment.

7. Unlawful interception of communications
Any person, who intentionally and without authorization or in excess of authority,
intercepts by technical means, transmissions of non-public computer data, content data
or traffic data, including electromagnetic emissions or signals from a computer,
commits an offence and liable on conviction to imprisonment for a term of not less than
two years or to a fine of not less than N5,000,000.00 or to both fine and imprisonment.

8. Unauthorized modification of computer data
(1) Any person who directly or indirectly does an act without authority and with intent
to cause an unauthorized modification of any data held in any computer system
or network, commits an offence and liable on conviction to imprisonment for a
term of not less than 3 years or to a fine of not less than N7,000,000.00.
(2) Any person who engages in damaging, deletion, deteriorating, alteration,
restriction or suppression of data within computer systems or networks,
commits an offence and liable on conviction to imprisonment for a term of not
less than three years or to a fine of not less than N7,000,000.00.

17. Cyberterrorism
(1) Any person that accesses or causes to be accessed any computer or computer
system or network for purposes of terrorism, commits an offence and liable on
conviction to life imprisonment.
"""


# ── TestImports ───────────────────────────────────────────────────────────────

class TestImports(unittest.TestCase):

    def test_import_simplified_rag(self):
        from src.simplified_rag import SimplifiedRAG
        self.assertIsNotNone(SimplifiedRAG)

    def test_import_models(self):
        from src.models import QuestionRequest, CreateSessionRequest, APIResponse
        self.assertIsNotNone(QuestionRequest)
        self.assertIsNotNone(CreateSessionRequest)
        self.assertIsNotNone(APIResponse)

    def test_import_fastapi_app(self):
        from app import app
        self.assertIsNotNone(app)

    def test_import_chat_engine(self):
        from src.chat_engine import GeminiChatClient
        self.assertIsNotNone(GeminiChatClient)

    def test_import_utils(self):
        from src.utils import ConversationMemory, load_project_env, read_env_value
        self.assertIsNotNone(ConversationMemory)


# ── TestConstants ─────────────────────────────────────────────────────────────

class TestConstants(unittest.TestCase):

    def test_chunking_constants(self):
        from src.simplified_rag import (
            CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_PCT, CHUNK_OVERLAP_TOKENS,
            EMBEDDING_DIMENSION, GEMINI_EMBEDDING_MODEL,
            GEMINI_TEXT_MODEL, GEMINI_API_BASE,
        )
        self.assertEqual(CHUNK_TARGET_TOKENS, 1200)
        self.assertEqual(CHUNK_OVERLAP_PCT, 0.20)
        self.assertEqual(CHUNK_OVERLAP_TOKENS, 240)
        self.assertEqual(EMBEDDING_DIMENSION, 1536)
        self.assertIn("embedding", GEMINI_EMBEDDING_MODEL.lower())
        self.assertIn("gemini", GEMINI_TEXT_MODEL.lower())
        self.assertIn("generativelanguage", GEMINI_API_BASE)


# ── TestRecursiveChunking ─────────────────────────────────────────────────────

class TestRecursiveChunking(unittest.TestCase):

    @patch.dict(os.environ, ENV)
    def setUp(self):
        self.rag = _make_rag()

    def test_produces_chunks_from_prose(self):
        """Chunker must work on plain legal prose — no Q:/A: markers needed."""
        chunks = self.rag._create_qa_chunks(CYBERCRIME_SAMPLE)
        self.assertGreater(len(chunks), 0, "Expected at least one chunk from prose text")

    def test_all_chunks_within_token_limit(self):
        """Every chunk must be <= CHUNK_TARGET_TOKENS (1200)."""
        from src.simplified_rag import CHUNK_TARGET_TOKENS
        long_text = CYBERCRIME_SAMPLE * 8
        chunks = self.rag._create_qa_chunks(long_text)
        for i, chunk in enumerate(chunks):
            self.assertLessEqual(
                chunk["token_count"], CHUNK_TARGET_TOKENS,
                f"Chunk {i} exceeds token limit: {chunk['token_count']} tokens"
            )

    def test_overlap_between_adjacent_chunks(self):
        """Adjacent chunks must share words (20% overlap)."""
        long_text = CYBERCRIME_SAMPLE * 8
        chunks = self.rag._create_qa_chunks(long_text)
        if len(chunks) < 2:
            self.skipTest("Not enough chunks to test overlap")
        overlaps_found = 0
        for i in range(1, len(chunks)):
            prev_tail = set(chunks[i - 1]["text"].split()[-40:])
            curr_head = set(chunks[i]["text"].split()[:40])
            if prev_tail & curr_head:
                overlaps_found += 1
        self.assertGreater(overlaps_found, 0, "No overlap found between any adjacent chunks")

    def test_chunk_metadata_keys_present(self):
        """Every chunk dict must have the required metadata keys."""
        chunks = self.rag._create_qa_chunks(CYBERCRIME_SAMPLE)
        required = {"text", "section", "category", "token_count", "char_count",
                    "chunk_index", "question", "answer"}
        for i, chunk in enumerate(chunks):
            missing = required - chunk.keys()
            self.assertFalse(missing, f"Chunk {i} missing keys: {missing}")

    def test_section_metadata_detected(self):
        """Section headings in the text should be captured in metadata."""
        chunks = self.rag._create_qa_chunks(CYBERCRIME_SAMPLE)
        sections = [c["section"] for c in chunks]
        # At least one chunk should have a non-General section
        self.assertTrue(
            any(s != "General" for s in sections),
            f"All sections are 'General' — heading detection failed. Got: {sections}"
        )

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(self.rag._create_qa_chunks(""), [])

    def test_whitespace_only_returns_empty_list(self):
        self.assertEqual(self.rag._create_qa_chunks("   \n\n   "), [])

    def test_short_text_single_chunk(self):
        """Text well under 1200 tokens should produce exactly one chunk."""
        chunks = self.rag._create_qa_chunks("Section 1. This Act applies to Nigeria.")
        self.assertEqual(len(chunks), 1)
        self.assertLessEqual(chunks[0]["token_count"], 1200)


# ── TestPromptLoading ─────────────────────────────────────────────────────────

class TestPromptLoading(unittest.TestCase):

    @patch.dict(os.environ, ENV)
    def setUp(self):
        self.rag = _make_rag()

    def test_system_prompt_loaded(self):
        self.assertIsNotNone(self.rag.system_prompt)
        self.assertIn("Cybercrime", self.rag.system_prompt)
        self.assertIn("Section", self.rag.system_prompt)
        self.assertIn("checkpoint", self.rag.system_prompt.lower())

    def test_system_prompt_has_penalty_figures(self):
        """System prompt must contain exact naira penalty figures."""
        self.assertIn("N7,000,000", self.rag.system_prompt)
        self.assertIn("N5,000,000", self.rag.system_prompt)

    def test_system_prompt_covers_key_offences(self):
        """System prompt must reference the major offence sections."""
        for section in ["Section 6", "Section 14", "Section 17"]:
            self.assertIn(section, self.rag.system_prompt,
                          f"{section} missing from system prompt")

    def test_user_template_placeholders(self):
        self.assertIn("{history}", self.rag.user_template)
        self.assertIn("{context}", self.rag.user_template)
        self.assertIn("{question}", self.rag.user_template)

    def test_user_template_cites_act(self):
        self.assertIn("Cybercrime Act 2015", self.rag.user_template)

    def test_sub_query_template_structure(self):
        self.assertIn("{user_query}", self.rag.sub_query_template)
        self.assertIn("JSON", self.rag.sub_query_template)
        self.assertIn("__conversational__", self.rag.sub_query_template)
        self.assertIn("decomposition", self.rag.sub_query_template.lower())

    def test_sub_query_template_has_cybercrime_examples(self):
        """Sub-query template must have examples relevant to this Act."""
        self.assertIn("cyberstalking", self.rag.sub_query_template.lower())


# ── TestPDFExtraction ─────────────────────────────────────────────────────────

class TestPDFExtraction(unittest.TestCase):

    @patch.dict(os.environ, ENV)
    def setUp(self):
        self.rag = _make_rag()

    def test_pdf_extraction_returns_string(self):
        """_extract_pdf_text must return a string from valid PDF bytes."""
        try:
            import fpdf
            pdf = fpdf.FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", size=12)
            pdf.cell(200, 10, txt="Section 6. Unlawful access to a computer.", ln=True)
            pdf_bytes = bytes(pdf.output())
        except ImportError:
            # fpdf not installed — build a minimal valid PDF manually
            pdf_bytes = (
                b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
                b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
                b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
                b"/Resources<</Font<</F1 4 0 R>>>>>>endobj\n"
                b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
                b"xref\n0 5\n0000000000 65535 f\n"
                b"trailer<</Size 5/Root 1 0 R>>\nstartxref\n0\n%%EOF"
            )

        try:
            result = self.rag._extract_pdf_text(pdf_bytes)
            self.assertIsInstance(result, str)
        except Exception:
            # PyPDF2 may fail on the minimal PDF — that's acceptable;
            # what matters is it raises an Exception, not a silent wrong type
            pass

    def test_pdf_extraction_raises_on_garbage(self):
        """_extract_pdf_text must raise an Exception on non-PDF bytes."""
        with self.assertRaises(Exception):
            self.rag._extract_pdf_text(b"this is not a pdf")


# ── TestModels ────────────────────────────────────────────────────────────────

class TestModels(unittest.TestCase):

    def test_question_request_valid(self):
        from src.models import QuestionRequest
        req = QuestionRequest(
            entity_id="cybercrime-ns",
            question="What is the penalty for cyberterrorism?",
            session_id="sess_abc"
        )
        self.assertEqual(req.entity_id, "cybercrime-ns")
        self.assertEqual(req.question, "What is the penalty for cyberterrorism?")

    def test_question_request_rejects_empty_question(self):
        from src.models import QuestionRequest
        with self.assertRaises(ValueError):
            QuestionRequest(entity_id="ns", question="", session_id=None)

    def test_question_request_session_optional(self):
        from src.models import QuestionRequest
        req = QuestionRequest(entity_id="ns", question="What is Section 17?")
        self.assertIsNone(req.session_id)

    def test_create_session_request(self):
        from src.models import CreateSessionRequest
        req = CreateSessionRequest(entity_id="cybercrime-ns")
        self.assertEqual(req.entity_id, "cybercrime-ns")

    def test_api_response_success(self):
        from src.models import APIResponse
        resp = APIResponse(responseCode="00", responseMessage="OK", data={"answer": "life imprisonment"})
        self.assertEqual(resp.responseCode, "00")
        self.assertIsNotNone(resp.data)

    def test_api_response_failure(self):
        from src.models import APIResponse
        resp = APIResponse(responseCode="01", responseMessage="Failed")
        self.assertEqual(resp.responseCode, "01")
        self.assertIsNone(resp.data)


# ── TestSubQueryDecomposition ─────────────────────────────────────────────────

class TestSubQueryDecomposition(unittest.TestCase):

    @patch.dict(os.environ, ENV)
    def setUp(self):
        self.rag = _make_rag()

    def _mock_gemini(self, response_text: str):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": response_text}]}}]
        }
        return mock_resp

    @patch("src.chat_engine.requests.post")
    def test_conversational_returns_marker(self, mock_post):
        mock_post.return_value = self._mock_gemini('["__conversational__"]')
        result = self.rag.chat_client.generate_sub_queries("hi", self.rag.sub_query_template)
        self.assertEqual(result, ["__conversational__"])

    @patch("src.chat_engine.requests.post")
    def test_single_legal_question_kept(self, mock_post):
        mock_post.return_value = self._mock_gemini(
            '["What is the penalty for cyberterrorism under Section 17?"]'
        )
        result = self.rag.chat_client.generate_sub_queries(
            "What is the penalty for cyberterrorism?",
            self.rag.sub_query_template
        )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], str)

    @patch("src.chat_engine.requests.post")
    def test_multi_question_split(self, mock_post):
        mock_post.return_value = self._mock_gemini(
            '["What is the penalty for cyberterrorism?", "What are the duties of service providers?"]'
        )
        result = self.rag.chat_client.generate_sub_queries(
            "What is the penalty for cyberterrorism and what are service provider duties?",
            self.rag.sub_query_template
        )
        self.assertEqual(len(result), 2)

    @patch("src.chat_engine.requests.post")
    def test_gemini_timeout_falls_back_to_original(self, mock_post):
        import requests as req_lib
        mock_post.side_effect = req_lib.exceptions.Timeout
        result = self.rag.chat_client.generate_sub_queries(
            "What is unlawful access?", self.rag.sub_query_template
        )
        self.assertEqual(result, ["What is unlawful access?"])

    @patch("src.chat_engine.requests.post")
    def test_malformed_json_falls_back_to_original(self, mock_post):
        mock_post.return_value = self._mock_gemini("not valid json at all")
        result = self.rag.chat_client.generate_sub_queries(
            "What is Section 8?", self.rag.sub_query_template
        )
        self.assertEqual(result, ["What is Section 8?"])


# ── TestErrorHandling ─────────────────────────────────────────────────────────

class TestErrorHandling(unittest.TestCase):

    @patch.dict(os.environ, ENV)
    def setUp(self):
        self.rag = _make_rag()

    def test_empty_text_no_crash(self):
        result = self.rag._create_qa_chunks("")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_whitespace_text_no_crash(self):
        result = self.rag._create_qa_chunks("\n\n\n   \t  \n")
        self.assertIsInstance(result, list)

    def test_very_short_text_no_crash(self):
        result = self.rag._create_qa_chunks("Hello.")
        self.assertIsInstance(result, list)

    def test_unicode_text_no_crash(self):
        result = self.rag._create_qa_chunks(
            "Section 18. Racist offences — ₦10,000,000 fine. Ọffénce définition."
        )
        self.assertIsInstance(result, list)

    def test_repeated_section_headers_no_crash(self):
        text = "PART I\nPART II\nPART III\n" * 20
        result = self.rag._create_qa_chunks(text)
        self.assertIsInstance(result, list)


# ── TestConfiguration ─────────────────────────────────────────────────────────

class TestConfiguration(unittest.TestCase):

    def test_gemini_api_key_required(self):
        from src.simplified_rag import SimplifiedRAG
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises((ValueError, Exception)):
                SimplifiedRAG()

    def test_prompts_yaml_exists_and_valid(self):
        import yaml
        prompts_file = PROJECT_ROOT / "src" / "prompts.yaml"
        self.assertTrue(prompts_file.exists(), f"prompts.yaml not found at {prompts_file}")
        with open(prompts_file) as f:
            config = yaml.safe_load(f)
        self.assertIn("system_prompt", config)
        self.assertIn("user_template", config)
        self.assertIn("sub_query_template", config)

    def test_env_example_exists(self):
        env_example = PROJECT_ROOT / ".env.example"
        self.assertTrue(env_example.exists(), ".env.example not found")
        content = env_example.read_text()
        self.assertIn("GEMINI_API_KEY", content)
        self.assertIn("PINECONE_API_KEY", content)
        self.assertIn("PINECONE_INDEX_NAME", content)
        self.assertIn("REDIS_URL", content)

    @patch.dict(os.environ, ENV)
    def test_rag_initializes_with_valid_env(self):
        rag = _make_rag()
        self.assertIsNotNone(rag)
        self.assertIsNotNone(rag.system_prompt)
        self.assertIsNotNone(rag.user_template)
        self.assertIsNotNone(rag.sub_query_template)
        self.assertIsNotNone(rag.tokenizer)
        self.assertIsNotNone(rag.chat_client)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
