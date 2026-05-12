"""Gemini chat and sub-query logic for RAG question answering."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import List

import requests

logger = logging.getLogger(__name__)


class GeminiChatClient:
    """Handles Gemini text generation and sub-query decomposition."""

    def __init__(self, api_key: str, api_base: str, text_model: str):
        self.api_key = api_key
        self.api_base = api_base
        self.text_model = text_model

    def generate_sub_queries(self, question: str, sub_query_template: str) -> List[str]:
        """Return 1-5 sub-queries or ["__conversational__"] for small talk."""
        started = time.time()
        url = f"{self.api_base}/{self.text_model}:generateContent?key={self.api_key}"
        prompt = sub_query_template.format(user_query=question)

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": 200,
                "temperature": 0.3,
            },
        }

        try:
            logger.info("[SUB-QUERY] Calling Gemini for query decomposition...")
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code != 200:
                logger.warning("[SUB-QUERY] Gemini API error %s: %s", response.status_code, response.text)
                return [question]

            result = response.json()
            candidates = result.get("candidates", [])
            if not candidates:
                logger.warning("[SUB-QUERY] No candidates in Gemini response")
                return [question]

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                logger.warning("[SUB-QUERY] No parts in Gemini response")
                return [question]

            text = parts[0].get("text", "").strip()
            json_match = re.search(r"\[.*?\]", text, re.DOTALL)
            if json_match:
                text = json_match.group(0)

            sub_queries = json.loads(text)
            if isinstance(sub_queries, list) and 1 <= len(sub_queries) <= 5:
                logger.info(
                    "[SUB-QUERY] Generated in %.2fs | %s queries: %s",
                    time.time() - started,
                    len(sub_queries),
                    sub_queries,
                )
                return sub_queries

            logger.warning("[SUB-QUERY] Parsed output is not a valid query list")
            return [question]
        except requests.exceptions.Timeout:
            logger.warning("[SUB-QUERY] Gemini API timeout after 30s")
            return [question]
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("[SUB-QUERY] Failed to parse Gemini response: %s", exc)
            return [question]
        except Exception as exc:
            logger.warning("[SUB-QUERY] Generation failed after %.2fs: %s", time.time() - started, exc)
            return [question]

    def generate_text(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate text response with Gemini."""
        url = f"{self.api_base}/{self.text_model}:generateContent?key={self.api_key}"
        full_user_message = f"{system_prompt}\n\n{user_message}"

        payload = {
            "contents": [{"role": "user", "parts": [{"text": full_user_message}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                "topP": 0.95,
                "topK": 40,
            },
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )

            if response.status_code != 200:
                raise Exception(f"Gemini API error {response.status_code}: {response.text}")

            result = response.json()
            candidates = result.get("candidates", [])
            if not candidates:
                raise Exception("No candidates in Gemini response")

            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise Exception("No parts in Gemini response")

            answer_text = parts[0].get("text", "").strip()
            if not answer_text:
                raise Exception("Empty answer text from Gemini")
            return answer_text
        except requests.exceptions.Timeout:
            raise Exception("Gemini API timeout after 60 seconds")
        except Exception as exc:
            logger.error("[GEMINI] Text generation failed: %s", exc)
            raise
