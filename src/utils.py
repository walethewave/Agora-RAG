"""Utility helpers shared across the RAG modules."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import redis
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_project_env() -> Path:
    """Load .env from project root and return the file path used."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_file, override=True)
    return env_file


def read_env_value(key: str, env_file: Path) -> str | None:
    """Read a value directly from a .env file, bypassing os.environ."""
    if not env_file.exists():
        return None

    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue

        env_key, _, env_value = line.partition("=")
        if env_key.strip() == key:
            env_value = env_value.strip().strip('"').strip("'")
            return env_value if env_value else None
    return None


class ConversationMemory:
    """Manages per-session conversation history in Upstash Redis."""

    TTL = 1800
    MAX_MESSAGES = 5

    def __init__(self, redis_url: str):
        self.client = redis.from_url(redis_url, decode_responses=True)
        logger.info("[CONFIG] Connected to Upstash Redis")

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:history"

    def get_history(self, session_id: str) -> str:
        """Return last MAX_MESSAGES Q&A pairs as text context."""
        try:
            raw_messages = self.client.lrange(self._key(session_id), -(self.MAX_MESSAGES * 2), -1)
            messages = [json.loads(message) for message in raw_messages]
            if not messages:
                return "No previous conversation."

            lines = []
            for message in messages:
                role = "User" if message["role"] == "user" else "Qorpy"
                lines.append(f"{role}: {message['content']}")
            return "\n".join(lines)
        except Exception as exc:
            logger.warning("[REDIS] Failed to get history for session %s: %s", session_id, exc)
            return "No previous conversation."

    def save(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        """Append user + assistant messages, trim to window, refresh TTL."""
        try:
            key = self._key(session_id)
            pipe = self.client.pipeline()
            pipe.rpush(key, json.dumps({"role": "user", "content": user_msg}))
            pipe.rpush(key, json.dumps({"role": "assistant", "content": assistant_msg}))
            pipe.ltrim(key, -(self.MAX_MESSAGES * 2), -1)
            pipe.expire(key, self.TTL)
            pipe.execute()
            logger.info("[REDIS] Saved conversation turn for session %s", session_id)
        except Exception as exc:
            logger.warning("[REDIS] Failed to save history for session %s: %s", session_id, exc)
