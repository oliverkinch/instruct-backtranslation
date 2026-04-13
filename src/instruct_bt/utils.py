"""Shared utilities for async API calls, JSONL I/O, and LLM response caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import sqlite3
from pathlib import Path

from openai import AsyncOpenAI

from instruct_bt.config import Settings


# ---------------------------------------------------------------------------
# Disk-based LLM response cache (SQLite)
# ---------------------------------------------------------------------------


class LLMCache:
    """Simple disk-based cache for LLM API responses.

    The cache key is a SHA-256 hash of ``(model, temperature, system_msg,
    user_msg)``.  This ensures that:

    - Same input + same prompt template → cache hit
    - Same input + different format (A/B/C) → cache miss
    - Temperature or model changes → cache miss

    The database is stored at ``{data_dir}/.cache.db``.
    """

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  key TEXT PRIMARY KEY,"
            "  response TEXT NOT NULL"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def _make_key(
        model: str,
        temperature: float,
        system_msg: str,
        user_msg: str,
    ) -> str:
        """Compute a deterministic cache key."""
        blob = json.dumps(
            [model, temperature, system_msg, user_msg],
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(
        self,
        model: str,
        temperature: float,
        system_msg: str,
        user_msg: str,
    ) -> str | None:
        """Return the cached response, or ``None`` on cache miss."""
        key = self._make_key(model, temperature, system_msg, user_msg)
        row = self._conn.execute(
            "SELECT response FROM cache WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def put(
        self,
        model: str,
        temperature: float,
        system_msg: str,
        user_msg: str,
        response: str,
    ) -> None:
        """Store a response in the cache."""
        key = self._make_key(model, temperature, system_msg, user_msg)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)",
            (key, response),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Low-level API call with retry + back-off + caching
# ---------------------------------------------------------------------------


async def call_api(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    user_msg: str,
    settings: Settings,
    system_msg: str,
    max_retries: int = 5,
    cache: LLMCache | None = None,
) -> str | None:
    """Send a single chat completion request with retry + back-off.

    If a *cache* is provided, cached responses are returned without
    making an API call.
    """
    # Check cache first
    if cache is not None:
        cached = cache.get(
            settings.model_name, settings.temperature, system_msg, user_msg
        )
        if cached is not None:
            return cached

    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=settings.model_name,
                    temperature=settings.temperature,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                content = resp.choices[0].message.content

                # Store in cache
                if cache is not None and content is not None:
                    cache.put(
                        settings.model_name,
                        settings.temperature,
                        system_msg,
                        user_msg,
                        content,
                    )

                return content
            except Exception as exc:  # noqa: BLE001
                wait = random.uniform(1, min(30, 2**attempt))
                print(f"  [retry {attempt + 1}/{max_retries}] {exc!r} — waiting {wait:.1f}s")
                await asyncio.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    """Write a list of dicts to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
