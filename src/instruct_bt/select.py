"""LLM-based paragraph selection: pick the best chatbot-response passage from
heuristically-extracted sections, then verify the output is a verbatim extract.

This stage sits between the heuristic extraction (extract.py) and instruction
generation (generate.py).

Flow:
    paragraphs.jsonl  →  [LLM select + verify]  →  selected.jsonl
"""

from __future__ import annotations

import asyncio
from difflib import SequenceMatcher
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from instruct_bt.config import Settings
from instruct_bt.prompts import SELECT_TEMPLATES, SELECT_SYSTEM_MSG
from instruct_bt.utils import LLMCache, call_api, read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Verification: ensure the LLM output is a verbatim extract
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Normalize whitespace for comparison."""
    return " ".join(text.split()).strip()


def _is_verbatim(source: str, extracted: str, threshold: float = 0.90) -> bool:
    """Check that *extracted* is (nearly) a verbatim substring of *source*.

    Uses SequenceMatcher ratio as a character-level similarity measure.
    A threshold of 0.90 allows for minor whitespace or punctuation
    differences while catching real paraphrasing.

    Also checks for direct substring containment (fast path).
    """
    norm_source = _normalize(source)
    norm_extracted = _normalize(extracted)

    # Fast path: exact substring
    if norm_extracted in norm_source:
        return True

    if len(norm_extracted) == 0:
        return False

    # Find the best matching blocks and compute local similarity
    matcher = SequenceMatcher(None, norm_source, norm_extracted)
    # Get all matching blocks
    blocks = matcher.get_matching_blocks()
    matched_chars = sum(block.size for block in blocks)
    local_ratio = matched_chars / len(norm_extracted)

    return local_ratio >= threshold


# ---------------------------------------------------------------------------
# Main selection stage
# ---------------------------------------------------------------------------


async def select_paragraphs(settings: Settings) -> Path:
    """For each heuristically-extracted paragraph, ask the LLM to select the
    best passage that works as a chatbot response.

    Reads from ``{data_dir}/paragraphs.jsonl``, writes to
    ``{data_dir}/selected.jsonl``.

    Each output record replaces ``paragraph`` with the LLM-selected passage
    and adds ``original_paragraph`` for traceability.

    Supports resumption.
    """
    src = settings.data_dir / "paragraphs.jsonl"
    dst = settings.data_dir / "selected.jsonl"

    docs = read_jsonl(src)

    # Resume support
    if dst.exists():
        done = read_jsonl(dst)
        done_originals = {d["original_paragraph"] for d in done}
        remaining = [d for d in docs if d["paragraph"] not in done_originals]
        print(f"Resuming selection: {len(done)} done, {len(remaining)} remaining")
    else:
        done = []
        remaining = docs

    if not remaining:
        print("Selection stage already complete.")
        return dst

    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
    sem = asyncio.Semaphore(settings.max_concurrency)
    cache = LLMCache(settings.data_dir / ".cache.db")

    # Counters are safe to mutate from concurrent coroutines because
    # asyncio.gather runs them on a single thread (no true parallelism).
    skipped_skip = 0
    skipped_verify = 0
    skipped_short = 0
    skipped_fail = 0

    async def _process(doc: dict) -> dict | None:
        nonlocal skipped_skip, skipped_verify, skipped_short, skipped_fail

        source_type = doc.get("source_type", "knowledge")
        template = SELECT_TEMPLATES.get(source_type, SELECT_TEMPLATES["knowledge"])
        prompt = template.format(paragraph=doc["paragraph"])
        result = await call_api(client, sem, prompt, settings, SELECT_SYSTEM_MSG, cache=cache)

        if not result:
            skipped_fail += 1
            return None

        result = result.strip()

        # LLM decided no good passage exists
        if result.upper() == "SKIP":
            skipped_skip += 1
            return None

        # Too short to be a satisfying chatbot response
        if len(result) < 150:
            skipped_short += 1
            return None

        # Verify it's a verbatim extract
        if not _is_verbatim(doc["paragraph"], result):
            skipped_verify += 1
            return None

        return {
            "paragraph": result,
            "original_paragraph": doc["paragraph"],
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "section_heading": doc.get("section_heading", ""),
            "source_type": doc.get("source_type", "knowledge"),
        }

    tasks = [_process(d) for d in remaining]
    results = await tqdm_asyncio.gather(*tasks, desc="Selecting paragraphs")

    new_docs = [r for r in results if r is not None]
    all_docs = done + new_docs

    write_jsonl(dst, all_docs)
    cache.close()

    total_skipped = skipped_skip + skipped_verify + skipped_short + skipped_fail
    print(
        f"Selection: {len(all_docs)} kept, {total_skipped} skipped "
        f"(SKIP={skipped_skip}, verify={skipped_verify}, "
        f"short={skipped_short}, fail={skipped_fail}) → {dst}"
    )
    return dst
