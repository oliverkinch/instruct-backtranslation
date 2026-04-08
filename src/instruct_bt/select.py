"""LLM-based paragraph selection: pick the best chatbot-response passage from
heuristically-extracted sections, then verify the output is a verbatim extract.

This stage sits between the heuristic extraction (extract.py) and instruction
generation (generate.py).

Flow:
    paragraphs.jsonl  →  [LLM select + verify]  →  selected.jsonl
"""

from __future__ import annotations

import asyncio
import json
import random
from difflib import SequenceMatcher
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from instruct_bt.config import Settings
from instruct_bt.prompts import SELECT_TEMPLATE, SYSTEM_MSG


# ---------------------------------------------------------------------------
# Low-level API call (shared pattern)
# ---------------------------------------------------------------------------


async def _call_api(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    user_msg: str,
    settings: Settings,
    system_msg: str = SYSTEM_MSG,
    max_retries: int = 5,
) -> str | None:
    """Send a single chat completion request with retry + back-off."""
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
                return resp.choices[0].message.content
            except Exception as exc:  # noqa: BLE001
                wait = random.uniform(1, min(30, 2**attempt))
                print(f"  [retry {attempt + 1}/{max_retries}] {exc!r} — waiting {wait:.1f}s")
                await asyncio.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, items: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


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

    # Slow path: character-level similarity
    ratio = SequenceMatcher(None, norm_source, norm_extracted).ratio()
    # The ratio is over the full source, but we want to know if the extracted
    # text is a near-exact match to *some part* of the source. So we also
    # check the ratio against just the extracted text length.
    if len(norm_extracted) == 0:
        return False

    # Find the best matching block and compute local similarity
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

    docs = _read_jsonl(src)

    # Resume support
    if dst.exists():
        done = _read_jsonl(dst)
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

    skipped_skip = 0
    skipped_verify = 0
    skipped_short = 0
    skipped_fail = 0

    async def _process(doc: dict) -> dict | None:
        nonlocal skipped_skip, skipped_verify, skipped_short, skipped_fail

        prompt = SELECT_TEMPLATE.format(paragraph=doc["paragraph"])
        result = await _call_api(client, sem, prompt, settings)

        if not result:
            skipped_fail += 1
            return None

        result = result.strip()

        # LLM decided no good passage exists
        if result.upper() == "SKIP":
            skipped_skip += 1
            return None

        # Too short to be useful
        if len(result) < 50:
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
        }

    tasks = [_process(d) for d in remaining]
    results = await tqdm_asyncio.gather(*tasks, desc="Selecting paragraphs")

    new_docs = [r for r in results if r is not None]
    all_docs = done + new_docs

    _write_jsonl(dst, all_docs)

    total_skipped = skipped_skip + skipped_verify + skipped_short + skipped_fail
    print(
        f"Selection: {len(all_docs)} kept, {total_skipped} skipped "
        f"(SKIP={skipped_skip}, verify={skipped_verify}, "
        f"short={skipped_short}, fail={skipped_fail}) → {dst}"
    )
    return dst
