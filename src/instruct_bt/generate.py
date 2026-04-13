"""Async generation: call the OpenAI-compatible API to produce instructions for paragraphs."""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from instruct_bt.config import Settings
from instruct_bt.prompts import INSTRUCTION_TEMPLATES, INSTRUCTION_FORMATS, GENERATE_SYSTEM_MSG
from instruct_bt.utils import LLMCache, call_api, read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Danish language check (same heuristic as webr-da)
# ---------------------------------------------------------------------------

_DA_MARKERS = frozenset(
    "og er det en at på for med har den til af kan de som ikke vil der"
    " jeg han hun vi efter også blev fra deres eller denne var mere".split()
)
_DA_CHECK_THRESHOLD = 0.06


def _looks_danish(text: str) -> bool:
    """Fast heuristic: does *text* look like Danish based on stop-word frequency?"""
    words = text.lower().split()
    if len(words) < 8:
        return True
    hits = sum(1 for w in words if w in _DA_MARKERS)
    return (hits / len(words)) >= _DA_CHECK_THRESHOLD


# ---------------------------------------------------------------------------
# Stage: Generate instructions
# ---------------------------------------------------------------------------


async def generate_instructions(settings: Settings) -> Path:
    """For each selected paragraph, generate the instruction that would produce it.

    Reads from ``{data_dir}/selected.jsonl``, writes to
    ``{data_dir}/with_instructions.jsonl``.

    Supports resumption: already-processed paragraphs (matched by text)
    are skipped.
    """
    src = settings.data_dir / "selected.jsonl"
    dst = settings.data_dir / "with_instructions.jsonl"

    docs = read_jsonl(src)

    # Resume support
    if dst.exists():
        done = read_jsonl(dst)
        done_texts = {d["paragraph"] for d in done}
        remaining = [d for d in docs if d["paragraph"] not in done_texts]
        print(f"Resuming instruction generation: {len(done)} done, {len(remaining)} remaining")
    else:
        done = []
        remaining = docs

    if not remaining:
        print("Instruction generation already complete.")
        return dst

    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
    sem = asyncio.Semaphore(settings.max_concurrency)
    cache = LLMCache(settings.data_dir / ".cache.db")

    async def _process(doc: dict) -> dict | None:
        source_type = doc.get("source_type", "knowledge")
        format_templates = INSTRUCTION_TEMPLATES.get(source_type, INSTRUCTION_TEMPLATES["knowledge"])
        # Pick format deterministically from paragraph content so that
        # re-runs hit the cache for the same paragraph.
        fmt_idx = int(hashlib.sha256(doc["paragraph"].encode()).hexdigest(), 16) % len(INSTRUCTION_FORMATS)
        fmt = INSTRUCTION_FORMATS[fmt_idx]
        template = format_templates[fmt]
        prompt = template.format(paragraph=doc["paragraph"])
        instruction = await call_api(client, sem, prompt, settings, GENERATE_SYSTEM_MSG, cache=cache)

        if not instruction:
            return None

        instruction = instruction.strip()

        # Strip format labels the LLM sometimes echoes from the prompt
        # (e.g. "Format C: Jeg skal holde en tale...")
        instruction = re.sub(
            r"^Format\s+[A-C]\s*[-:—]\s*", "", instruction
        ).strip()
        # Also strip if it's on its own line before the actual message
        instruction = re.sub(
            r"^Format\s+[A-C]\s*\n+", "", instruction
        ).strip()

        if not _looks_danish(instruction):
            print(f"  [lang-check] Skipping non-Danish instruction: {instruction[:80]}...")
            return None

        return {**doc, "instruction": instruction}

    tasks = [_process(d) for d in remaining]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating instructions")

    new_docs = [r for r in results if r is not None]
    skipped = len(remaining) - len(new_docs)
    all_docs = done + new_docs
    write_jsonl(dst, all_docs)
    cache.close()
    print(
        f"Instruction generation: {len(all_docs)} ok, {skipped} skipped "
        f"(non-Danish or failed) → {dst}"
    )
    return dst
