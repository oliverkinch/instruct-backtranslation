"""Async generation: call the OpenAI-compatible API to produce instructions for paragraphs."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from instruct_bt.config import Settings
from instruct_bt.prompts import INSTRUCTION_TEMPLATE, SYSTEM_MSG


# ---------------------------------------------------------------------------
# Low-level API call (same pattern as webr-da)
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
# Danish language check (same heuristic as webr-da)
# ---------------------------------------------------------------------------

_DA_MARKERS = frozenset(
    "og er det en at på for med har den til af kan de som ikke vil der"
    " jeg han hun vi efter også blev fra deres eller denne var blev mere".split()
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

    docs = _read_jsonl(src)

    # Resume support
    if dst.exists():
        done = _read_jsonl(dst)
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

    async def _process(doc: dict) -> dict | None:
        prompt = INSTRUCTION_TEMPLATE.format(paragraph=doc["paragraph"])
        instruction = await _call_api(client, sem, prompt, settings)

        if not instruction:
            return None

        instruction = instruction.strip()

        if not _looks_danish(instruction):
            print(f"  [lang-check] Skipping non-Danish instruction: {instruction[:80]}...")
            return None

        return {**doc, "instruction": instruction}

    tasks = [_process(d) for d in remaining]
    results = await tqdm_asyncio.gather(*tasks, desc="Generating instructions")

    new_docs = [r for r in results if r is not None]
    skipped = len(remaining) - len(new_docs)
    all_docs = done + new_docs
    _write_jsonl(dst, all_docs)
    print(
        f"Instruction generation: {len(all_docs)} ok, {skipped} skipped "
        f"(non-Danish or failed) → {dst}"
    )
    return dst
