"""Post-processing: filter refusals, MinHash dedup, format to chat parquet."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pyarrow
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from transformers import AutoTokenizer

from instruct_bt.config import Settings


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

_REFUSAL_PREFIXES = (
    "I'm sorry",
    "I apologize",
    "I cannot",
    "I can't",
    "Jeg beklager",
    "Undskyld",
    "Jeg kan ikke",
    "Det er ikke muligt",
)


def _filter_refusals(docs: list[dict]) -> list[dict]:
    """Remove documents where the instruction starts with a refusal pattern."""
    kept = []
    removed = 0
    for doc in docs:
        instruction = doc.get("instruction", "")
        paragraph = doc.get("paragraph", "")
        if not instruction or not paragraph:
            removed += 1
            continue
        if any(instruction.startswith(prefix) for prefix in _REFUSAL_PREFIXES):
            removed += 1
            continue
        kept.append(doc)
    print(f"Refusal filter: kept {len(kept)}, removed {removed}")
    return kept


# ---------------------------------------------------------------------------
# MinHash deduplication
# ---------------------------------------------------------------------------


def _minhash_dedup(
    docs: list[dict],
    tokenizer: AutoTokenizer,
    *,
    key: str = "instruction",
    threshold: float = 0.7,
    num_perm: int = 128,
) -> list[dict]:
    """Remove near-duplicate documents based on MinHash of the given key's tokens."""

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[int, MinHash] = {}

    for i, doc in enumerate(tqdm(docs, desc="Building MinHash signatures")):
        m = MinHash(num_perm=num_perm)
        token_ids = tokenizer(doc[key])["input_ids"]
        for tid in token_ids:
            m.update(str(tid).encode("utf8"))
        minhashes[i] = m
        lsh.insert(f"doc_{i}", m)

    remove_indices: set[int] = set()
    for i, mh in tqdm(minhashes.items(), desc="Querying MinHash LSH"):
        if i in remove_indices:
            continue
        results = lsh.query(mh)
        if len(results) > 1:
            sorted_results = sorted(results, key=lambda x: int(x.split("_")[1]))
            for r in sorted_results[1:]:
                remove_indices.add(int(r.split("_")[1]))

    kept = [docs[i] for i in range(len(docs)) if i not in remove_indices]
    print(
        f"MinHash dedup: kept {len(kept)}, removed {len(remove_indices)} "
        f"({len(remove_indices) / max(len(docs), 1) * 100:.1f}%)"
    )
    return kept


# ---------------------------------------------------------------------------
# Format to chat messages
# ---------------------------------------------------------------------------


def _format_chat(docs: list[dict]) -> list[dict]:
    """Convert to the final chat-message format expected for instruction tuning.

    Each sample becomes:
        user: <instruction>
        assistant: <paragraph>  (the real, non-synthetic text)
    """
    formatted = []
    for i, doc in enumerate(docs):
        messages = [
            {"content": doc["instruction"], "role": "user"},
            {"content": doc["paragraph"], "role": "assistant"},
        ]
        formatted.append({
            "prompt_id": f"instruct_bt_{i}",
            "messages": messages,
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "section_heading": doc.get("section_heading", ""),
        })
    return formatted


# ---------------------------------------------------------------------------
# Main post-processing entry point
# ---------------------------------------------------------------------------


def postprocess(settings: Settings) -> Path:
    """Run the full post-processing pipeline and write the final parquet."""
    src = settings.data_dir / "with_instructions.jsonl"
    dst = settings.data_dir / "final.parquet"

    with open(src, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(docs)} instruction-paragraph pairs")

    # Filter refusals
    docs = _filter_refusals(docs)

    # Load tokenizer once for both dedup passes
    print(f"Loading tokenizer {settings.minhash_tokenizer} for MinHash dedup ...")
    tokenizer = AutoTokenizer.from_pretrained(settings.minhash_tokenizer)

    # MinHash dedup on instructions
    docs = _minhash_dedup(
        docs,
        tokenizer=tokenizer,
        key="instruction",
        threshold=settings.minhash_threshold,
        num_perm=settings.minhash_num_perm,
    )

    # Also dedup on paragraphs (in case the same paragraph appears with different instructions)
    docs = _minhash_dedup(
        docs,
        tokenizer=tokenizer,
        key="paragraph",
        threshold=settings.minhash_threshold,
        num_perm=settings.minhash_num_perm,
    )

    # Shuffle (seeded for reproducibility)
    random.seed(42)
    random.shuffle(docs)

    print(f"Final dataset: {len(docs)} samples")

    # Format to chat messages
    formatted = _format_chat(docs)

    # Save as parquet
    df = pd.DataFrame(formatted)
    print(f"Writing {len(df)} samples to {dst}")
    pq.write_table(table=pyarrow.Table.from_pandas(df), where=str(dst))

    # Also save a JSONL copy for easy inspection
    jsonl_dst = settings.data_dir / "final.jsonl"
    with open(jsonl_dst, "w", encoding="utf-8") as f:
        for item in formatted:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    print(f"Also wrote JSONL copy to {jsonl_dst}")

    return dst
