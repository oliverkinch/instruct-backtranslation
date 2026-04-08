"""Extract candidate paragraphs from a HuggingFace dataset.

The default extraction strategy splits Wikipedia articles by section
headings (## in markdown) and filters for paragraphs that are plausible
as standalone chatbot responses.

To adapt for a different dataset, implement a new ``extract_paragraphs``
function with the same signature and wire it into the pipeline.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


@dataclass
class Paragraph:
    """A single extracted paragraph with provenance metadata."""

    text: str
    title: str
    url: str
    section_heading: str


# ---------------------------------------------------------------------------
# Section headings that are almost never useful as chatbot responses
# ---------------------------------------------------------------------------

_SKIP_HEADINGS = re.compile(
    r"^(se også|eksterne henvisninger|kilder|referencer|litteratur|"
    r"fodnoter|noter|galleri|spor|medvirkende|diskografi|filmografi|"
    r"bibliografi|priser|links|weblinks|external links|see also|references)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Section splitting (Wikipedia markdown)
# ---------------------------------------------------------------------------


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a Wikipedia article into (heading, body) pairs.

    The first section (before any heading) uses the article title area
    as its heading. Subsequent sections are delimited by markdown
    headings (## or ###).

    Returns a list of (heading, body_text) tuples.
    """
    # Split on lines that start with ##
    parts: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            # Flush the previous section
            body = "\n".join(current_lines).strip()
            if body:
                parts.append((current_heading, body))
            # Start new section — strip the heading markers
            current_heading = stripped.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush the last section
    body = "\n".join(current_lines).strip()
    if body:
        parts.append((current_heading, body))

    return parts


# ---------------------------------------------------------------------------
# Paragraph-level filters
# ---------------------------------------------------------------------------


def _is_mostly_list(text: str) -> bool:
    """Return True if more than 50% of non-empty lines are list items or table rows."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return True
    list_lines = sum(
        1 for l in lines if l.strip().startswith(("* ", "- ", "| ", "1.", "2.", "3."))
    )
    return list_lines / len(lines) > 0.5


def _has_too_many_links(text: str) -> bool:
    """Return True if the text is dominated by markdown links (e.g. reference sections)."""
    link_chars = sum(len(m.group()) for m in re.finditer(r"\[.*?\]\(.*?\)", text))
    if len(text) == 0:
        return True
    return link_chars / len(text) > 0.4


def _is_stub(text: str) -> bool:
    """Return True if the paragraph is a Wikipedia stub notice or too short to be useful."""
    lower = text.lower()
    if "stub" in lower and ("denne artikel" in lower or "denne side" in lower):
        return True
    return False


def _starts_with_title_repetition(text: str, title: str) -> bool:
    """Check if the section body is just the article title repeated as a header.

    Wikipedia articles often start with ``# Title\\nTitle er ...``.
    We want to keep the explanatory text but strip the repeated title line.
    """
    first_line = text.split("\n")[0].strip().lstrip("#").strip()
    return first_line == title


def _clean_paragraph(text: str, title: str) -> str:
    """Light cleaning of a paragraph before filtering.

    - Strip the leading ``# Title`` line that Wikipedia articles start with.
    - Remove trailing whitespace.
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip top-level headings (# Title) — these are article titles, not content
        if stripped.startswith("# ") and not stripped.startswith("## "):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def filter_paragraph(
    text: str,
    heading: str,
    *,
    min_chars: int = 100,
    max_chars: int = 3000,
) -> bool:
    """Return True if the paragraph passes all quality filters."""
    # Skip known non-content sections
    if _SKIP_HEADINGS.match(heading):
        return False

    # Length filters
    if len(text) < min_chars or len(text) > max_chars:
        return False

    # Content quality filters
    if _is_mostly_list(text):
        return False

    if _has_too_many_links(text):
        return False

    if _is_stub(text):
        return False

    return True


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------


def extract_paragraphs(
    output_path: Path,
    *,
    dataset_name: str = "oliverkinch/danish_wikipedia",
    text_column: str = "text",
    title_column: str = "title",
    url_column: str = "url",
    n_samples: int = 0,
    seed: int = 42,
    min_chars: int = 100,
    max_chars: int = 3000,
    streaming: bool = False,
) -> Path:
    """Extract candidate paragraphs from a HuggingFace dataset.

    Parameters
    ----------
    output_path
        Where to write the JSONL output.
    dataset_name
        HuggingFace dataset identifier.
    text_column
        Name of the column containing the article text.
    title_column
        Name of the column containing the article title (use "" to skip).
    url_column
        Name of the column containing the source URL (use "" to skip).
    n_samples
        Maximum number of paragraphs to keep. 0 means keep all.
    seed
        Random seed for sampling.
    min_chars
        Minimum paragraph length in characters.
    max_chars
        Maximum paragraph length in characters.
    streaming
        Whether to stream the dataset (useful for very large datasets).

    Returns
    -------
    Path
        The path to the written JSONL file.
    """
    random.seed(seed)

    if output_path.exists():
        existing = sum(1 for _ in open(output_path))
        if n_samples > 0 and existing >= n_samples:
            print(f"Paragraphs already extracted at {output_path} ({existing} paragraphs). Skipping.")
            return output_path
        elif n_samples == 0 and existing > 0:
            print(f"Paragraphs already extracted at {output_path} ({existing} paragraphs). Skipping.")
            return output_path
        print(f"Found {existing} paragraphs but need more. Re-extracting.")

    print(f"Loading dataset {dataset_name} ...")
    ds = load_dataset(dataset_name, split="train", streaming=streaming)

    paragraphs: list[Paragraph] = []

    for example in tqdm(ds, desc="Extracting paragraphs"):
        text = example[text_column]
        title = example.get(title_column, "") if title_column else ""
        url = example.get(url_column, "") if url_column else ""

        sections = _split_sections(text)

        for heading, body in sections:
            cleaned = _clean_paragraph(body, title)
            if filter_paragraph(cleaned, heading, min_chars=min_chars, max_chars=max_chars):
                paragraphs.append(
                    Paragraph(
                        text=cleaned,
                        title=title,
                        url=url,
                        section_heading=heading,
                    )
                )

    print(f"Extracted {len(paragraphs)} candidate paragraphs from {dataset_name}")

    # Sample if requested
    if n_samples > 0 and len(paragraphs) > n_samples:
        random.shuffle(paragraphs)
        paragraphs = paragraphs[:n_samples]
        print(f"Sampled down to {n_samples} paragraphs")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for p in paragraphs:
            record = {
                "paragraph": p.text,
                "title": p.title,
                "url": p.url,
                "section_heading": p.section_heading,
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(paragraphs)} paragraphs to {output_path}")
    return output_path
