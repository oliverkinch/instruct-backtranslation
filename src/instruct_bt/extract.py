"""Extract candidate paragraphs from a dataset.

Supports two data sources:

* **HuggingFace** — loads a dataset via ``datasets.load_dataset()``.
* **Local parquet** — if ``source.parquet`` exists in the data directory,
  it is loaded directly instead of fetching from HuggingFace.

Two extraction strategies are available:

* **Wikipedia** — splits articles by ``##`` / ``###`` markdown section
  headings and applies Wikipedia-specific filters (skip headings, stub
  detection).  Used automatically when the dataset looks like Wikipedia.

* **Generic** — splits documents on blank-line boundaries, merges short
  consecutive paragraphs, and applies domain-agnostic quality filters.
  Used for DynaWord subsets and any other plain-text source.

The strategy is selected automatically based on the dataset name and
subset, but the caller can also override it.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class Paragraph:
    """A single extracted paragraph with provenance metadata."""

    text: str
    title: str
    url: str
    section_heading: str
    source_type: str = "knowledge"


# ---------------------------------------------------------------------------
# Section headings that are almost never useful as chatbot responses
# ---------------------------------------------------------------------------

_SKIP_HEADINGS = re.compile(
    r"^(se også|eksterne henvisninger|kilder|referencer|litteratur|"
    r"fodnoter|noter|galleri|spor|medvirkende|diskografi|filmografi|"
    r"bibliografi|priser|links|weblinks|external links|see also|references|"
    r"hædersbevisninger|gengivelser|vælgertilslutning|mandater|"
    r"valg(?:resultater)?|resultater|statistik|rekorder|meritter|"
    r"karrierestatistik|kampresultater|placeringer|hold|besætning|"
    r"udgivelser|singler|album|værker|roller|episode(?:r|liste)|"
    r"sæson(?:er)?|æresbevisninger|ordner og dekorationer|udmærkelser|"
    r"personlige rekorder)$",
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
# Generic paragraph splitting (for non-Wikipedia sources like DynaWord)
# ---------------------------------------------------------------------------


def _merge_blocks(
    raw_blocks: list[str],
    *,
    max_chars: int = 3000,
    joiner: str = "\n\n",
) -> list[str]:
    """Merge consecutive short text blocks into chunks up to *max_chars*.

    Blocks that individually exceed *max_chars* are emitted as-is.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    joiner_len = len(joiner)

    for block in raw_blocks:
        block_len = len(block)

        # If a single block already exceeds max_chars, flush current
        # and emit the block on its own
        if block_len > max_chars:
            if current:
                chunks.append(joiner.join(current))
                current = []
                current_len = 0
            chunks.append(block)
            continue

        # Would adding this block exceed max_chars?
        new_len = current_len + block_len + (joiner_len if current else 0)
        if new_len > max_chars and current:
            chunks.append(joiner.join(current))
            current = [block]
            current_len = block_len
        else:
            current.append(block)
            current_len = new_len

    if current:
        chunks.append(joiner.join(current))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on ``. ! ?`` followed by a space or end.

    This is intentionally simple — it works well enough for chunking
    purposes even if it occasionally splits on abbreviations.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in parts if s.strip()]


def _split_paragraphs(
    text: str,
    *,
    min_chars: int = 100,
    max_chars: int = 3000,
) -> list[tuple[str, str]]:
    """Split a document into paragraph chunks.

    Unlike ``_split_sections`` which relies on markdown ``##`` headings,
    this function works with *any* plain-text document.

    Strategy (each step is tried only if the previous one yields a
    single oversized block):

      1. Split on blank lines (``\\n\\n``).
      2. Split on single newlines.
      3. Split on sentence boundaries (``. ! ?`` followed by whitespace).

    Consecutive short blocks are merged so that each chunk aims to stay
    below *max_chars* characters.

    Returns a list of ``("", body_text)`` tuples (heading is always
    empty because generic documents don't have section headings).
    """
    # ----- First attempt: split on blank lines -----
    raw_blocks = re.split(r"\n\s*\n", text)
    raw_blocks = [b.strip() for b in raw_blocks if b.strip()]

    # ----- Fallback 1: single newlines -----
    if len(raw_blocks) <= 1 and len(text) > max_chars:
        raw_blocks = text.split("\n")
        raw_blocks = [b.strip() for b in raw_blocks if b.strip()]

    # ----- Fallback 2: sentence boundaries (for no-newline OCR text) -----
    if len(raw_blocks) <= 1 and len(text) > max_chars:
        raw_blocks = _split_sentences(text)

    if not raw_blocks:
        return []

    joiner = " " if len(raw_blocks) > 1 and "\n" not in raw_blocks[0] else "\n"
    chunks = _merge_blocks(raw_blocks, max_chars=max_chars, joiner=joiner)

    return [("", chunk) for chunk in chunks]


# ---------------------------------------------------------------------------
# Paragraph-level filters
# ---------------------------------------------------------------------------


def _is_mostly_list(text: str) -> bool:
    """Return True if more than 50% of non-empty lines are list items or table rows."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True
    list_lines = sum(
        1 for ln in lines if ln.strip().startswith(("* ", "- ", "| ", "1.", "2.", "3."))
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


def _is_year_heavy(text: str) -> bool:
    """Return True if the text is dominated by year mentions — a strong signal
    of chronological dumps (election results, career CVs, season summaries).

    Heuristic: if more than 1 in 30 words is a 4-digit year (1000-2099),
    the text is likely a timeline rather than an explanation.
    """
    words = text.split()
    if len(words) < 20:
        return False
    year_count = sum(
        1 for w in words
        if re.match(r"^[\(]?\d{4}[\)\.\-,;:]*$", w)
        and 1000 <= int(re.search(r"\d{4}", w).group()) <= 2099
    )
    return year_count / len(words) > (1 / 30)


def _too_few_sentences(text: str, min_sentences: int = 3) -> bool:
    """Return True if the text has fewer than *min_sentences* sentences.

    Catches short encyclopedic blurbs that technically pass the character
    minimum but would make terrible chatbot responses (e.g. "X var en
    dansk forfatter. Han modtog fem Emmy Awards.").
    """
    # Simple sentence splitter: split on . ! ? followed by space or end-of-string
    sentences = re.split(r"[.!?](?:\s|$)", text.strip())
    # Filter out empty splits
    sentences = [s for s in sentences if s.strip()]
    return len(sentences) < min_sentences


# ---------------------------------------------------------------------------
# Regex for non-Latin script blocks that should never appear in Danish text.
# Covers CJK Unified, Katakana, Hiragana, Hangul, Cyrillic, Arabic, Thai,
# Devanagari, and other scripts unlikely in Danish.
# ---------------------------------------------------------------------------

_NON_LATIN_RE = re.compile(
    r"[\u0400-\u04FF"   # Cyrillic
    r"\u0600-\u06FF"    # Arabic
    r"\u0900-\u097F"    # Devanagari
    r"\u0E00-\u0E7F"    # Thai
    r"\u3000-\u303F"    # CJK symbols
    r"\u3040-\u309F"    # Hiragana
    r"\u30A0-\u30FF"    # Katakana
    r"\u4E00-\u9FFF"    # CJK Unified Ideographs
    r"\uAC00-\uD7AF"    # Hangul
    r"\U00020000-\U0002A6DF"  # CJK Extension B
    r"]"
)

# Doubled-syllable pattern: 2-4 word characters repeated immediately.
# Matches corruption like "overføføre", "tilskukud", "foforbrbedre".
# We look for the pattern inside word boundaries to avoid false positives
# on legitimate Danish words.
_DOUBLED_SYLLABLE_RE = re.compile(r"(\w{2,4})\1")

# Common Danish words that legitimately contain repeated patterns.
# These would otherwise trigger the doubled-syllable detector.
_DOUBLED_LEGIT = frozenset({
    "alle", "allerede", "alligevel",
    "behandlingen",
    "eventuelt", "eventually",
    "hovedsageligt",
    "indeholder",
    "lille",
    "mange",
    "muligvis",
    "nødvendigvis",
    "parallelle", "parallel",
    "visse",
})

# Danish vowels — used by ``_has_encoding_corruption`` to detect stray
# consonant clusters that signal PDF extraction artefacts.
_DA_VOWELS = frozenset("aeiouyæøå")


def _has_encoding_corruption(text: str) -> bool:
    """Return True if the text shows signs of encoding corruption or OCR artifacts.

    Checks for:
    1. Non-Latin script characters (CJK, Cyrillic, Arabic, etc.) that
       should never appear in Danish text — catches encoding corruption.
    2. Doubled syllables from PDF extraction artifacts (e.g. "overføføre",
       "frfra", "tilskukud"). Aggressive: rejects on ≥ 2 occurrences.
    """
    # --- Check 1: non-Latin characters ---
    if _NON_LATIN_RE.search(text):
        return True

    # --- Check 2: doubled syllables ---
    words = text.split()
    doubled_count = 0
    for word in words:
        clean = word.strip(".,;:!?()[]{}\"'""''»«–—-")
        if clean.lower() in _DOUBLED_LEGIT:
            continue
        if _DOUBLED_SYLLABLE_RE.search(clean.lower()):
            doubled_count += 1
    # Aggressive threshold: 2 or more suspicious doubled-syllable words
    if doubled_count >= 2:
        return True

    # --- Check 3: stray non-word character clusters ---
    # Catches PDF extraction artifacts like "åå ygy" — short isolated
    # character sequences (1-3 chars) that don't form real Danish words.
    # We count words that are very short AND contain no vowels or are
    # all-consonant gibberish.
    stray_count = 0
    for word in words:
        clean = word.strip(".,;:!?()[]{}\"'""''»«–—-/\\")
        if len(clean) <= 3 and clean and not any(c in _DA_VOWELS for c in clean.lower()):
            stray_count += 1
    # Allow a few (prepositions etc.), but flag if many
    if len(words) > 10 and stray_count / len(words) > 0.05:
        return True

    return False


def _is_quote_heavy(text: str) -> bool:
    """Return True if the text is dominated by direct speech / dialogue.

    Catches passages that are mostly quotes — unsuitable as AI-assistant
    responses because they're someone else's words in dialogue form.

    Checks:
    1. >40% of non-empty lines start with a speech marker (–, —, », ").
    2. >40% of total characters are inside quotation marks.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return True

    # --- Check 1: lines starting with speech markers ---
    speech_markers = ("– ", "— ", "» ", "\"", "\u201c")  # –, —, », ", "
    speech_lines = sum(
        1 for ln in lines if ln.strip().startswith(speech_markers)
    )
    if speech_lines / len(lines) > 0.4:
        return True

    # --- Check 2: proportion of text inside quotation marks ---
    # Match text between common Danish/international quote pairs
    quoted_chars = 0
    for pattern in [
        r'"[^"]{2,}"',       # "..."
        r'»[^«]{2,}«',       # »...«
        r'\u201c[^\u201d]{2,}\u201d',  # "..."
        r'\u201e[^\u201c]{2,}\u201c',  # „..."
    ]:
        for m in re.finditer(pattern, text):
            quoted_chars += len(m.group())
    if len(text) > 0 and quoted_chars / len(text) > 0.4:
        return True

    return False


def _is_boilerplate(text: str) -> bool:
    """Return True if the text is mostly structural/boilerplate content.

    Catches table-of-contents, program listings, and other non-prose
    content where most lines are very short labels or headings rather
    than actual paragraphs.

    Heuristic: if >60% of non-empty lines have fewer than 8 words AND
    are not proper prose sentences (don't end with sentence-final
    punctuation), the text is likely boilerplate.
    """
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 3:
        return False

    short_nonsentence_lines = 0
    for line in lines:
        stripped = line.strip()
        word_count = len(stripped.split())
        ends_with_punct = stripped.rstrip().endswith((".", "!", "?", ":"))
        if word_count < 8 and not ends_with_punct:
            short_nonsentence_lines += 1

    return short_nonsentence_lines / len(lines) > 0.6


# ---------------------------------------------------------------------------
# Archaic Danish detection
# ---------------------------------------------------------------------------

# Modern Danish proper nouns and place names that legitimately contain "aa".
_AA_PROPER_NOUNS = frozenset({
    "aalborg", "aarhus", "aabenraa", "aarup", "aakirkeby", "aalestrup",
    "aabybro", "aars", "aasiaat",
})

# Archaic function words / verb forms that are distinctive to pre-reform Danish.
_ARCHAIC_WORDS = frozenset({
    "thi", "skulde", "saadan", "saadanne", "eder", "giøre", "giorde",
    "hielpe", "hielp", "saae", "staaer", "haabe", "gaaer", "gaae",
    "maae", "maatte", "altsaa", "saaledes", "ligeledes",
})


def _is_archaic_danish(text: str) -> bool:
    """Return True if the text appears to be written in pre-1948 Danish orthography.

    Detects archaic Danish via two signals:

    1. **"aa" words**: Pre-reform Danish used "aa" where modern Danish uses
       "å" (e.g. "paa" → "på", "Aar" → "år").  Proper nouns (Aalborg,
       Aarhus) are excluded.  If >2 % of words contain "aa", the text is
       likely archaic.

    2. **Archaic vocabulary**: A small set of function words and verb forms
       that are distinctive to old Danish ("thi", "skulde", "giøre", etc.).
       If >1 % of words are archaic markers, the text is likely archaic.

    Either signal alone is sufficient to flag the text.
    """
    words = text.split()
    if len(words) < 20:
        return False

    # --- Signal 1: "aa" words ---
    aa_count = 0
    for word in words:
        clean = word.strip(".,;:!?()[]{}\"'""''»«–—-/\\").lower()
        if "aa" in clean and clean not in _AA_PROPER_NOUNS:
            aa_count += 1
    if aa_count / len(words) > 0.02:
        return True

    # --- Signal 2: archaic vocabulary ---
    archaic_count = 0
    for word in words:
        clean = word.strip(".,;:!?()[]{}\"'""''»«–—-/\\").lower()
        if clean in _ARCHAIC_WORDS:
            archaic_count += 1
    if archaic_count / len(words) > 0.01:
        return True

    return False


# Regex patterns for embedded metadata lines (e.g. dkmedier).
_META_PUBLICERET_RE = re.compile(r"^Publiceret:\s*\d{4}-\d{2}-\d{2}")
_META_KATEGORIER_RE = re.compile(r"^Kategorier:\s*")
_META_KILDE_RE = re.compile(r"^Kilde:\s*")


def _extract_metadata_url(text: str) -> str:
    """Extract the URL from a ``Kilde: <url>`` line, if present.

    This is called once per document (before paragraph splitting) so
    that the URL can be stored in the output record.  Returns an empty
    string if no ``Kilde:`` line is found.
    """
    for line in reversed(text.split("\n")):
        stripped = line.strip()
        if _META_KILDE_RE.match(stripped):
            return _META_KILDE_RE.sub("", stripped).strip()
    return ""


def _clean_paragraph(text: str, title: str) -> str:
    """Light cleaning of a paragraph before filtering.

    - Strip the leading ``# Title`` line that Wikipedia articles start with.
    - Strip Folketing (ft) speaker IDs like ``TALER 186:`` that appear
      in parliamentary transcripts.
    - Strip embedded metadata lines (``Publiceret:``, ``Kategorier:``,
      ``Kilde:``) found in some news corpora (e.g. dkmedier).
    - Remove trailing whitespace.
    """
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip top-level headings (# Title) — these are article titles, not content
        if stripped.startswith("# ") and not stripped.startswith("## "):
            continue
        # Strip Folketing speaker IDs (e.g. "TALER 186: Tak.") anywhere in line
        stripped = re.sub(r"TALER\s+\d+:\s*", "", stripped)
        # Skip embedded metadata lines
        if _META_PUBLICERET_RE.match(stripped):
            continue
        if _META_KATEGORIER_RE.match(stripped):
            continue
        if _META_KILDE_RE.match(stripped):
            continue
        cleaned.append(stripped)
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

    # Reject chronological year-dumps (career CVs, election results, etc.)
    if _is_year_heavy(text):
        return False

    # Reject very short blurbs (< 3 sentences) — they make poor chatbot answers
    if _too_few_sentences(text):
        return False

    # Reject encoding corruption / OCR artifacts (non-Latin chars, doubled syllables)
    if _has_encoding_corruption(text):
        return False

    # Reject dialogue-dominated passages
    if _is_quote_heavy(text):
        return False

    # Reject structural/boilerplate content (TOC, program listings)
    if _is_boilerplate(text):
        return False

    # Reject archaic (pre-1948) Danish orthography
    if _is_archaic_danish(text):
        return False

    return True


def filter_paragraph_generic(
    text: str,
    *,
    min_chars: int = 100,
    max_chars: int = 3000,
) -> bool:
    """Return True if the paragraph passes generic quality filters.

    This is a variant of :func:`filter_paragraph` that omits
    Wikipedia-specific checks (section headings, stub detection) and is
    suitable for any plain-text source.
    """
    # Length filters
    if len(text) < min_chars or len(text) > max_chars:
        return False

    if _is_mostly_list(text):
        return False

    if _has_too_many_links(text):
        return False

    if _is_year_heavy(text):
        return False

    if _too_few_sentences(text):
        return False

    # Reject encoding corruption / OCR artifacts (non-Latin chars, doubled syllables)
    if _has_encoding_corruption(text):
        return False

    # Reject dialogue-dominated passages
    if _is_quote_heavy(text):
        return False

    # Reject structural/boilerplate content (TOC, program listings)
    if _is_boilerplate(text):
        return False

    # Reject archaic (pre-1948) Danish orthography
    if _is_archaic_danish(text):
        return False

    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_wikipedia_like(dataset_name: str, subset: str) -> bool:
    """Return True if this dataset uses Wikipedia-style markdown section headings."""
    if "wikipedia" in dataset_name.lower() and not subset:
        return True
    if subset and subset.lower() == "wikipedia":
        return True
    return False


#: Conventional filename for a local source dataset inside a data directory.
_LOCAL_SOURCE_FILE = "source.parquet"


def _load_local_parquet(
    path: Path,
    text_column: str = "text",
    title_column: str = "id",
    url_column: str = "",
) -> Iterator[dict[str, str]]:
    """Yield rows from a local Parquet file in the same format as HuggingFace.

    Each yielded dict has at least a ``text_column`` key.  ``title_column``
    and ``url_column`` are included when they exist in the Parquet schema
    and are not empty strings.
    """
    table = pq.read_table(str(path))
    columns = set(table.column_names)

    for i in range(table.num_rows):
        row: dict[str, str] = {text_column: table[text_column][i].as_py()}
        if title_column and title_column in columns:
            row[title_column] = str(table[title_column][i].as_py())
        if url_column and url_column in columns:
            row[url_column] = str(table[url_column][i].as_py())
        yield row


def extract_paragraphs(
    output_path: Path,
    *,
    dataset_name: str = "oliverkinch/danish_wikipedia",
    subset: str = "",
    source_type: str = "knowledge",
    text_column: str = "text",
    title_column: str = "title",
    url_column: str = "url",
    n_samples: int = 0,
    seed: int = 42,
    min_chars: int = 100,
    max_chars: int = 3000,
    streaming: bool = False,
) -> Path:
    """Extract candidate paragraphs from a dataset.

    The data source is chosen automatically:

    * If ``source.parquet`` exists in the output directory, it is loaded
      directly as a local dataset.
    * Otherwise the dataset is fetched from HuggingFace via
      ``datasets.load_dataset()``.

    Parameters
    ----------
    output_path
        Where to write the JSONL output.
    dataset_name
        HuggingFace dataset identifier (ignored when a local parquet exists).
    subset
        Dataset subset/configuration name (e.g. ``"wikipedia"`` for DynaWord).
        When empty, the default configuration is loaded.
    source_type
        The type of source text (e.g. ``"knowledge"``, ``"speech"``).
        Stored in each output record so that downstream stages can select
        the appropriate prompt templates.
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
        Whether to stream the HuggingFace dataset (ignored for local files).

    Returns
    -------
    Path
        The path to the written JSONL file.
    """
    random.seed(seed)

    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            existing = sum(1 for _ in f)
        if n_samples > 0 and existing >= n_samples:
            print(f"Paragraphs already extracted at {output_path} ({existing} paragraphs). Skipping.")
            return output_path
        elif n_samples == 0 and existing > 0:
            print(f"Paragraphs already extracted at {output_path} ({existing} paragraphs). Skipping.")
            return output_path
        print(f"Found {existing} paragraphs but need more. Re-extracting.")

    # --- Choose data source ---
    local_parquet = output_path.parent / _LOCAL_SOURCE_FILE
    if local_parquet.exists():
        print(f"Loading local dataset from {local_parquet} ...")
        ds = _load_local_parquet(
            local_parquet,
            text_column=text_column,
            title_column=title_column,
            url_column=url_column,
        )
        label = str(local_parquet)
    else:
        label = f"{dataset_name}/{subset}" if subset else dataset_name
        print(f"Loading dataset {label} ...")
        load_kwargs: dict = dict(split="train", streaming=streaming)
        if subset:
            load_kwargs["name"] = subset
        ds = load_dataset(dataset_name, **load_kwargs)

    use_wiki_strategy = _is_wikipedia_like(dataset_name, subset)

    paragraphs: list[Paragraph] = []

    for example in tqdm(ds, desc="Extracting paragraphs"):
        text = example[text_column]
        title = example.get(title_column, "") if title_column else ""
        url = example.get(url_column, "") if url_column else ""

        # Try to extract a URL from embedded "Kilde:" metadata if no
        # explicit url column is available.
        if not url:
            url = _extract_metadata_url(text)

        if use_wiki_strategy:
            sections = _split_sections(text)
        else:
            sections = _split_paragraphs(text, min_chars=min_chars, max_chars=max_chars)

        for heading, body in sections:
            cleaned = _clean_paragraph(body, title)
            if use_wiki_strategy:
                passes = filter_paragraph(cleaned, heading, min_chars=min_chars, max_chars=max_chars)
            else:
                # Generic filtering: skip Wikipedia-specific heading/stub checks
                passes = filter_paragraph_generic(cleaned, min_chars=min_chars, max_chars=max_chars)
            if passes:
                paragraphs.append(
                    Paragraph(
                        text=cleaned,
                        title=title,
                        url=url,
                        section_heading=heading,
                        source_type=source_type,
                    )
                )

    print(f"Extracted {len(paragraphs)} candidate paragraphs from {label}")

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
                "source_type": p.source_type,
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {len(paragraphs)} paragraphs to {output_path}")
    return output_path
