# instruct-bt

Instruction backtranslation for Danish LLM post-training. Extracts non-synthetic paragraphs from text corpora and generates matching instructions via an LLM, producing instruction-response pairs where the response is real human-written text.

## Installation

```bash
uv sync
```

## Configuration

```bash
cp .env.example .env
```

Edit `.env` with your API credentials. The pipeline uses an OpenAI-compatible API (default: Alexandra Institute inference server).

## Usage

The pipeline has four stages:

```bash
# Stage 1: Heuristic extraction — split documents into paragraphs, filter junk
uv run instruct-bt extract

# Stage 2: LLM selection — pick the best chatbot-response passage from each paragraph
uv run instruct-bt select

# Stage 3: Generate instructions for each selected paragraph
uv run instruct-bt generate

# Stage 4: Deduplicate, filter, and format
uv run instruct-bt postprocess

# Or run everything end-to-end
uv run instruct-bt run-all -n 5
```

### DynaWord subsets (HuggingFace)

Process subsets from the `danish-foundation-models/danish-dynaword` dataset:

```bash
# Process a single subset (auto-detects dataset and column names)
uv run instruct-bt run-all --subset wikipedia -n 5 -d data/wikipedia --streaming

# Active subsets: wikipedia, danske-taler, ft, nordjyllandnews, miljoeportalen, ai-aktindsigt
# Each subset maps to a source type (knowledge, speech, news, government)
# which selects appropriate prompt templates.
```

### Process all datasets at once

Use `run-everything` to process every HuggingFace DynaWord subset **and** auto-discovered local parquet datasets in one invocation:

```bash
# Test run — 50 paragraphs per subset
uv run instruct-bt run-everything -n 50 --streaming

# Full scale
uv run instruct-bt run-everything --streaming
```

This iterates over all 6 DynaWord subsets (`wikipedia`, `danske-taler`, `ft`, `nordjyllandnews`, `miljoeportalen`, `ai-aktindsigt`) plus any local dataset discovered by scanning `data/*/source.parquet`. Each subset runs the full four-stage pipeline sequentially.

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `-n` / `--n-samples` | 0 (all) | Max paragraphs per subset |
| `-t` / `--temperature` | 1.0 | Instruction generation temperature |
| `--select-temperature` | 0.3 | Selection stage temperature |
| `--streaming` | off | Stream HuggingFace datasets |
| `--data-root` | `data/` | Root directory for per-subset dirs |

### Local parquet files

To process a local dataset, drop a `source.parquet` file into the data directory:

```bash
# Create the data directory and copy your parquet file
mkdir -p data/dkmedier
cp /path/to/dkmedier.parquet data/dkmedier/source.parquet

# Run — auto-detects local file, no --dataset needed
uv run instruct-bt run-all --subset dkmedier -n 5 -d data/dkmedier
```

The pipeline auto-detects `source.parquet` in the data directory and loads it instead of fetching from HuggingFace. The parquet file should have at least an `id` column and a `text` column.

Embedded metadata lines (`Publiceret:`, `Kategorier:`, `Kilde:`) are automatically stripped during extraction. URLs from `Kilde:` lines are preserved in the output.

### Directory structure

Each dataset (whether from HuggingFace or local) gets its own directory:

```
data/
├── wikipedia/                  # HuggingFace subset (no source.parquet)
│   ├── paragraphs.jsonl
│   ├── selected.jsonl
│   ├── with_instructions.jsonl
│   ├── final.jsonl
│   └── final.parquet
│
├── dkmedier/                   # Local dataset
│   ├── source.parquet          # ← user drops this in
│   ├── paragraphs.jsonl
│   ├── selected.jsonl
│   ├── with_instructions.jsonl
│   ├── final.jsonl
│   └── final.parquet
```

### Key options

```bash
# Sample 10k paragraphs instead of all
uv run instruct-bt extract -n 10000

# Adjust paragraph length bounds (chars)
uv run instruct-bt extract --min-chars 150 --max-chars 2000

# Set LLM temperature (selection uses low temp by default)
uv run instruct-bt select -t 0.3
uv run instruct-bt generate -t 0.7
```

See `--help` on any command for all options.

### Using other HuggingFace datasets

The extract stage accepts any HuggingFace text dataset:

```bash
uv run instruct-bt extract \
    --dataset "some-org/danish-news" \
    --text-column "content" \
    --title-column "" \
    --url-column ""
```

Wikipedia-style datasets (with `##` markdown headings) are auto-detected and split by section. All other datasets use paragraph-based splitting with a 3-tier cascade: blank lines → single newlines → sentence boundaries.

## Output

The final dataset is written to `data/{subset}/final.parquet` (and `final.jsonl`) in chat format:

```json
{
    "prompt_id": "instruct_bt_0",
    "messages": [
        {"role": "user", "content": "<synthetic instruction>"},
        {"role": "assistant", "content": "<real paragraph>"}
    ],
    "title": "...",
    "url": "...",
    "section_heading": "..."
}
```

## Pipeline overview

1. **Extract** — Loads data from HuggingFace or a local `source.parquet`, splits documents into paragraphs, and filters out stubs, link lists, track listings, reference sections, encoding corruption, archaic Danish, boilerplate, and quote-heavy passages.
2. **Select** — For each candidate paragraph, an LLM selects the best self-contained passage that reads naturally as a chatbot response. A verification step ensures the output is a verbatim extract (not a paraphrase). Source-type-aware prompts adapt to knowledge, speech, news, and government texts.
3. **Generate** — For each selected passage, an LLM generates a natural Danish instruction a user would have typed to receive it. Format diversity (short/medium/long) is enforced by randomly selecting one of three format templates per API call.
4. **Postprocess** — Filters refusals, runs MinHash deduplication on both instructions and paragraphs, shuffles, and formats to chat messages.
