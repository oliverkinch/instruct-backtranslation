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

The pipeline has three stages:

```bash
# Stage 1: Extract and filter paragraphs
uv run instruct-bt extract

# Stage 2: Generate instructions for each paragraph
uv run instruct-bt generate

# Stage 3: Deduplicate, filter, and format
uv run instruct-bt postprocess

# Or run everything end-to-end
uv run instruct-bt run-all
```

### Key options

```bash
# Sample 10k paragraphs instead of all
uv run instruct-bt extract -n 10000

# Use a different output directory
uv run instruct-bt run-all -d output/wiki

# Adjust paragraph length bounds (chars)
uv run instruct-bt extract --min-chars 150 --max-chars 2000

# Set LLM temperature
uv run instruct-bt generate -t 0.7
```

See `--help` on any command for all options.

### Using other datasets

The extract stage accepts any HuggingFace text dataset:

```bash
uv run instruct-bt extract \
    --dataset "some-org/danish-news" \
    --text-column "content" \
    --title-column "" \
    --url-column ""
```

The section splitting logic (`_split_sections` in `extract.py`) is Wikipedia-specific (splits on `##` headings). For non-markdown datasets, replace or extend that function.

## Output

The final dataset is written to `data/final.parquet` (and `data/final.jsonl`) in chat format:

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

1. **Extract** — Loads a HuggingFace dataset, splits articles into sections, filters for self-contained informative paragraphs (rejects stubs, link lists, track listings, reference sections, etc.)
2. **Generate** — For each paragraph, calls an LLM to generate the instruction a user would have sent to receive this paragraph as a response. Async with concurrency control and resume support.
3. **Postprocess** — Filters refusals, runs MinHash deduplication on both instructions and paragraphs, formats to chat messages.
