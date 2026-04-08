"""CLI entry point for the instruct-backtranslation pipeline."""

from __future__ import annotations

import asyncio

import click

from instruct_bt.config import load_settings
from instruct_bt.extract import extract_paragraphs
from instruct_bt.generate import generate_instructions
from instruct_bt.postprocess import postprocess


@click.group()
def cli() -> None:
    """instruct-bt: Danish instruction backtranslation from non-synthetic paragraphs."""


@cli.command()
@click.option("--dataset", default="oliverkinch/danish_wikipedia", show_default=True, help="HuggingFace dataset name.")
@click.option("--text-column", default="text", show_default=True, help="Column containing article text.")
@click.option("--title-column", default="title", show_default=True, help="Column containing article title (empty to skip).")
@click.option("--url-column", default="url", show_default=True, help="Column containing source URL (empty to skip).")
@click.option("-n", "--n-samples", default=0, show_default=True, help="Max paragraphs to extract (0 = all).")
@click.option("-d", "--data-dir", default="data", show_default=True, help="Directory for intermediate and output files.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--min-chars", default=100, show_default=True, help="Minimum paragraph length in characters.")
@click.option("--max-chars", default=3000, show_default=True, help="Maximum paragraph length in characters.")
@click.option("--streaming", is_flag=True, default=False, help="Stream the dataset (for very large datasets).")
def extract(
    dataset: str,
    text_column: str,
    title_column: str,
    url_column: str,
    n_samples: int,
    data_dir: str,
    seed: int,
    min_chars: int,
    max_chars: int,
    streaming: bool,
) -> None:
    """Stage 1: Extract and filter paragraphs from a HuggingFace dataset."""
    settings = load_settings(data_dir=data_dir)
    output_path = settings.data_dir / "paragraphs.jsonl"
    extract_paragraphs(
        output_path=output_path,
        dataset_name=dataset,
        text_column=text_column,
        title_column=title_column,
        url_column=url_column,
        n_samples=n_samples,
        seed=seed,
        min_chars=min_chars,
        max_chars=max_chars,
        streaming=streaming,
    )


@cli.command()
@click.option("-d", "--data-dir", default="data", show_default=True, help="Directory for intermediate and output files.")
@click.option("-t", "--temperature", default=1.0, show_default=True, help="LLM sampling temperature.")
def generate(data_dir: str, temperature: float) -> None:
    """Stage 2: Generate instructions for each extracted paragraph."""
    settings = load_settings(data_dir=data_dir, temperature=temperature)
    asyncio.run(generate_instructions(settings))


@cli.command(name="postprocess")
@click.option("-d", "--data-dir", default="data", show_default=True, help="Directory for intermediate and output files.")
def postprocess_cmd(data_dir: str) -> None:
    """Stage 3: Filter, deduplicate, and format the final dataset."""
    settings = load_settings(data_dir=data_dir)
    postprocess(settings)


@cli.command(name="run-all")
@click.option("--dataset", default="oliverkinch/danish_wikipedia", show_default=True, help="HuggingFace dataset name.")
@click.option("--text-column", default="text", show_default=True, help="Column containing article text.")
@click.option("--title-column", default="title", show_default=True, help="Column containing article title (empty to skip).")
@click.option("--url-column", default="url", show_default=True, help="Column containing source URL (empty to skip).")
@click.option("-n", "--n-samples", default=0, show_default=True, help="Max paragraphs to extract (0 = all).")
@click.option("-d", "--data-dir", default="data", show_default=True, help="Directory for intermediate and output files.")
@click.option("-t", "--temperature", default=1.0, show_default=True, help="LLM sampling temperature.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--min-chars", default=100, show_default=True, help="Minimum paragraph length in characters.")
@click.option("--max-chars", default=3000, show_default=True, help="Maximum paragraph length in characters.")
@click.option("--streaming", is_flag=True, default=False, help="Stream the dataset (for very large datasets).")
def run_all(
    dataset: str,
    text_column: str,
    title_column: str,
    url_column: str,
    n_samples: int,
    data_dir: str,
    temperature: float,
    seed: int,
    min_chars: int,
    max_chars: int,
    streaming: bool,
) -> None:
    """Run the full pipeline end-to-end."""
    settings = load_settings(data_dir=data_dir, temperature=temperature)

    click.echo("=== Stage 1: Extract paragraphs ===")
    output_path = settings.data_dir / "paragraphs.jsonl"
    extract_paragraphs(
        output_path=output_path,
        dataset_name=dataset,
        text_column=text_column,
        title_column=title_column,
        url_column=url_column,
        n_samples=n_samples,
        seed=seed,
        min_chars=min_chars,
        max_chars=max_chars,
        streaming=streaming,
    )

    click.echo("\n=== Stage 2: Generate instructions ===")
    asyncio.run(generate_instructions(settings))

    click.echo("\n=== Stage 3: Post-process ===")
    postprocess(settings)

    click.echo(f"\nDone! Final dataset at {settings.data_dir / 'final.parquet'}")
