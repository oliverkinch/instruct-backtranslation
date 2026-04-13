"""CLI entry point for the instruct-backtranslation pipeline."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click

from instruct_bt.config import load_settings
from instruct_bt.extract import extract_paragraphs
from instruct_bt.generate import generate_instructions
from instruct_bt.postprocess import postprocess
from instruct_bt.prompts import DYNAWORD_SUBSETS, source_type_for_subset
from instruct_bt.select import select_paragraphs


def _resolve_defaults(
    subset: str,
    dataset: str,
    title_column: str,
    url_column: str,
    data_dir: str,
) -> tuple[str, str, str]:
    """Apply DynaWord / local-parquet column defaults when a subset is given.

    Returns ``(dataset, title_column, url_column)`` — possibly overridden.
    """
    if not subset:
        return dataset, title_column, url_column

    # If a local source.parquet exists, don't auto-switch to DynaWord on HF.
    local_parquet = Path(data_dir) / "source.parquet"
    if not local_parquet.exists():
        if dataset == "oliverkinch/danish_wikipedia":
            dataset = "danish-foundation-models/danish-dynaword"

    # Default column mappings for subset-based datasets
    if title_column == "title":
        title_column = "id"
    if url_column == "url":
        url_column = ""

    return dataset, title_column, url_column


@click.group()
def cli() -> None:
    """instruct-bt: Danish instruction backtranslation from non-synthetic paragraphs."""


@cli.command()
@click.option("--dataset", default="oliverkinch/danish_wikipedia", show_default=True, help="HuggingFace dataset name.")
@click.option("--subset", default="", show_default=True, help="Dataset subset/config name (e.g. 'adl' for DynaWord).")
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
    subset: str,
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
    """Stage 1: Extract and filter paragraphs from a dataset."""
    dataset, title_column, url_column = _resolve_defaults(
        subset, dataset, title_column, url_column, data_dir,
    )

    src_type = source_type_for_subset(subset) if subset else "knowledge"

    settings = load_settings(data_dir=data_dir)
    output_path = settings.data_dir / "paragraphs.jsonl"
    extract_paragraphs(
        output_path=output_path,
        dataset_name=dataset,
        subset=subset,
        source_type=src_type,
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
@click.option("-t", "--temperature", default=0.3, show_default=True, help="LLM sampling temperature.")
def select(data_dir: str, temperature: float) -> None:
    """Stage 2: LLM-based selection of best chatbot-response passages."""
    settings = load_settings(data_dir=data_dir, temperature=temperature)
    asyncio.run(select_paragraphs(settings))


@cli.command()
@click.option("-d", "--data-dir", default="data", show_default=True, help="Directory for intermediate and output files.")
@click.option("-t", "--temperature", default=1.0, show_default=True, help="LLM sampling temperature.")
def generate(data_dir: str, temperature: float) -> None:
    """Stage 3: Generate instructions for each selected paragraph."""
    settings = load_settings(data_dir=data_dir, temperature=temperature)
    asyncio.run(generate_instructions(settings))


@cli.command(name="postprocess")
@click.option("-d", "--data-dir", default="data", show_default=True, help="Directory for intermediate and output files.")
def postprocess_cmd(data_dir: str) -> None:
    """Stage 4: Filter, deduplicate, and format the final dataset."""
    settings = load_settings(data_dir=data_dir)
    postprocess(settings)


@cli.command(name="run-all")
@click.option("--dataset", default="oliverkinch/danish_wikipedia", show_default=True, help="HuggingFace dataset name.")
@click.option("--subset", default="", show_default=True, help="Dataset subset/config name (e.g. 'adl' for DynaWord).")
@click.option("--text-column", default="text", show_default=True, help="Column containing article text.")
@click.option("--title-column", default="title", show_default=True, help="Column containing article title (empty to skip).")
@click.option("--url-column", default="url", show_default=True, help="Column containing source URL (empty to skip).")
@click.option("-n", "--n-samples", default=0, show_default=True, help="Max paragraphs to extract (0 = all).")
@click.option("-d", "--data-dir", default="data", show_default=True, help="Directory for intermediate and output files.")
@click.option("-t", "--temperature", default=1.0, show_default=True, help="LLM sampling temperature for instruction generation.")
@click.option("--select-temperature", default=0.3, show_default=True, help="LLM sampling temperature for paragraph selection.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--min-chars", default=100, show_default=True, help="Minimum paragraph length in characters.")
@click.option("--max-chars", default=3000, show_default=True, help="Maximum paragraph length in characters.")
@click.option("--streaming", is_flag=True, default=False, help="Stream the dataset (for very large datasets).")
def run_all(
    dataset: str,
    subset: str,
    text_column: str,
    title_column: str,
    url_column: str,
    n_samples: int,
    data_dir: str,
    temperature: float,
    select_temperature: float,
    seed: int,
    min_chars: int,
    max_chars: int,
    streaming: bool,
) -> None:
    """Run the full pipeline end-to-end."""
    dataset, title_column, url_column = _resolve_defaults(
        subset, dataset, title_column, url_column, data_dir,
    )

    src_type = source_type_for_subset(subset) if subset else "knowledge"

    click.echo("=== Stage 1: Extract paragraphs ===")
    settings = load_settings(data_dir=data_dir, temperature=temperature)
    output_path = settings.data_dir / "paragraphs.jsonl"
    extract_paragraphs(
        output_path=output_path,
        dataset_name=dataset,
        subset=subset,
        source_type=src_type,
        text_column=text_column,
        title_column=title_column,
        url_column=url_column,
        n_samples=n_samples,
        seed=seed,
        min_chars=min_chars,
        max_chars=max_chars,
        streaming=streaming,
    )

    click.echo("\n=== Stage 2: Select passages (LLM) ===")
    select_settings = load_settings(data_dir=data_dir, temperature=select_temperature)
    asyncio.run(select_paragraphs(select_settings))

    click.echo("\n=== Stage 3: Generate instructions ===")
    asyncio.run(generate_instructions(settings))

    click.echo("\n=== Stage 4: Post-process ===")
    postprocess(settings)

    click.echo(f"\nDone! Final dataset at {settings.data_dir / 'final.parquet'}")


# ------------------------------------------------------------------
# run-everything: process ALL datasets in one invocation
# ------------------------------------------------------------------


def _discover_local_datasets(data_root: Path) -> list[str]:
    """Find local datasets by scanning ``data_root/*/source.parquet``.

    Returns subset names (directory names) that are NOT already covered
    by :data:`DYNAWORD_SUBSETS`.
    """
    local: list[str] = []
    if not data_root.is_dir():
        return local
    for child in sorted(data_root.iterdir()):
        if child.is_dir() and (child / "source.parquet").exists():
            name = child.name
            if name not in DYNAWORD_SUBSETS:
                local.append(name)
    return local


@cli.command(name="run-everything")
@click.option("-n", "--n-samples", default=0, show_default=True, help="Max paragraphs to extract per subset (0 = all).")
@click.option("-t", "--temperature", default=1.0, show_default=True, help="LLM sampling temperature for instruction generation.")
@click.option("--select-temperature", default=0.3, show_default=True, help="LLM sampling temperature for paragraph selection.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--min-chars", default=100, show_default=True, help="Minimum paragraph length in characters.")
@click.option("--max-chars", default=3000, show_default=True, help="Maximum paragraph length in characters.")
@click.option("--streaming", is_flag=True, default=False, help="Stream HuggingFace datasets.")
@click.option("--data-root", default="data", show_default=True, help="Root directory containing per-subset data dirs.")
@click.pass_context
def run_everything(
    ctx: click.Context,
    n_samples: int,
    temperature: float,
    select_temperature: float,
    seed: int,
    min_chars: int,
    max_chars: int,
    streaming: bool,
    data_root: str,
) -> None:
    """Run the full pipeline for ALL datasets (HuggingFace DynaWord + local parquets)."""
    root = Path(data_root)

    # Build the full list: HF subsets first, then auto-discovered local ones.
    local_subsets = _discover_local_datasets(root)
    all_subsets = list(DYNAWORD_SUBSETS) + local_subsets

    click.echo(f"Processing {len(all_subsets)} datasets from {root.resolve()}")
    click.echo(f"  HuggingFace DynaWord: {list(DYNAWORD_SUBSETS)}")
    if local_subsets:
        click.echo(f"  Local parquet:        {local_subsets}")
    click.echo()

    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    for i, subset in enumerate(all_subsets, 1):
        header = f"[{i}/{len(all_subsets)}] {subset}"
        click.echo("=" * 60)
        click.echo(header)
        click.echo("=" * 60)

        subset_dir = str(root / subset)
        try:
            ctx.invoke(
                run_all,
                dataset="oliverkinch/danish_wikipedia",  # resolved by _resolve_defaults
                subset=subset,
                text_column="text",
                title_column="title",
                url_column="url",
                n_samples=n_samples,
                data_dir=subset_dir,
                temperature=temperature,
                select_temperature=select_temperature,
                seed=seed,
                min_chars=min_chars,
                max_chars=max_chars,
                streaming=streaming,
            )
            succeeded.append(subset)
        except Exception as exc:
            click.echo(f"\nERROR processing {subset}: {exc}", err=True)
            failed.append((subset, str(exc)))

        click.echo()

    # --- Summary ---
    click.echo("=" * 60)
    click.echo("SUMMARY")
    click.echo("=" * 60)
    click.echo(f"  Succeeded: {len(succeeded)}/{len(all_subsets)}")
    for name in succeeded:
        click.echo(f"    - {name}")
    if failed:
        click.echo(f"  Failed:    {len(failed)}/{len(all_subsets)}")
        for name, reason in failed:
            click.echo(f"    - {name}: {reason}")
