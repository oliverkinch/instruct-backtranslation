"""Configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    """Pipeline configuration loaded from environment variables and CLI flags."""

    api_key: str
    base_url: str
    model_name: str
    max_concurrency: int
    temperature: float
    data_dir: Path

    # MinHash deduplication parameters
    minhash_threshold: float = 0.7
    minhash_num_perm: int = 128
    minhash_tokenizer: str = "Qwen/Qwen2.5-7B"


def load_settings(
    data_dir: str | Path = "data",
    temperature: float = 1.0,
) -> Settings:
    """Load settings from .env file and environment variables."""
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get(
        "OPENAI_BASE_URL", "https://inference.projects.alexandrainst.dk/v1"
    )
    model_name = os.environ.get("MODEL_NAME", "qwen-235b")
    max_concurrency = int(os.environ.get("MAX_CONCURRENCY", "50"))

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        max_concurrency=max_concurrency,
        temperature=temperature,
        data_dir=data_path,
    )
