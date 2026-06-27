#!/usr/bin/env python3
"""Download a Hugging Face model to models/ for local inference."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face instruct model for local use"
    )
    parser.add_argument(
        "model_id",
        nargs="?",
        default=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        help="Hub model id, e.g. Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("HF_MODEL_CACHE_DIR", str(PROJECT_ROOT / "models")),
        help="Directory where the model folder will be saved",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    model_name = args.model_id.split("/")[-1]
    local_dir = Path(args.output_dir) / model_name
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model_id} ...")
    print(f"Destination: {local_dir}")

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )

    print(f"\nDone. Model saved to: {local_dir}")
    print("\nUpdate your .env:")
    print("  LLM_PROVIDER=huggingface")
    print("  HF_BACKEND=local")
    print(f"  LLM_MODEL={args.model_id}")
    print(f"  HF_MODEL_CACHE_DIR={args.output_dir}")
    print("\nThen restart: python frontend/app.py")


if __name__ == "__main__":
    main()
