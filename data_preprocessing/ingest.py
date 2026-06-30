"""CSV → SQLite + Chroma ingest entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def main() -> None:
    parser = argparse.ArgumentParser(description="Index recipes into SQLite and ChromaDB.")
    parser.add_argument(
        "--relational-only",
        action="store_true",
        help="Only build SQLite from recipes.csv + enriched_recipes_spanish.csv",
    )
    parser.add_argument(
        "--vector-only",
        action="store_true",
        help="Only rebuild ChromaDB from enriched_recipes_spanish.csv",
    )
    args = parser.parse_args()

    python = sys.executable

    if args.relational_only and args.vector_only:
        parser.error("Use only one of --relational-only or --vector-only.")

    if args.relational_only:
        subprocess.check_call([python, str(SCRIPTS / "ingest_relational_db.py")])
        return

    if args.vector_only:
        subprocess.check_call([python, str(SCRIPTS / "ingest_vectorial_db.py")])
        return

    subprocess.check_call([python, str(SCRIPTS / "ingest_relational_db.py")])
    subprocess.check_call([python, str(SCRIPTS / "ingest_vectorial_db.py")])


if __name__ == "__main__":
    main()
