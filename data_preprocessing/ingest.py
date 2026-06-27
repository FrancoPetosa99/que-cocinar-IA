#!/usr/bin/env python3
"""
Ingest recipes: CSV -> SQLite (full rows) + Chroma (search index only).

Usage:
  python data_preprocessing/ingest.py              # both phases
  python data_preprocessing/ingest.py --relational-only
  python data_preprocessing/ingest.py --vector-only
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import CHROMA_DIR, COLLECTION_NAME, PROJECT_ROOT as ROOT, SQLITE_PATH  # noqa: E402
from backend.recipe_db import iter_recipes_for_indexing, load_csv_to_sqlite  # noqa: E402
from backend.vector_store import recipe_to_vector_document  # noqa: E402

CSV_PATH = ROOT / "data" / "recipes.csv"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 500


def ingest_relational() -> int:
    print(f"Phase 1: CSV -> SQLite ({SQLITE_PATH})")
    count = load_csv_to_sqlite(CSV_PATH, Path(SQLITE_PATH))
    print(f"  Inserted {count:,} recipes")
    return count


def ingest_vector() -> None:
    print(f"Phase 2: SQLite -> Chroma ({CHROMA_DIR})")
    recipes = iter_recipes_for_indexing()
    documents = [recipe_to_vector_document(r) for r in recipes]
    print(f"  Documents (name + ingredients only): {len(documents):,}")
    print(f"  Sample page_content length: {len(documents[0].page_content)} chars")
    assert "Directions" not in documents[0].page_content

    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        print(f"  Removing old {chroma_path}...")
        shutil.rmtree(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = None
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(chroma_path),
            )
        else:
            vectorstore.add_documents(batch)
        print(f"  Indexed {min(i + BATCH_SIZE, len(documents)):,} / {len(documents):,}")

    sample = vectorstore.similarity_search("chicken", k=1)[0]
    assert "csv_row_id" in sample.metadata
    assert "Directions" not in sample.page_content
    print(f"  Verify: csv_row_id={sample.metadata['csv_row_id']} | {sample.metadata['recipe_name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest recipes into SQLite + Chroma")
    parser.add_argument("--relational-only", action="store_true")
    parser.add_argument("--vector-only", action="store_true")
    args = parser.parse_args()

    if args.relational_only:
        ingest_relational()
    elif args.vector_only:
        ingest_vector()
    else:
        ingest_relational()
        ingest_vector()

    print("\nDone. Restart the app: python frontend/app.py")


if __name__ == "__main__":
    main()
