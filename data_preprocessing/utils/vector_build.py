"""ChromaDB index building for ingestion pipelines."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from data_preprocessing.utils.db_connections import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    iter_recipe_rows,
)


def _embedding_text(recipe_name: str, ingredients: str) -> str:
    return f"{recipe_name}\n\nIngredients:\n{ingredients}"


def recipe_row_to_document(row: sqlite3.Row | dict[str, Any]) -> Document:
    """Build a slim Chroma document: embed name+ingredients; metadata for filters."""
    if isinstance(row, sqlite3.Row):
        data = dict(row)
    else:
        data = row

    recipe_name = str(data["recipe_name"])
    ingredients = str(data["ingredients"])
    metadata: dict[str, Any] = {
        "csv_row_id": int(data["id"]),
        "recipe_name": recipe_name,
    }
    optional = {
        "prep_time_min": data.get("prep_time_min"),
        "cook_time_min": data.get("cook_time_min"),
        "total_time_min": data.get("total_time_min"),
        "servings": data.get("servings"),
        "rating": data.get("rating"),
        "cuisine_path": data.get("cuisine_path") or "",
        "calories": data.get("calories"),
        "protein_g": data.get("protein_g"),
        "carbs_g": data.get("carbs_g"),
        "fat_g": data.get("fat_g"),
    }
    for key, val in optional.items():
        if val is not None and val != "":
            metadata[key] = val

    return Document(
        page_content=_embedding_text(recipe_name, ingredients),
        metadata=metadata,
    )


def rows_to_documents(rows: list[sqlite3.Row]) -> list[Document]:
    return [recipe_row_to_document(row) for row in rows]


def build_chroma_index(
    documents: list[Document],
    *,
    chroma_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_model: str | None = None,
    batch_size: int = 500,
) -> Chroma:
    """Wipe and rebuild the Chroma collection from documents."""
    chroma_path = Path(chroma_dir or CHROMA_DIR)
    name = collection_name or COLLECTION_NAME
    model = embedding_model or EMBEDDING_MODEL

    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=model)
    vectorstore = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=name,
                persist_directory=str(chroma_path),
            )
        else:
            vectorstore.add_documents(batch)
        print(f"  Indexed {min(i + batch_size, len(documents)):,} / {len(documents):,}")

    if vectorstore is None:
        raise ValueError("No documents provided for Chroma indexing.")
    return vectorstore


def build_chroma_index_from_sqlite(
    db_path: str | Path | None = None,
    **kwargs: Any,
) -> Chroma:
    """Read SQLite rows and build the Chroma index."""
    rows = iter_recipe_rows(db_path)
    documents = rows_to_documents(rows)
    return build_chroma_index(documents, **kwargs)
