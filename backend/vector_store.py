"""ChromaDB vector index — search only (name + ingredients). Returns recipe IDs."""

from __future__ import annotations

import os
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    get_embeddings,
    validate_chroma_exists,
)
from backend.recipe_db import Recipe

_vectorstore: Chroma | None = None

RETRIEVAL_MAX_DISTANCE = float(
    os.getenv(
        "RETRIEVAL_MAX_DISTANCE",
        os.getenv("RETRIEVAL_SCORE_THRESHOLD", "1.35"),
    )
)


def reset_vectorstore() -> None:
    global _vectorstore
    _vectorstore = None


def _validate_collection(store: Chroma) -> None:
    try:
        count = store._collection.count()  # noqa: SLF001
        if count == 0:
            raise ValueError("La colección ChromaDB está vacía.")
        peek = store._collection.peek(1)  # noqa: SLF001
        metas = peek.get("metadatas") or []
        if not metas or "csv_row_id" not in (metas[0] or {}):
            raise ValueError(
                "El índice vectorial no tiene csv_row_id. "
                "Ejecutá: python data_preprocessing/ingest.py"
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "No se pudo validar chroma_db. Ejecutá: python data_preprocessing/ingest.py"
        ) from exc


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    validate_chroma_exists()
    try:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        _validate_collection(_vectorstore)
    except Exception as exc:
        _vectorstore = None
        if isinstance(exc, ValueError):
            raise
        raise ConnectionError(
            f"No se pudo conectar a ChromaDB en '{CHROMA_DIR}': {exc}"
        ) from exc
    return _vectorstore


def recipe_to_vector_document(recipe: Recipe) -> Document:
    """Build a slim Chroma document: embed name+ingredients; metadata for filters."""
    metadata: dict[str, Any] = {
        "csv_row_id": recipe.id,
        "recipe_name": recipe.recipe_name,
    }
    optional = {
        "prep_time_min": recipe.prep_time_min,
        "cook_time_min": recipe.cook_time_min,
        "total_time_min": recipe.total_time_min,
        "servings": recipe.servings,
        "rating": recipe.rating,
        "cuisine_path": recipe.cuisine_path or "",
        "calories": recipe.calories,
        "protein_g": recipe.protein_g,
        "carbs_g": recipe.carbs_g,
        "fat_g": recipe.fat_g,
    }
    for key, val in optional.items():
        if val is not None and val != "":
            metadata[key] = val

    return Document(page_content=recipe.embedding_text(), metadata=metadata)


def _build_where_filter(
    *,
    max_total_time: int | None = None,
    min_protein: float | None = None,
    max_calories: float | None = None,
) -> dict[str, Any] | None:
    conditions: list[dict[str, Any]] = []
    if max_total_time is not None:
        conditions.append({"total_time_min": {"$lte": int(max_total_time)}})
    if min_protein is not None:
        conditions.append({"protein_g": {"$gte": float(min_protein)}})
    if max_calories is not None:
        conditions.append({"calories": {"$lte": float(max_calories)}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search_recipe_ids_with_scores(
    query: str,
    *,
    max_total_time: int | None = None,
    min_protein: float | None = None,
    max_calories: float | None = None,
    k: int = 4,
    score_threshold: float | None = None,
) -> list[tuple[int, float]]:
    """
    Semantic search in Chroma. Returns (csv_row_id, distance) pairs.
    Lower distance = more similar.
    """
    threshold = score_threshold if score_threshold is not None else RETRIEVAL_MAX_DISTANCE
    where = _build_where_filter(
        max_total_time=max_total_time,
        min_protein=min_protein,
        max_calories=max_calories,
    )

    kwargs: dict[str, Any] = {"k": k}
    if where:
        kwargs["filter"] = where

    results = get_vectorstore().similarity_search_with_score(query, **kwargs)
    ids: list[tuple[int, float]] = []
    seen: set[int] = set()

    for doc, score in results:
        if score > threshold:
            continue
        row_id = doc.metadata.get("csv_row_id")
        if row_id is None:
            continue
        rid = int(row_id)
        if rid not in seen:
            seen.add(rid)
            ids.append((rid, score))
    return ids


def search_recipe_ids(
    query: str,
    *,
    max_total_time: int | None = None,
    min_protein: float | None = None,
    max_calories: float | None = None,
    k: int = 4,
    score_threshold: float | None = None,
) -> list[int]:
    """Return ordered recipe IDs from vector search (no full recipe data)."""
    return [
        rid
        for rid, _ in search_recipe_ids_with_scores(
            query,
            max_total_time=max_total_time,
            min_protein=min_protein,
            max_calories=max_calories,
            k=k,
            score_threshold=score_threshold,
        )
    ]


def best_match_distance(query: str) -> float | None:
    """Top-1 distance regardless of threshold (for debug hints)."""
    results = get_vectorstore().similarity_search_with_score(query, k=1)
    return results[0][1] if results else None
