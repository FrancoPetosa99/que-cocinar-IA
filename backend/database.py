"""ChromaDB initialization, retriever setup, and metadata-filtered search."""

from __future__ import annotations

import os
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from backend.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    get_embeddings,
    validate_chroma_exists,
)

_vectorstore: Chroma | None = None

# Chroma returns a DISTANCE per hit (not cosine similarity). Lower = more similar.
# Default collection metric is L2. Typical ranges on this dataset:
#   ~0.7–1.0  good match | ~1.2–1.5  weak | >1.6  poor / irrelevant
# Env: RETRIEVAL_MAX_DISTANCE (alias: RETRIEVAL_SCORE_THRESHOLD for backward compat)
RETRIEVAL_MAX_DISTANCE = float(
    os.getenv(
        "RETRIEVAL_MAX_DISTANCE",
        os.getenv("RETRIEVAL_SCORE_THRESHOLD", "1.35"),
    )
)


def get_vectorstore() -> Chroma:
    """Open (or create a cached handle to) the persisted Chroma collection."""
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
    except Exception as exc:
        raise ConnectionError(
            f"No se pudo conectar a ChromaDB en '{CHROMA_DIR}': {exc}"
        ) from exc

    return _vectorstore


def _build_where_filter(
    *,
    max_total_time: int | None = None,
    min_protein: float | None = None,
    max_calories: float | None = None,
) -> dict[str, Any] | None:
    """Build a Chroma-compatible metadata filter from optional constraints."""
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


def get_retriever(
    *,
    filters: dict[str, Any] | None = None,
    k: int = 4,
) -> VectorStoreRetriever:
    """Return a retriever, optionally with a Chroma metadata filter."""
    search_kwargs: dict[str, Any] = {"k": k}
    if filters:
        search_kwargs["filter"] = filters

    return get_vectorstore().as_retriever(search_kwargs=search_kwargs)


def search_recipe_documents(
    query: str,
    *,
    max_total_time: int | None = None,
    min_protein: float | None = None,
    max_calories: float | None = None,
    k: int = 4,
    score_threshold: float | None = None,
) -> list[Document]:
    """
    Strict search: returns only documents from ChromaDB that pass relevance threshold.

    No fallback to unfiltered search when metadata filters return nothing.
    """
    threshold = score_threshold if score_threshold is not None else RETRIEVAL_MAX_DISTANCE
    where = _build_where_filter(
        max_total_time=max_total_time,
        min_protein=min_protein,
        max_calories=max_calories,
    )

    vectorstore = get_vectorstore()
    kwargs: dict[str, Any] = {"k": k}
    if where:
        kwargs["filter"] = where

    results = vectorstore.similarity_search_with_score(query, **kwargs)

    if not results:
        return []

    docs = [doc for doc, score in results if score <= threshold]
    return docs


def _format_recipe(doc: Document) -> str:
    """Format a single recipe document for the LLM context."""
    meta = doc.metadata
    row_id = meta.get("csv_row_id", "?")
    lines = [
        f"[csv_row_id={row_id}] **{meta.get('recipe_name', 'Receta sin nombre')}**",
        f"Tiempo total: {meta.get('total_time_min', '?')} min | "
        f"Porciones: {meta.get('servings', '?')} | "
        f"Rating: {meta.get('rating', 'N/A')}",
    ]

    nutrition_parts = []
    for key, label in [
        ("calories", "cal"),
        ("protein_g", "proteína"),
        ("carbs_g", "carbos"),
        ("fat_g", "grasa"),
    ]:
        if key in meta and meta[key] is not None:
            nutrition_parts.append(f"{label}: {meta[key]}")
    if nutrition_parts:
        lines.append("Nutrición: " + ", ".join(nutrition_parts))

    if meta.get("cuisine_path"):
        lines.append(f"Cocina: {meta['cuisine_path']}")

    lines.append("")
    lines.append(doc.page_content)
    return "\n".join(lines)


def search_recipes(
    query: str,
    *,
    max_total_time: int | None = None,
    min_protein: float | None = None,
    max_calories: float | None = None,
    k: int = 4,
) -> str:
    """
    Search recipes by semantic query with optional metadata filters.

    Returns a formatted string of top-k matches for LLM consumption.
    """
    docs = search_recipe_documents(
        query,
        max_total_time=max_total_time,
        min_protein=min_protein,
        max_calories=max_calories,
        k=k,
    )

    if not docs:
        return "No se encontraron recetas que coincidan con la búsqueda."

    formatted = [_format_recipe(doc) for doc in docs]
    return "\n\n---\n\n".join(formatted)


def get_csv_row_preview(csv_row_id: int) -> str:
    """Return key fields from data/recipes.csv for audit/testing."""
    import pandas as pd

    from backend.config import PROJECT_ROOT

    csv_path = PROJECT_ROOT / "data" / "recipes.csv"
    if not csv_path.exists():
        return f"CSV no encontrado: {csv_path}"

    df = pd.read_csv(csv_path)
    id_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
    match = df[df[id_col] == csv_row_id]
    if match.empty:
        return f"No existe fila con csv_row_id={csv_row_id}"

    row = match.iloc[0]
    return (
        f"csv_row_id={csv_row_id}\n"
        f"recipe_name={row.get('recipe_name')}\n"
        f"ingredients={str(row.get('ingredients', ''))[:200]}...\n"
        f"directions={str(row.get('directions', ''))[:200]}..."
    )
