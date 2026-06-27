"""Strict cooking pipeline: recipes come only from ChromaDB, never invented."""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from backend.agents import scaling_expert, substitution_expert
from backend.config import get_llm
from backend.database import search_recipe_documents
from backend.grounding import RecipeSource, format_grounding_footer, validate_grounding
from backend.translation import translate_to_english, translate_to_spanish

# User-facing messages (Spanish)
NON_COOKING_MESSAGE = (
    "🍳 Solo puedo ayudarte con recetas y temas relacionados con la cocina."
)

NO_RECIPES_MESSAGE = (
    "🍳 No encontré recetas en nuestra base que satisfagan tu consulta.\n\n"
    "Probá con otros ingredientes, menos restricciones de tiempo o criterios "
    "nutricionales más flexibles."
)

_last_recipe_by_thread: dict[str, Document] = {}

COOKING_HINTS = re.compile(
    r"\b(receta|cocinar|cocina|ingrediente|comida|plato|preparar|hornear|freír|freir|"
    r"pollo|arroz|huevo|papa|tomate|queso|pan|carne|pescado|verdura|fruta|"
    r"vegetarian|vegano|proteína|proteina|porcion|porción|sustitu|reemplaz|"
    r"recipe|cook|cooking|ingredient|meal|dish|bake|fry|chicken|rice|egg|"
    r"potato|tomato|cheese|bread|meat|fish|vegetable|fruit|protein|serving|"
    r"substitute|replace)\b",
    re.IGNORECASE,
)

SCALING_PATTERN = re.compile(
    r"(?:adaptar|escalar|ajustar|cambiar|scale|adapt).*?(\d+)\s*(?:porcion|porciones|serving|servings)|"
    r"(\d+)\s*(?:porcion(?:es)?|serving(?:s)?)\s*(?:en vez|en lugar|instead)|"
    r"para\s*(\d+)\s*(?:personas?|people)",
    re.IGNORECASE,
)

SUBSTITUTION_PATTERN = re.compile(
    r"\b(no tengo|sin |sustitu|reemplaz|alternativa|en vez de|cambiar el|cambiar la|"
    r"don't have|do not have|without |substitut|replac|alternative|instead of)\b",
    re.IGNORECASE,
)


def is_cooking_query(message_es: str, message_en: str, thread_id: str) -> bool:
    """Heuristic gate on Spanish and/or English text."""
    if thread_id in _last_recipe_by_thread and (
        SCALING_PATTERN.search(message_es) or SUBSTITUTION_PATTERN.search(message_es)
    ):
        return True
    return bool(COOKING_HINTS.search(message_es) or COOKING_HINTS.search(message_en))


def extract_filters(message_es: str, message_en: str) -> dict:
    """Derive metadata filters from Spanish or English message (no LLM)."""
    msg = f"{message_es} {message_en}".lower()
    filters: dict = {}

    if any(
        w in msg
        for w in [
            "rápid", "rapido", "rápida", "rapida", "pronto", "hambre",
            "quick", "fast", "hurry", "hungry",
        ]
    ):
        filters["max_total_time"] = 20

    if any(
        w in msg
        for w in [
            "proteína", "proteina", "atleta", "muscul", "muscular",
            "high protein", "athlete",
        ]
    ):
        filters["min_protein"] = 30

    if "calor" in msg and any(w in msg for w in ["bajo", "pocas", "light", "baja", "low"]):
        filters["max_calories"] = 400

    return filters


def _parse_target_servings(message_es: str) -> int | None:
    match = SCALING_PATTERN.search(message_es)
    if not match:
        return None
    for group in match.groups():
        if group:
            return int(group)
    return None


def _format_directions(directions_text: str) -> str:
    steps = [s.strip() for s in re.split(r"\n+", directions_text) if s.strip()]
    if len(steps) <= 1:
        return directions_text.strip()
    return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))


def _parse_recipe_sections(content: str) -> tuple[str, str]:
    """Parse ingredients and directions from indexed page_content (ES or EN headers)."""
    for ing_hdr, dir_hdr in [
        ("Ingredientes:\n", "\n\nPreparación:\n"),
        ("Ingredients:\n", "\n\nDirections:\n"),
        ("Ingredients:\n", "\n\nPreparation:\n"),
    ]:
        if ing_hdr in content and dir_hdr in content:
            body = content.split(ing_hdr, 1)[1]
            ingredients, directions = body.split(dir_hdr, 1)
            return ingredients.strip(), directions.strip()

    if "Ingredients:\n" in content:
        return content.split("Ingredients:\n", 1)[1].strip(), ""
    if "Ingredientes:\n" in content:
        return content.split("Ingredientes:\n", 1)[1].strip(), ""
    return content, ""


def format_recipe_strict(doc: Document) -> str:
    """
    Build the answer from database fields in English (dataset language).
    The LLM is NOT used — content cannot be invented.
    """
    meta = doc.metadata
    name = meta.get("recipe_name", "Recipe")
    ingredients, directions = _parse_recipe_sections(doc.page_content)

    lines = [name, "", "Ingredients:", ""]
    for item in re.split(r",\s*", ingredients.strip()):
        item = item.strip()
        if item:
            lines.append(f"* {item}")

    lines.extend(["", "Directions:", ""])
    lines.append(_format_directions(directions))

    row_id = meta.get("csv_row_id", "?")
    lines.append("")
    lines.append(f"Verified source: csv_row_id={row_id} | name={name}")

    return "\n".join(lines)


def _doc_to_source(doc: Document) -> RecipeSource:
    row_id = doc.metadata.get("csv_row_id")
    if row_id is None:
        raise ValueError(
            "Recipe metadata is missing csv_row_id. "
            "Run: python data_preprocessing/ingest.py then restart the app."
        )
    return RecipeSource(
        csv_row_id=int(row_id),
        recipe_name=str(doc.metadata.get("recipe_name", "unknown")),
    )


async def _search_recipes_async(message_en: str, filters: dict) -> tuple[list[Document], str | None]:
    """Run Chroma search in English (worker thread)."""
    docs = await asyncio.to_thread(
        search_recipe_documents,
        message_en,
        max_total_time=filters.get("max_total_time"),
        min_protein=filters.get("min_protein"),
        max_calories=filters.get("max_calories"),
        k=4,
    )
    if docs:
        return docs, None

    from backend.database import RETRIEVAL_MAX_DISTANCE, get_vectorstore

    def _best_distance() -> float | None:
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search_with_score(message_en, k=1)
        return results[0][1] if results else None

    best = await asyncio.to_thread(_best_distance)
    if best is not None and best > RETRIEVAL_MAX_DISTANCE:
        return [], (
            f"\n\n_(Best match distance: {best:.2f}; "
            f"max threshold: {RETRIEVAL_MAX_DISTANCE}. "
            f"Increase RETRIEVAL_MAX_DISTANCE in .env to be more permissive.)_"
        )
    return [], None


def _audit_footer(response_en: str, sources: list[RecipeSource]) -> str:
    validation = validate_grounding(response_en, sources)
    return format_grounding_footer(validation)


async def _stream_text_chunks(text: str) -> AsyncIterator[str]:
    """Yield progressively longer slices for streaming UX."""
    chunk_size = 40
    if len(text) <= chunk_size:
        yield text
        return
    for i in range(chunk_size, len(text) + chunk_size, chunk_size):
        yield text[:i]


async def _translate_and_stream(english_text: str) -> AsyncIterator[str]:
    """LCEL post-step: English response -> Spanish, streamed in chunks."""
    spanish = await translate_to_spanish(english_text)
    async for chunk in _stream_text_chunks(spanish):
        yield chunk


async def stream_query(message: str, thread_id: str) -> AsyncIterator[str]:
    """
    Strict query handler with LCEL translation:
      Spanish user input -> English (pre) -> pipeline -> Spanish (post)
    """
    message_es = message.strip()
    if not message_es:
        return

    # LCEL pre-processing: Spanish -> English for retrieval & English dataset
    message_en = await translate_to_english(message_es)

    if not is_cooking_query(message_es, message_en, thread_id):
        yield NON_COOKING_MESSAGE
        return

    # --- Scaling follow-up ---
    target_servings = _parse_target_servings(message_es)
    if target_servings and thread_id in _last_recipe_by_thread:
        doc = _last_recipe_by_thread[thread_id]
        current = doc.metadata.get("servings") or 4
        scaled_en = await asyncio.to_thread(
            scaling_expert.invoke,
            {
                "recipe_text": doc.page_content,
                "current_servings": int(current),
                "target_servings": target_servings,
            },
        )
        row_id = doc.metadata.get("csv_row_id", "?")
        name = doc.metadata.get("recipe_name", "")
        full_en = f"{scaled_en}\n\nVerified source: csv_row_id={row_id} | name={name}"
        async for chunk in _translate_and_stream(full_en):
            yield chunk
        return

    # --- Substitution ---
    if SUBSTITUTION_PATTERN.search(message_es) and not target_servings:
        result_en = await asyncio.to_thread(
            substitution_expert.invoke,
            {
                "ingredient": message_en,
                "dietary_constraint": "",
            },
        )
        async for chunk in _translate_and_stream(result_en):
            yield chunk
        return

    # --- Main path: English search, English template, Spanish output ---
    filters = extract_filters(message_es, message_en)
    docs, threshold_hint = await _search_recipes_async(message_en, filters)

    if not docs:
        hint_es = ""
        if threshold_hint:
            hint_es = await translate_to_spanish(threshold_hint)
        yield NO_RECIPES_MESSAGE + hint_es
        return

    best = docs[0]
    _last_recipe_by_thread[thread_id] = best
    response_en = format_recipe_strict(best)
    try:
        sources = [_doc_to_source(best)]
    except ValueError as exc:
        yield f"⚠️ {exc}"
        return

    footer_es = _audit_footer(response_en, sources)
    body_es = await translate_to_spanish(response_en)
    full_es = f"{body_es}\n\n---\n{footer_es}"

    async for chunk in _stream_text_chunks(full_es):
        yield chunk


async def stream_query_llm_phrase(message: str, thread_id: str) -> AsyncIterator[str]:
    """Optional: one-line LLM intro in English, then strict recipe, output in Spanish."""
    message_es = message.strip()
    message_en = await translate_to_english(message_es)
    filters = extract_filters(message_es, message_en)
    docs = await asyncio.to_thread(
        search_recipe_documents,
        message_en,
        max_total_time=filters.get("max_total_time"),
        min_protein=filters.get("min_protein"),
        max_calories=filters.get("max_calories"),
        k=4,
    )
    if not docs:
        yield NO_RECIPES_MESSAGE
        return

    best = docs[0]
    _last_recipe_by_thread[thread_id] = best
    body_en = format_recipe_strict(best)

    llm = get_llm(streaming=True)
    prompt = (
        "Write ONE short sentence introducing the recipe. "
        "Do NOT list ingredients or steps. English only.\n"
        f"Recipe: {best.metadata.get('recipe_name')}"
    )
    intro_en = ""
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            intro_en += chunk.content if isinstance(chunk.content, str) else str(chunk.content)

    sources = [_doc_to_source(best)]
    full_en = intro_en + "\n\n" + body_en + "\n\n---\n" + _audit_footer(body_en, sources)
    async for chunk in _translate_and_stream(full_en):
        yield chunk
