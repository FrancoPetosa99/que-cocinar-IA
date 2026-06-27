"""Strict cooking pipeline: vector search (IDs) + SQLite (full rows)."""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from langchain_core.messages import HumanMessage

from backend.agents import scaling_expert, substitution_expert
from backend.config import get_llm
from backend.database import (
    RETRIEVAL_MAX_DISTANCE,
    best_match_distance,
    get_recipe_by_id,
)
from backend.grounding import RecipeSource, format_grounding_footer, validate_grounding
from backend.recipe_db import Recipe
from backend.translation import translate_to_english, translate_to_spanish
from backend.vector_store import search_recipe_ids

NON_COOKING_MESSAGE = (
    "🍳 Solo puedo ayudarte con recetas y temas relacionados con la cocina."
)

NO_RECIPES_MESSAGE = (
    "🍳 No encontré recetas en nuestra base que satisfagan tu consulta.\n\n"
    "Probá con otros ingredientes, menos restricciones de tiempo o criterios "
    "nutricionales más flexibles."
)

_last_recipe_by_thread: dict[str, Recipe] = {}

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
    if thread_id in _last_recipe_by_thread and (
        SCALING_PATTERN.search(message_es) or SUBSTITUTION_PATTERN.search(message_es)
    ):
        return True
    return bool(COOKING_HINTS.search(message_es) or COOKING_HINTS.search(message_en))


def extract_filters(message_es: str, message_en: str) -> dict:
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


def format_recipe_from_sql(recipe: Recipe) -> str:
    """Build answer from SQLite row (English). Directions come from relational DB only."""
    lines = [recipe.recipe_name, "", "Ingredients:", ""]
    for item in re.split(r",\s*", recipe.ingredients.strip()):
        item = item.strip()
        if item:
            lines.append(f"* {item}")

    lines.extend(["", "Directions:", ""])
    lines.append(_format_directions(recipe.directions))
    lines.append("")
    lines.append(f"Verified source: csv_row_id={recipe.id} | name={recipe.recipe_name}")
    return "\n".join(lines)


def _recipe_to_source(recipe: Recipe) -> RecipeSource:
    return RecipeSource(csv_row_id=recipe.id, recipe_name=recipe.recipe_name)


async def _search_ids_async(message_en: str, filters: dict) -> tuple[list[int], str | None]:
    ids = await asyncio.to_thread(
        search_recipe_ids,
        message_en,
        max_total_time=filters.get("max_total_time"),
        min_protein=filters.get("min_protein"),
        max_calories=filters.get("max_calories"),
        k=4,
    )
    if ids:
        return ids, None

    best = await asyncio.to_thread(best_match_distance, message_en)
    if best is not None and best > RETRIEVAL_MAX_DISTANCE:
        return [], (
            f"\n\n_(Best match distance: {best:.2f}; "
            f"max threshold: {RETRIEVAL_MAX_DISTANCE}. "
            f"Increase RETRIEVAL_MAX_DISTANCE in .env to be more permissive.)_"
        )
    return [], None


def _audit_footer(response_en: str, sources: list[RecipeSource]) -> str:
    return format_grounding_footer(validate_grounding(response_en, sources))


async def _stream_text_chunks(text: str) -> AsyncIterator[str]:
    chunk_size = 40
    if len(text) <= chunk_size:
        yield text
        return
    for i in range(chunk_size, len(text) + chunk_size, chunk_size):
        yield text[:i]


async def _translate_and_stream(english_text: str) -> AsyncIterator[str]:
    spanish = await translate_to_spanish(english_text)
    async for chunk in _stream_text_chunks(spanish):
        yield chunk


async def stream_query(message: str, thread_id: str) -> AsyncIterator[str]:
    message_es = message.strip()
    if not message_es:
        return

    message_en = await translate_to_english(message_es)

    if not is_cooking_query(message_es, message_en, thread_id):
        yield NON_COOKING_MESSAGE
        return

    target_servings = _parse_target_servings(message_es)
    if target_servings and thread_id in _last_recipe_by_thread:
        recipe = _last_recipe_by_thread[thread_id]
        current = recipe.servings or 4
        scaled_en = await asyncio.to_thread(
            scaling_expert.invoke,
            {
                "recipe_text": recipe.full_text(),
                "current_servings": int(current),
                "target_servings": target_servings,
            },
        )
        full_en = (
            f"{scaled_en}\n\n"
            f"Verified source: csv_row_id={recipe.id} | name={recipe.recipe_name}"
        )
        async for chunk in _translate_and_stream(full_en):
            yield chunk
        return

    if SUBSTITUTION_PATTERN.search(message_es) and not target_servings:
        result_en = await asyncio.to_thread(
            substitution_expert.invoke,
            {"ingredient": message_en, "dietary_constraint": ""},
        )
        async for chunk in _translate_and_stream(result_en):
            yield chunk
        return

    filters = extract_filters(message_es, message_en)
    ids, threshold_hint = await _search_ids_async(message_en, filters)

    if not ids:
        hint_es = await translate_to_spanish(threshold_hint) if threshold_hint else ""
        yield NO_RECIPES_MESSAGE + hint_es
        return

    recipe = await asyncio.to_thread(get_recipe_by_id, ids[0])
    if recipe is None:
        yield (
            f"⚠️ Integridad de datos: id {ids[0]} está en Chroma pero no en SQLite. "
            "Ejecutá: python data_preprocessing/ingest.py"
        )
        return

    _last_recipe_by_thread[thread_id] = recipe
    response_en = format_recipe_from_sql(recipe)
    sources = [_recipe_to_source(recipe)]

    footer_es = _audit_footer(response_en, sources)
    body_es = await translate_to_spanish(response_en)
    full_es = f"{body_es}\n\n---\n{footer_es}"

    async for chunk in _stream_text_chunks(full_es):
        yield chunk
