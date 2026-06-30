from __future__ import annotations

import asyncio
import re
import unicodedata

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from backend.agents import scaling_expert, substitution_expert
from backend.database import (
    RETRIEVAL_MAX_DISTANCE,
    best_match_distance,
    get_recipe_by_id,
)
from backend.recipe_db import Recipe
from backend.validate_query import (
    QueryClassification,
    classify_query,
)
from backend.vector_store import search_recipe_ids


NON_COOKING_MESSAGE = (
    "🍳 Solo puedo ayudarte con recetas y temas relacionados con la cocina."
)

NO_RECIPES_MESSAGE = (
    "🍳 No encontré recetas en nuestra base que satisfagan tu consulta.\n\n"
    "Probá con otros ingredientes, menos restricciones de tiempo "
    "o criterios nutricionales más flexibles."
)

_last_recipe_by_thread: dict[str, Recipe] = {}

SCALING_PATTERN = re.compile(
    r"(?:adaptar|escalar|ajustar|cambiar|scale|adapt).*?"
    r"(\d+)\s*(?:porcion|porciones|serving|servings)|"
    r"(\d+)\s*(?:porcion(?:es)?|serving(?:s)?)"
    r"\s*(?:en vez|en lugar|instead)|"
    r"para\s*(\d+)\s*(?:personas?|people)",
    re.IGNORECASE,
)

SUBSTITUTION_PATTERN = re.compile(
    r"\b("
    r"no tengo|"
    r"sin |"
    r"sustitu|"
    r"reemplaz|"
    r"alternativa|"
    r"en vez de|"
    r"cambiar el|"
    r"cambiar la|"
    r"don't have|"
    r"do not have|"
    r"without |"
    r"substitut|"
    r"replac|"
    r"alternative|"
    r"instead of"
    r")\b",
    re.IGNORECASE,
)

def extract_filters(message: str) -> dict:
    msg = message.lower()

    filters: dict = {}

    if any(
        word in msg
        for word in (
            "rápid",
            "rapido",
            "rápida",
            "rapida",
            "quick",
            "fast",
            "hungry",
            "hambre",
        )
    ):
        filters["max_total_time"] = 20

    if any(
        word in msg
        for word in (
            "proteína",
            "proteina",
            "muscular",
            "athlete",
            "atleta",
            "high protein",
        )
    ):
        filters["min_protein"] = 30

    if (
        "calor" in msg
        and any(
            word in msg
            for word in (
                "light",
                "low",
                "bajo",
                "baja",
            )
        )
    ):
        filters["max_calories"] = 400

    return filters

def parse_target_servings(message: str) -> int | None:
    match = SCALING_PATTERN.search(message)

    if not match:
        return None

    for value in match.groups():
        if value:
            return int(value)

    return None

def format_directions(directions: str) -> str:
    steps = [
        step.strip()
        for step in re.split(r"\n+", directions)
        if step.strip()
    ]

    if len(steps) <= 1:
        return directions.strip()

    return "\n".join(
        f"{i}. {step}"
        for i, step in enumerate(steps, 1)
    )

def _format_meta_line(recipe: Recipe) -> str | None:
    parts: list[str] = []
    if recipe.total_time_min:
        parts.append(f"⏱️ {recipe.total_time_min} min")
    elif recipe.total_time:
        parts.append(f"⏱️ {recipe.total_time}")
    if recipe.servings:
        parts.append(f"🍽️ {recipe.servings} porciones")
    return " | ".join(parts) if parts else None

def _ingredient_bullets(ingredients: str) -> list[str]:
    text = ingredients.strip()
    if not text:
        return []

    if "\n" in text:
        items = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        items = [part.strip() for part in re.split(r",\s*", text) if part.strip()]

    return [f"- {item}" for item in items]

def format_recipe_from_sql(recipe: Recipe) -> str:
    lines = [f"¡Te recomiendo **{recipe.recipe_name}**! 🍳", ""]

    if recipe.semantic_summary:
        lines.extend([recipe.semantic_summary, ""])

    meta = _format_meta_line(recipe)
    if meta:
        lines.extend([meta, ""])

    lines.extend(["**Ingredientes:**", ""])
    lines.extend(_ingredient_bullets(recipe.ingredients))
    lines.extend(
        [
            "",
            "**Preparación:**",
            "",
            format_directions(recipe.directions),
        ]
    )

    return "\n".join(lines)

async def search_ids_async(query: str, filters: dict) -> tuple[list[int], str | None]:
    ids = await asyncio.to_thread(
        search_recipe_ids,
        query,
        max_total_time=filters.get("max_total_time"),
        min_protein=filters.get("min_protein"),
        max_calories=filters.get("max_calories"),
        k=4,
    )

    if ids:
        return ids, None

    best = await asyncio.to_thread(
        best_match_distance,
        query,
    )

    if (
        best is not None
        and best > RETRIEVAL_MAX_DISTANCE
    ):
        return (
            [],
            (
                f"\n\n_(Distancia del mejor match: {best:.2f}; "
                f"umbral máximo: "
                f"{RETRIEVAL_MAX_DISTANCE})_"
            ),
        )

    return [], None

async def stream_text_chunks(text: str, *, delay_sec: float = 0.06) -> AsyncIterator[str]:
    """Yield the response section by section so the title appears immediately."""
    if not text:
        return

    paragraphs = [paragraph for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        yield text
        return

    buffer = ""
    for index, paragraph in enumerate(paragraphs):
        buffer = paragraph if index == 0 else f"{buffer}\n\n{paragraph}"
        yield buffer
        if delay_sec > 0:
            await asyncio.sleep(delay_sec)

async def stream_query(message: str, thread_id: str) -> AsyncIterator[str]:
    message_es = message.strip()

    if not message_es:
        return

    context = PipelineContext(
        thread_id=thread_id,
        message_es=message_es,
    )

    await pipeline.handle(context)

    if context.response_es:
        async for chunk in stream_text_chunks(context.response_es):
            yield chunk
        return

    yield "🍳 Ocurrió un error procesando tu consulta."

def normalize_query(text: str) -> str:
    """
    Normalize a user query without changing its meaning.

    Operations:
    - Normalize Unicode characters.
    - Remove control characters.
    - Remove emojis and decorative symbols.
    - Collapse repeated punctuation.
    - Collapse multiple spaces.
    - Trim whitespace.
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(
        r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ.,!?;:()/%+\-]",
        " ",
        text,
        flags=re.UNICODE,
    )
    text = re.sub(r"([!?.,;:])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

@dataclass
class PipelineContext:

    thread_id: str

    message_es: str

    classification: QueryClassification | None = None

    filters: dict = field(default_factory=dict)

    recipe_ids: list[int] = field(default_factory=list)

    recipe: Recipe | None = None

    threshold_hint: str | None = None

    response_es: str | None = None

    stop: bool = False

class Handler(ABC):

    def __init__(self):
        self._next: Handler | None = None

    def set_next(self, handler: "Handler") -> "Handler":
        self._next = handler
        return handler

    async def handle(self, context: PipelineContext) -> PipelineContext:
        await self.process(context)

        if context.stop:
            return context

        if self._next is not None:
            return await self._next.handle(context)

        return context

    @abstractmethod
    async def process(self, context: PipelineContext) -> None:
        ...

class QueryClassifierHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        context.classification = await asyncio.to_thread(
            classify_query,
            context.message_es,
        )

        if not context.classification.valid:
            context.response_es = NON_COOKING_MESSAGE
            context.stop = True

class ScalingHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        target_servings = parse_target_servings(context.message_es)

        if target_servings is None:
            return

        if context.thread_id not in _last_recipe_by_thread:
            return

        recipe = _last_recipe_by_thread[context.thread_id]

        current_servings = recipe.servings or 4

        scaled_recipe = await asyncio.to_thread(
            scaling_expert.invoke,
            {
                "recipe_text": recipe.full_text(),
                "current_servings": int(current_servings),
                "target_servings": target_servings,
            },
        )

        context.response_es = (
            f"{scaled_recipe}\n\n"
            f"_Basado en: **{recipe.recipe_name}**_"
        )

        context.stop = True

class SubstitutionHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        if parse_target_servings(context.message_es):
            return

        if not SUBSTITUTION_PATTERN.search(context.message_es):
            return

        response = await asyncio.to_thread(
            substitution_expert.invoke,
            {
                "ingredient": context.message_es,
                "dietary_constraint": "",
            },
        )

        context.response_es = response
        context.stop = True

class RetrievalHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        context.filters = extract_filters(context.message_es)
        context.recipe_ids, context.threshold_hint = await search_ids_async(
            context.message_es,
            context.filters,
        )

class DatabaseHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        if not context.recipe_ids:
            hint = context.threshold_hint or ""
            context.response_es = NO_RECIPES_MESSAGE + hint
            context.stop = True
            return

        recipe = await asyncio.to_thread(get_recipe_by_id, context.recipe_ids[0])

        if recipe is None:
            context.response_es = (
                "⚠️ Integridad de datos: "
                f"id {context.recipe_ids[0]} "
                "está en Chroma pero no en SQLite.\n\n"
                "Ejecutá:\n"
                "python data_preprocessing/ingest.py"
            )
            context.stop = True
            return

        context.recipe = recipe
        _last_recipe_by_thread[context.thread_id] = recipe

class ResponseHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        context.response_es = format_recipe_from_sql(context.recipe)

class NormalizeQueryHandler(Handler):
    """Clean the user query before any LLM or retrieval step."""

    async def process(self, context: PipelineContext) -> None:
        context.message_es = normalize_query(context.message_es)

pipeline = NormalizeQueryHandler()

pipeline \
    .set_next(RetrievalHandler())   \
    .set_next(DatabaseHandler())    \
    .set_next(ResponseHandler())
