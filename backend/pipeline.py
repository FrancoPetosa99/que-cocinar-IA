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
from backend.grounding import (
    RecipeSource,
    format_grounding_footer,
    validate_grounding,
)
from backend.recipe_db import Recipe
from backend.translation import (
    present_recipe_in_spanish,
    translate_to_english,
    translate_to_spanish,
)
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

def extract_filters(message_es: str, message_en: str) -> dict:
    msg = f"{message_es} {message_en}".lower()

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

def format_recipe_from_sql(recipe: Recipe) -> str:
    lines = [
        recipe.recipe_name,
        "",
        "Ingredients:",
        "",
    ]

    for ingredient in re.split(
        r",\s*",
        recipe.ingredients.strip(),
    ):
        if ingredient.strip():
            lines.append(f"* {ingredient.strip()}")

    lines.extend(
        [
            "",
            "Directions:",
            "",
            format_directions(recipe.directions),
            "",
            (
                f"Verified source: "
                f"csv_row_id={recipe.id} | "
                f"name={recipe.recipe_name}"
            ),
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
                f"\n\n_(Best match distance: {best:.2f}; "
                f"max threshold: "
                f"{RETRIEVAL_MAX_DISTANCE})_"
            ),
        )

    return [], None

async def stream_text_chunks(text: str) -> AsyncIterator[str]:
    chunk_size = 40

    if len(text) <= chunk_size:
        yield text
        return

    for i in range(chunk_size, len(text) + chunk_size, chunk_size):
        yield text[:i]

async def stream_query(message: str, thread_id: str) -> AsyncIterator[str]:
    message_es = message.strip()

    if not message_es:
        return

    context = PipelineContext(
        thread_id=thread_id,
        message_es=message_es,
    )

    # Ejecutar pipeline completo
    await pipeline.handle(context)

    # Si algún handler ya resolvió la respuesta, devolverla
    if context.response_es:

        async for chunk in stream_text_chunks(
            context.response_es
        ):
            yield chunk

        return

    # Fallback de seguridad (por si algo falló)
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

    # Normalize Unicode representation
    text = unicodedata.normalize("NFKC", text)

    # Replace tabs/newlines with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove emojis and most symbols while preserving letters,
    # numbers and common punctuation.
    text = re.sub(
        r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ.,!?;:()/%+\-]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # Collapse repeated punctuation
    text = re.sub(r"([!?.,;:])\1+", r"\1", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

@dataclass
class PipelineContext:

    thread_id: str

    message_es: str

    message_en: str | None = None

    classification: QueryClassification | None = None

    filters: dict = field(default_factory=dict)

    recipe_ids: list[int] = field(default_factory=list)

    recipe: Recipe | None = None

    threshold_hint: str | None = None

    response_en: str | None = None

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

class TranslationHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        context.message_en = await translate_to_english(context.message_es)

class QueryClassifierHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        context.classification = await asyncio.to_thread(
            classify_query,
            context.message_en,
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

        response_en = (
            f"{scaled_recipe}\n\n"
            f"Verified source: "
            f"csv_row_id={recipe.id} | "
            f"name={recipe.recipe_name}"
        )

        context.response_es = await translate_to_spanish(response_en)

        context.stop = True

class SubstitutionHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        if parse_target_servings(context.message_es):
            return

        if not SUBSTITUTION_PATTERN.search(context.message_es):
            return

        response_en = await asyncio.to_thread(
            substitution_expert.invoke,
            {
                "ingredient": context.message_en,
                "dietary_constraint": "",
            },
        )

        context.response_es = await translate_to_spanish(response_en)

        context.stop = True

class RetrievalHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        context.filters = extract_filters(context.message_es, context.message_en)
        (context.recipe_ids, context.threshold_hint, ) = await search_ids_async(context.message_en, context.filters)

class DatabaseHandler(Handler):

    async def process(self, context: PipelineContext) -> None:
        if not context.recipe_ids:
            hint = ""

            if context.threshold_hint:
                hint = await translate_to_spanish(context.threshold_hint)

            context.response_es = (NO_RECIPES_MESSAGE + hint)

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
        context.response_en = format_recipe_from_sql(context.recipe)

        sources = [
            RecipeSource(
                csv_row_id=context.recipe.id,
                recipe_name=context.recipe.recipe_name,
            )
        ]

        footer = format_grounding_footer(
            validate_grounding(
                context.response_en,
                sources,
            )
        )

        body = await present_recipe_in_spanish(
            user_query=context.message_es,
            recipe_name=context.recipe.recipe_name,
            ingredients=context.recipe.ingredients,
            directions=context.recipe.directions,
            servings=context.recipe.servings,
            prep_time=context.recipe.prep_time,
            total_time=context.recipe.total_time,
        )

        context.response_es = (
            f"{body}\n\n"
            f"---\n"
            f"{footer}"
        )

class NormalizeQueryHandler(Handler):
    """
    Clean the user query before any LLM or retrieval step.
    """

    async def process(
        self,
        context: PipelineContext,
    ) -> None:

        context.message_es = normalize_query(
            context.message_es
        )

pipeline = NormalizeQueryHandler()

pipeline \
    .set_next(TranslationHandler()) \
    .set_next(QueryClassifierHandler()) \
    .set_next(ScalingHandler()) \
    .set_next(SubstitutionHandler()) \
    .set_next(RetrievalHandler()) \
    .set_next(DatabaseHandler()) \
    .set_next(ResponseHandler())