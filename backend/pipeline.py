"""Strict cooking pipeline: recipes come only from ChromaDB, never invented."""

from __future__ import annotations

import re
from typing import AsyncIterator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from backend.agents import scaling_expert, substitution_expert
from backend.config import get_llm
from backend.database import search_recipe_documents
from backend.grounding import RecipeSource, format_grounding_footer, validate_grounding

NON_COOKING_MESSAGE = (
    "🍳 Solo puedo ayudarte con recetas y temas relacionados con la cocina."
)

NO_RECIPES_MESSAGE = (
    "🍳 No encontré recetas en nuestra base que satisfagan tu consulta.\n\n"
    "Probá con otros ingredientes, menos restricciones de tiempo o criterios "
    "nutricionales más flexibles."
)

# Last recipe shown per session (for scaling follow-ups)
_last_recipe_by_thread: dict[str, Document] = {}

COOKING_HINTS = re.compile(
    r"\b(receta|cocinar|cocina|ingrediente|comida|plato|preparar|hornear|freír|freir|"
    r"pollo|arroz|huevo|papa|tomate|queso|pan|carne|pescado|verdura|fruta|"
    r"vegetarian|vegano|proteína|proteina|porcion|porción|sustitu|reemplaz)\b",
    re.IGNORECASE,
)

SCALING_PATTERN = re.compile(
    r"(?:adaptar|escalar|ajustar|cambiar).*?(\d+)\s*porcion|"
    r"(\d+)\s*porcion(?:es)?\s*(?:en vez|en lugar|instead)|"
    r"para\s*(\d+)\s*personas?",
    re.IGNORECASE,
)

SUBSTITUTION_PATTERN = re.compile(
    r"\b(no tengo|sin |sustitu|reemplaz|alternativa|en vez de|cambiar el|cambiar la)\b",
    re.IGNORECASE,
)


def is_cooking_query(message: str, thread_id: str) -> bool:
    """Heuristic gate: cooking topic or follow-up on a recipe in session."""
    if thread_id in _last_recipe_by_thread and (
        SCALING_PATTERN.search(message) or SUBSTITUTION_PATTERN.search(message)
    ):
        return True
    return bool(COOKING_HINTS.search(message))


def extract_filters(message: str) -> dict:
    """Derive metadata filters from the user message (no LLM)."""
    msg = message.lower()
    filters: dict = {}

    if any(
        w in msg
        for w in ["rápid", "rapido", "rápida", "rapida", "pronto", "hambre", "quick", "fast"]
    ):
        filters["max_total_time"] = 20

    if any(
        w in msg
        for w in ["proteína", "proteina", "atleta", "muscul", "muscular", "high protein"]
    ):
        filters["min_protein"] = 30

    if "calor" in msg and any(w in msg for w in ["bajo", "pocas", "light", "baja"]):
        filters["max_calories"] = 400

    return filters


def _parse_target_servings(message: str) -> int | None:
    match = SCALING_PATTERN.search(message)
    if not match:
        return None
    for group in match.groups():
        if group:
            return int(group)
    return None


def _format_directions(directions_text: str) -> str:
    """Turn CSV directions into a numbered list without changing content."""
    steps = [s.strip() for s in re.split(r"\n+", directions_text) if s.strip()]
    if len(steps) <= 1:
        return directions_text.strip()
    return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))


def format_recipe_strict(doc: Document) -> str:
    """
    Build the user-facing answer directly from database fields.
    The LLM is NOT used — content cannot be invented.
    """
    meta = doc.metadata
    name = meta.get("recipe_name", "Receta")
    content = doc.page_content

    # page_content is "name\n\nIngredientes:\n...\n\nPreparación:\n..."
    ingredients = ""
    directions = ""
    if "Ingredientes:\n" in content and "\n\nPreparación:\n" in content:
        body = content.split("\n\nIngredientes:\n", 1)[1]
        ingredients, directions = body.split("\n\nPreparación:\n", 1)
    else:
        ingredients = content
        directions = ""

    lines = [name, "", "Ingredientes:", ""]
    for item in re.split(r",\s*", ingredients.strip()):
        item = item.strip()
        if item:
            lines.append(f"* {item}")

    lines.extend(["", "Preparación:", ""])
    lines.append(_format_directions(directions))

    row_id = meta.get("csv_row_id", "?")
    lines.append("")
    lines.append(f"Fuente verificada: csv_row_id={row_id} | nombre={name}")

    return "\n".join(lines)


def _doc_to_source(doc: Document) -> RecipeSource:
    return RecipeSource(
        csv_row_id=int(doc.metadata["csv_row_id"]),
        recipe_name=str(doc.metadata.get("recipe_name", "desconocida")),
    )


def _audit_footer(response: str, sources: list[RecipeSource]) -> str:
    validation = validate_grounding(response, sources)
    return format_grounding_footer(validation)


async def stream_query(message: str, thread_id: str) -> AsyncIterator[str]:
    """
    Strict query handler. Yields progressively longer response text.

    Recipe suggestions are always template-formatted from ChromaDB.
    """
    message = message.strip()
    if not message:
        return

    if not is_cooking_query(message, thread_id):
        yield NON_COOKING_MESSAGE
        return

    # --- Scaling follow-up (uses last recipe from session) ---
    target_servings = _parse_target_servings(message)
    if target_servings and thread_id in _last_recipe_by_thread:
        doc = _last_recipe_by_thread[thread_id]
        current = doc.metadata.get("servings") or 4
        scaled = scaling_expert.invoke(
            {
                "recipe_text": doc.page_content,
                "current_servings": int(current),
                "target_servings": target_servings,
            }
        )
        row_id = doc.metadata.get("csv_row_id", "?")
        name = doc.metadata.get("recipe_name", "")
        footer = (
            f"\n\nFuente verificada: csv_row_id={row_id} | nombre={name}"
        )
        yield scaled + footer
        return

    # --- Substitution (does not invent a full recipe) ---
    if SUBSTITUTION_PATTERN.search(message) and not target_servings:
        result = substitution_expert.invoke(
            {
                "ingredient": message,
                "dietary_constraint": "",
            }
        )
        yield result
        return

    # --- Main path: search DB, template response, no LLM recipe generation ---
    filters = extract_filters(message)
    docs = search_recipe_documents(
        message,
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
    response = format_recipe_strict(best)
    sources = [_doc_to_source(best)]

    # Stream in chunks for UX (content is deterministic, not LLM-generated)
    chunk_size = 40
    for i in range(chunk_size, len(response) + chunk_size, chunk_size):
        yield response[:i]

    full = response + "\n\n---\n" + _audit_footer(response, sources)
    yield full


async def stream_query_llm_phrase(message: str, thread_id: str) -> AsyncIterator[str]:
    """
    Optional: wrap strict template with a one-line LLM intro (still no recipe invention).
    Not used by default — kept for future use.
    """
    filters = extract_filters(message)
    docs = search_recipe_documents(message, **filters, k=4)
    if not docs:
        yield NO_RECIPES_MESSAGE
        return

    best = docs[0]
    _last_recipe_by_thread[thread_id] = best
    body = format_recipe_strict(best)

    llm = get_llm(streaming=True)
    prompt = (
        "Escribí UNA sola frase corta introduciendo la receta. "
        "NO des ingredientes ni pasos. Solo la frase introductoria.\n"
        f"Receta: {best.metadata.get('recipe_name')}"
    )
    intro = ""
    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        if chunk.content:
            intro += chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            yield intro + "\n\n" + body

    sources = [_doc_to_source(best)]
    yield intro + "\n\n" + body + "\n\n---\n" + _audit_footer(body, sources)
