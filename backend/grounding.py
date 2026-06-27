"""Utilities to trace and verify that answers come from the recipe database."""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import ToolMessage

# Matches [csv_row_id=123] in retriever output
RECIPE_ID_TAG = re.compile(r"\[csv_row_id=(\d+)\]")
# Matches "csv_row_id=123" or "csv_row_id: 123" in the assistant reply
CITATION_IN_RESPONSE = re.compile(r"csv_row_id\s*[=:]\s*(\d+)", re.IGNORECASE)
RECIPE_NAME_IN_CITATION = re.compile(
    r"(?:nombre|name)\s*[=:]\s*(.+?)(?:\n|$)", re.IGNORECASE
)


@dataclass
class RecipeSource:
    csv_row_id: int
    recipe_name: str


def parse_sources_from_tool_output(tool_output: str) -> list[RecipeSource]:
    """Extract recipe ids returned by recipe_retriever."""
    sources: list[RecipeSource] = []
    for block in tool_output.split("---"):
        id_match = RECIPE_ID_TAG.search(block)
        if not id_match:
            continue
        name_match = re.search(r"\*\*(.+?)\*\*", block)
        sources.append(
            RecipeSource(
                csv_row_id=int(id_match.group(1)),
                recipe_name=name_match.group(1).strip() if name_match else "desconocida",
            )
        )
    return sources


def extract_retrieved_sources(state) -> list[RecipeSource]:
    """Collect all recipes returned by recipe_retriever in the agent state."""
    messages = state.values.get("messages", [])
    seen: set[int] = set()
    unique: list[RecipeSource] = []

    for msg in messages:
        if not isinstance(msg, ToolMessage) or msg.name != "recipe_retriever":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        for source in parse_sources_from_tool_output(content):
            if source.csv_row_id not in seen:
                seen.add(source.csv_row_id)
                unique.append(source)

    return unique


def parse_citation_from_response(text: str) -> RecipeSource | None:
    """Parse the 'Fuente verificada' line from the assistant response."""
    id_match = CITATION_IN_RESPONSE.search(text)
    if not id_match:
        return None
    name_match = RECIPE_NAME_IN_CITATION.search(text)
    name = name_match.group(1).strip() if name_match else "desconocida"
    return RecipeSource(csv_row_id=int(id_match.group(1)), recipe_name=name)


def validate_grounding(
    response_text: str,
    retrieved: list[RecipeSource],
) -> dict:
    """
    Check whether the assistant cited a recipe that was actually retrieved.

    Returns a dict with grounded (bool), reason, cited, and retrieved lists.
    """
    cited = parse_citation_from_response(response_text)
    retrieved_ids = {s.csv_row_id for s in retrieved}

    if not retrieved:
        return {
            "grounded": False,
            "reason": "El agente no consultó recipe_retriever en esta conversación.",
            "cited": cited,
            "retrieved": retrieved,
        }

    if cited is None:
        return {
            "grounded": False,
            "reason": "La respuesta no incluye la línea 'Fuente verificada: csv_row_id=...'.",
            "cited": None,
            "retrieved": retrieved,
        }

    if cited.csv_row_id not in retrieved_ids:
        return {
            "grounded": False,
            "reason": (
                f"El csv_row_id citado ({cited.csv_row_id}) no está entre "
                f"las recetas recuperadas ({sorted(retrieved_ids)})."
            ),
            "cited": cited,
            "retrieved": retrieved,
        }

    return {
        "grounded": True,
        "reason": "La receta citada proviene de la base vectorial.",
        "cited": cited,
        "retrieved": retrieved,
    }


def format_grounding_footer(validation: dict) -> str:
    """Build a human-readable audit block appended to each assistant reply."""
    cited = validation.get("cited")
    retrieved: list[RecipeSource] = validation.get("retrieved", [])
    grounded = validation.get("grounded", False)
    reason = validation.get("reason", "")

    status = "✅ VERIFICADO" if grounded else "⚠️ NO VERIFICADO"
    lines = [
        f"**Auditoría de fuente** {status}",
        reason,
    ]

    if cited:
        lines.append(
            f"Receta citada por el modelo: `csv_row_id={cited.csv_row_id}` | `{cited.recipe_name}`"
        )

    if retrieved:
        lines.append("Recetas recuperadas de ChromaDB en esta consulta:")
        for src in retrieved:
            lines.append(f"- `csv_row_id={src.csv_row_id}` | {src.recipe_name}")

    lines.append(
        "Para contrastar con el CSV: `data/recipes.csv`, fila con índice = csv_row_id."
    )
    return "\n".join(lines)
