"""LCEL translation chains: Spanish user I/O, English internal processing."""

from __future__ import annotations

import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from backend.config import get_llm

_translate_to_english: Runnable | None = None
_translate_to_spanish: Runnable | None = None
_present_recipe_in_spanish: Runnable | None = None

TRANSLATE_TO_ENGLISH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a professional translator for a cooking assistant.
Translate the user message from Spanish to English.
Rules:
- Output ONLY the English translation, no preamble or explanation.
- Preserve numbers, units, portion counts, and ingredient names accurately.
- Keep the cooking intent (ingredients available, dietary needs, time constraints).
- If the text is already in English, return it unchanged.""",
        ),
        ("human", "{text}"),
    ]
)

TRANSLATE_TO_SPANISH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a professional translator for a cooking assistant.
Translate the assistant response from English to Spanish (Rioplatense / Argentina).
Rules:
- Output ONLY the Spanish translation, no preamble.
- Use natural, warm Spanish suitable for a recipe app.
- Keep proper recipe titles in English if they are dish names (e.g. "Chicken Française").
- Do NOT translate or alter lines containing "csv_row_id=" — copy them exactly.
- Preserve ALL sections: title, ingredients, and directions. Never omit a section.
- Preserve markdown, bullet lists, and numbered steps.
- Use correct culinary terms (e.g. rack = rejilla, backbone = espinazo, lemon = limón).""",
        ),
        ("human", "{text}"),
    ]
)

RECIPE_PRESENTATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Qué Cocinar IA, a warm and encouraging cooking assistant for home cooks in Argentina.

Present the recipe below in natural Spanish (Rioplatense). Use ONLY the data provided — never invent ingredients, quantities, or steps.

User message: {user_query}

Recipe data:
- Name: {recipe_name}
- Servings: {servings}
- Prep time: {prep_time}
- Total time: {total_time}
- Ingredients: {ingredients}
- Directions: {directions}

RESPONSE STRUCTURE — include every section, in this order:

1. Opening (1-2 sentences): greet the cook warmly and connect with what they asked or the ingredients they have.
2. **{recipe_name}** as the recipe title (keep the original dish name; add a short Spanish subtitle in parentheses only if it helps).
3. **Porciones:** (only if servings is not "—")
4. **Tiempo:** combine prep and total time when available (skip if both are "—")
5. **Ingredientes:** bullet list with every ingredient from the data, translated clearly with accurate quantities and units.
6. **Preparación:** numbered steps covering every direction from the data.

RULES:
- Be friendly and encouraging, like a friend helping in the kitchen — not dry or robotic.
- Include ALL ingredients and ALL steps. Do not summarize or skip sections.
- Use correct culinary vocabulary (rack = rejilla de horno, backbone = espinazo, lemon = limón, herbs = hierbas).
- Do NOT include source verification lines or metadata.
- Output ONLY the recipe presentation in Spanish.""",
        ),
        ("human", "Presentá la receta para el usuario."),
    ]
)


def get_translate_to_english_chain() -> Runnable:
    """LCEL chain: Spanish text -> English text."""
    global _translate_to_english
    if _translate_to_english is None:
        llm = get_llm(streaming=False)
        _translate_to_english = TRANSLATE_TO_ENGLISH_PROMPT | llm | StrOutputParser()
    return _translate_to_english


def reset_translation_chains() -> None:
    """Clear cached LCEL chains (e.g. after switching LLM provider)."""
    global _translate_to_english, _translate_to_spanish, _present_recipe_in_spanish
    _translate_to_english = None
    _translate_to_spanish = None
    _present_recipe_in_spanish = None


def get_translate_to_spanish_chain() -> Runnable:
    """LCEL chain: English text -> Spanish text."""
    global _translate_to_spanish
    if _translate_to_spanish is None:
        llm = get_llm(streaming=False)
        _translate_to_spanish = TRANSLATE_TO_SPANISH_PROMPT | llm | StrOutputParser()
    return _translate_to_spanish


async def translate_to_english(text: str) -> str:
    """Async wrapper: translate user input to English for retrieval and tools."""
    chain = get_translate_to_english_chain()
    result = await asyncio.to_thread(chain.invoke, {"text": text})
    return result.strip() or text


async def translate_to_spanish(text: str) -> str:
    """Async wrapper: translate assistant output back to Spanish for the user."""
    chain = get_translate_to_spanish_chain()
    result = await asyncio.to_thread(chain.invoke, {"text": text})
    return result.strip() or text


def get_present_recipe_in_spanish_chain() -> Runnable:
    """LCEL chain: recipe row + user query -> friendly Spanish presentation."""
    global _present_recipe_in_spanish
    if _present_recipe_in_spanish is None:
        llm = get_llm(streaming=False)
        _present_recipe_in_spanish = (
            RECIPE_PRESENTATION_PROMPT | llm | StrOutputParser()
        )
    return _present_recipe_in_spanish


def _format_optional_field(value: str | int | None) -> str:
    if value is None:
        return "—"
    text = str(value).strip()
    return text or "—"


async def present_recipe_in_spanish(
    *,
    user_query: str,
    recipe_name: str,
    ingredients: str,
    directions: str,
    servings: int | None = None,
    prep_time: str | None = None,
    total_time: str | None = None,
) -> str:
    """Format a database recipe as a warm, complete Spanish response."""
    chain = get_present_recipe_in_spanish_chain()
    result = await asyncio.to_thread(
        chain.invoke,
        {
            "user_query": user_query,
            "recipe_name": recipe_name,
            "ingredients": ingredients,
            "directions": directions,
            "servings": _format_optional_field(servings),
            "prep_time": _format_optional_field(prep_time),
            "total_time": _format_optional_field(total_time),
        },
    )
    return result.strip()
