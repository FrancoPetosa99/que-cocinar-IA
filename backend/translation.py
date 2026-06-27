"""LCEL translation chains: Spanish user I/O, English internal processing."""

from __future__ import annotations

import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from backend.config import get_llm

_translate_to_english: Runnable | None = None
_translate_to_spanish: Runnable | None = None

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
- Use natural, direct Spanish suitable for a recipe app.
- Keep proper recipe titles in English if they are dish names (e.g. "Chicken Française").
- Do NOT translate or alter lines containing "csv_row_id=" — copy them exactly.
- Preserve markdown, bullet lists, and numbered steps.""",
        ),
        ("human", "{text}"),
    ]
)


def get_translate_to_english_chain() -> Runnable:
    """LCEL chain: Spanish text -> English text."""
    global _translate_to_english
    if _translate_to_english is None:
        llm = get_llm(streaming=False)
        _translate_to_english = TRANSLATE_TO_ENGLISH_PROMPT | llm | StrOutputParser()
    return _translate_to_english


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
