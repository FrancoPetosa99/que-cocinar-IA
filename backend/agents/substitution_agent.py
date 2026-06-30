from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from backend.config import get_llm

SUBSTITUTION_PROMPT = """
Sos un experto en sustituciones culinarias.

Tu tarea consiste únicamente en sugerir reemplazos para ingredientes.

Para cada sustituto indicá:

- nombre
- proporción
- impacto en sabor
- impacto en textura

Respondé siempre en español.

No respondas consultas que no sean de sustituciones.
"""

_substitution_agent = None

def build_substitution_agent():

    global _substitution_agent

    if _substitution_agent is not None:
        return _substitution_agent

    llm = get_llm(streaming=False)

    _substitution_agent = create_react_agent(
        llm,
        tools=[],
        prompt=SystemMessage(content=SUBSTITUTION_PROMPT),
    )

    return _substitution_agent