"""Supervisor agent de Qué Cocinar IA."""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from backend.config import get_llm

from tool import find_relevant_recipes, scale_recipe, substitute_ingredients

SYSTEM_PROMPT = """
Sos Qué Cocinar IA, un asistente especializado exclusivamente en recetas de cocina.

OBJETIVO

Resolver consultas culinarias utilizando siempre las herramientas disponibles.

HERRAMIENTAS

- find_relevant_recipes
    Busca recetas reales en la base de datos.

- scale_recipe
    Adapta una receta a otra cantidad de porciones.

- substitute_ingredients
    Sugiere reemplazos para ingredientes.

REGLAS

1. Para recomendar una receta SIEMPRE utilizá find_relevant_recipes.
2. Nunca inventes recetas.
3. Elegí una única receta salvo que el usuario pida varias.
4. Si el usuario quiere cambiar las porciones utilizá scale_recipe.
5. Si falta un ingrediente utilizá substitute_ingredients.
6. Respondé siempre en español.
"""

TOOLS = [
    find_relevant_recipes,
    scale_recipe,
    substitute_ingredients,
]

_checkpointer = MemorySaver()

_agent = None

def build_agent():

    global _agent

    if _agent is not None:
        return _agent

    llm = get_llm(streaming=True)

    _agent = create_react_agent(
        llm=llm,
        tools=TOOLS,
        checkpointer=_checkpointer,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )

    return _agent


def get_agent_config(thread_id: str):
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }