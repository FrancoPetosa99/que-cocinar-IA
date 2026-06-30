"""LangGraph agent, tools, and session memory (prompts en español)."""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from backend.config import get_llm
from backend.database import search_recipes

SYSTEM_PROMPT = """Sos Qué Cocinar IA, un asistente especializado exclusivamente en recetas de cocina.

OBJETIVO
Ayudar al usuario a cocinar con los ingredientes disponibles y responder solo consultas relacionadas con cocina, recetas, ingredientes y técnicas culinarias.

HERRAMIENTAS DISPONIBLES
- recipe_retriever: busca recetas reales en la base. Usala para ingredientes disponibles, temas, clima (calor/frío), tiempo o restricciones nutricionales.
- scaling_expert: escala cantidades cuando el usuario pide más o menos porciones.
- substitution_expert: sugiere reemplazos cuando falta un ingrediente o hay restricciones dietéticas.

REGLAS DE RESPUESTA
1. Sugerí UNA receta por respuesta.
2. Elegí la mejor receta según los ingredientes o restricciones del usuario.
3. No listes múltiples alternativas salvo que lo pidan explícitamente.
4. Sin meta-comentarios innecesarios ("es muy popular...", "ideal para ganar músculo...").
5. Sé directo y orientado a la acción.
6. Si faltan ingredientes, usá substitution_expert o sugerí adaptaciones simples.
7. Si la consulta no es de cocina, respondé: "Solo puedo ayudarte con recetas y temas relacionados con la cocina."
8. Para recetas rápidas, usá recipe_retriever con max_total_time_min bajo (ej. 20).
9. Para atletas o alta proteína, usá recipe_retriever con min_protein_g alto (ej. 30).
10. SIEMPRE usá recipe_retriever antes de sugerir una receta. NUNCA inventes recetas ni nombres.
11. Elegí UNA receta de los resultados del retriever y basá tu respuesta en esa fila.
12. Al FINAL de CADA respuesta incluí exactamente esta línea (no cambies el formato):
    Fuente verificada: csv_row_id=XXX | name=YYY
    donde XXX es el csv_row_id de la receta elegida e YYY es su recipe_name exacto del retriever.

FORMATO DE RESPUESTA
Nombre de la receta.

Ingredientes:
* lista breve de ingredientes

Preparación:
1. Paso 1
2. Paso 2
3. Paso 3

Sé breve y práctico. Respondé siempre en español.
"""

SCALING_PROMPT = """Sos un experto en escalado de recetas. Recibís una receta y debés adaptarla de {current} a {target} porciones.

Reglas:
- Multiplicá cada cantidad numérica por el factor {factor:.4f}.
- Mantené las unidades originales.
- Si un ingrediente no tiene cantidad numérica, usá "a gusto" o "cantidad proporcional".
- Al final, incluí un breve resumen estimado de macros si hay datos nutricionales.

Receta original:
{recipe}

Respondé solo en español.
"""

SUBSTITUTION_PROMPT = """Sos un experto en sustituciones culinarias.

Ingrediente a reemplazar: {ingredient}
Restricción dietética (si hay): {constraint}

Sugerí 2-3 sustitutos prácticos con:
- nombre del sustituto
- proporción de reemplazo (ej. 1:1, 3/4 taza por taza)
- nota breve sobre impacto en sabor o textura

Respondé solo en español. Sé conciso y accionable.
"""


@tool
def recipe_retriever(
    query: str,
    max_total_time_min: int | None = None,
    min_protein_g: float | None = None,
    max_calories: float | None = None,
) -> str:
    """
    Busca recetas en la base por ingredientes, tema o restricciones.

    Args:
        query: Qué busca el usuario (ingredientes, clima, estilo). En español.
        max_total_time_min: Tiempo máximo en minutos (recetas rápidas).
        min_protein_g: Proteína mínima en gramos (dietas altas en proteína).
        max_calories: Calorías máximas por porción.
    """
    return search_recipes(
        query,
        max_total_time=max_total_time_min,
        min_protein=min_protein_g,
        max_calories=max_calories,
    )


@tool
def scaling_expert(
    recipe_text: str,
    current_servings: int,
    target_servings: int,
) -> str:
    """
    Escala cantidades de una receta de N a M porciones.

    Args:
        recipe_text: Texto completo de la receta a escalar.
        current_servings: Cantidad actual de porciones.
        target_servings: Porciones deseadas.
    """
    if current_servings <= 0 or target_servings <= 0:
        return "Las porciones deben ser números positivos."

    factor = target_servings / current_servings
    llm = get_llm(streaming=False)
    prompt = SCALING_PROMPT.format(
        current=current_servings,
        target=target_servings,
        factor=factor,
        recipe=recipe_text,
    )
    response = llm.invoke(prompt)
    return response.content


@tool
def substitution_expert(
    ingredient: str,
    dietary_constraint: str = "",
) -> str:
    """
    Sugiere alternativas para un ingrediente faltante o restricción dietética.

    Args:
        ingredient: Ingrediente faltante o a reemplazar.
        dietary_constraint: Restricción opcional (vegano, sin gluten, etc.).
    """
    llm = get_llm(streaming=False)
    prompt = SUBSTITUTION_PROMPT.format(
        ingredient=ingredient,
        constraint=dietary_constraint or "ninguna",
    )
    response = llm.invoke(prompt)
    return response.content


TOOLS = [recipe_retriever, scaling_expert, substitution_expert]

_checkpointer = MemorySaver()
_agent = None


def build_agent():
    """Create (or return cached) LangGraph ReAct agent with session memory."""
    global _agent
    if _agent is not None:
        return _agent

    llm = get_llm(streaming=True)
    _agent = create_react_agent(
        llm,
        TOOLS,
        checkpointer=_checkpointer,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    return _agent


def get_agent_config(thread_id: str) -> dict:
    """Return LangGraph config for a given session thread."""
    return {"configurable": {"thread_id": thread_id}}
