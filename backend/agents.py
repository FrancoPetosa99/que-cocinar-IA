"""LangGraph agent, tools, and session memory."""

from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from backend.config import get_llm
from backend.database import search_recipes

SYSTEM_PROMPT = """Sos Qué Cocinar IA, un asistente especializado exclusivamente en recetas de cocina.

OBJETIVO
Ayudar al usuario a cocinar utilizando los ingredientes disponibles y responder únicamente consultas relacionadas con cocina, recetas, ingredientes y técnicas culinarias.

HERRAMIENTAS DISPONIBLES
- recipe_retriever: buscá recetas reales en la base de datos. Usala para ingredientes disponibles, temas, clima (frío/calor), restricciones de tiempo o macros.
- scaling_expert: escalá cantidades de ingredientes cuando el usuario pida más o menos porciones.
- substitution_expert: sugerí reemplazos cuando falte un ingrediente o haya restricciones dietéticas.

REGLAS DE RESPUESTA
1. Proponé UNA sola receta por respuesta.
2. Elegí la receta más adecuada según los ingredientes o restricciones del usuario.
3. No enumeres múltiples alternativas salvo que el usuario lo solicite explícitamente.
4. No hagas metacomentarios innecesarios ("es muy popular...", "ideal para ganar masa...").
5. Sé directo y orientado a la acción.
6. Si faltan ingredientes, usá substitution_expert o proponé adaptaciones simples.
7. Si la consulta no es de cocina, respondé: "🍳 Solo puedo ayudarte con recetas y temas relacionados con la cocina."
8. Para recetas rápidas, usá recipe_retriever con max_total_time_min bajo (ej. 20).
9. Para atletas o alta proteína, usá recipe_retriever con min_protein_g alto (ej. 30).
10. SIEMPRE usá recipe_retriever antes de proponer una receta. NO inventes recetas ni nombres.
11. Elegí UNA receta de los resultados del retriever y basá tu respuesta en esa fila.
12. Al final de CADA respuesta incluí exactamente esta línea (sin modificar el formato):
    Fuente verificada: csv_row_id=XXX | nombre=YYY
    donde XXX es el csv_row_id de la receta elegida e YYY su recipe_name exacto del retriever.

FORMATO DE RESPUESTA
Nombre de la receta.

Ingredientes:
* lista breve de ingredientes

Preparación:
1. Paso 1
2. Paso 2
3. Paso 3

Mantené la respuesta breve y práctica. Respondé siempre en español.
"""

SCALING_PROMPT = """Sos un experto en escalado de recetas. Recibís una receta y debés adaptarla de {current} a {target} porciones.

Reglas:
- Multiplicá cada cantidad numérica por el factor {factor:.4f}.
- Mantené las unidades originales.
- Si un ingrediente no tiene cantidad numérica, indicá "a gusto" o "cantidad proporcional".
- Al final, incluí un breve resumen de macros estimados si hay datos nutricionales.

Receta original:
{recipe}
"""

SUBSTITUTION_PROMPT = """Sos un experto en sustituciones culinarias.

Ingrediente a reemplazar: {ingredient}
Restricción dietética (si aplica): {constraint}

Sugerí 2-3 sustitutos prácticos con:
- nombre del sustituto
- proporción de reemplazo (ej. 1:1, 3/4 taza por cada taza)
- nota breve sobre cómo afecta sabor o textura

Respondé en español, de forma concisa y accionable.
"""


@tool
def recipe_retriever(
    query: str,
    max_total_time_min: int | None = None,
    min_protein_g: float | None = None,
    max_calories: float | None = None,
) -> str:
    """
    Busca recetas en la base de datos por ingredientes, tema o restricciones.

  Args:
        query: Descripción de lo que el usuario busca (ingredientes, clima, estilo).
        max_total_time_min: Tiempo máximo en minutos (para recetas rápidas).
        min_protein_g: Proteína mínima en gramos (para dietas altas en proteína).
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
    Escala las cantidades de una receta de N a M porciones.

    Args:
        recipe_text: Texto completo de la receta a escalar.
        current_servings: Porciones actuales de la receta.
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
        ingredient: Ingrediente que falta o se quiere reemplazar.
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
