from __future__ import annotations

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from backend.config import get_llm

SCALING_PROMPT = """
Sos un experto en escalado de recetas.

Tu única tarea es adaptar recetas a una nueva cantidad de porciones.

Reglas:

- Multiplicá todas las cantidades por el factor correspondiente.
- Conservá las unidades originales.
- Si un ingrediente no posee cantidad numérica, indicá "cantidad proporcional".
- No inventes ingredientes.
- No modifiques el procedimiento salvo que sea necesario.
- Si existen datos nutricionales, adaptalos también.

Respondé siempre en español.
"""

_scaling_agent = None

def build_scaling_agent():

    global _scaling_agent

    if _scaling_agent is not None:
        return _scaling_agent

    llm = get_llm(streaming=False)

    _scaling_agent = create_react_agent(
        model=llm,
        tools=[],
        prompt=SystemMessage(content=SCALING_PROMPT),
    )

    return _scaling_agent