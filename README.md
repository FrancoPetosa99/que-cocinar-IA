# Qué Cocinar IA

Asistente de cocina con RAG: recetas indexadas en ChromaDB, agente LangGraph con herramientas, y chat con streaming en Gradio.

## Estructura

```
├── data_preprocessing/preprocessing.ipynb   # carga data/recipes.csv e indexa en ChromaDB
├── backend/
│   ├── config.py                            # LLM / embeddings (Gemini o Hugging Face)
│   ├── database.py                          # ChromaDB + búsqueda con filtros
│   └── agents.py                            # herramientas + agente ReAct
├── frontend/app.py                          # UI estilo cocina con streaming
└── chroma_db/                               # generado por el notebook (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Variables de entorno (`.env`)

**Gemini (default):**
```
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=tu_clave
```

**Hugging Face (hosted):**
```
LLM_PROVIDER=huggingface
HF_BACKEND=inference_api
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
HUGGINGFACEHUB_API_TOKEN=tu_token
```

**Hugging Face (local):**
```
LLM_PROVIDER=huggingface
HF_BACKEND=local
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

### Indexar recetas

1. Asegurate de tener el archivo `data/recipes.csv`.
2. Abrí y ejecutá `data_preprocessing/preprocessing.ipynb` de principio a fin.
3. Verificá que se creó la carpeta `chroma_db/`.

## Cómo ejecutar

Este proyecto **no tiene un servidor backend separado**. Los módulos en `backend/` (ChromaDB, agente LangGraph, herramientas) se cargan automáticamente cuando iniciás la app de Gradio.

### Requisitos previos

Antes de arrancar, verificá que tengas:

- Entorno virtual activado y dependencias instaladas (`pip install -r requirements.txt`)
- Archivo `.env` con al menos `GEMINI_API_KEY` y `LLM_MODEL`
- Carpeta `chroma_db/` generada por el notebook de preprocessing

### Iniciar la aplicación (backend + frontend)

Desde la raíz del proyecto:

```bash
source .venv/bin/activate
python frontend/app.py
```

Gradio abrirá el chat en el navegador (por defecto `http://127.0.0.1:7860`). Al enviar un mensaje, el flujo es:

1. **Frontend** (`frontend/app.py`) recibe la consulta y hace streaming de la respuesta.
2. **Backend** (`backend/agents.py`) enruta la consulta al agente ReAct y sus herramientas.
3. **Base de datos** (`backend/database.py`) busca recetas en ChromaDB cuando hace falta.

### Probar el backend sin la interfaz (opcional)

Útil para verificar que ChromaDB y el agente funcionan antes de abrir Gradio.

**Probar la base vectorial:**

```bash
source .venv/bin/activate
python -c "
from backend.database import search_recipes
print(search_recipes('pollo y arroz', max_total_time=30))
"
```

**Probar que el agente se construye correctamente:**

```bash
source .venv/bin/activate
python -c "
from backend.agents import build_agent, TOOLS
agent = build_agent()
print('Agente OK. Herramientas:', [t.name for t in TOOLS])
"
```

**Probar una consulta completa por consola** (sin streaming):

```bash
source .venv/bin/activate
python -c "
from backend.agents import build_agent, get_agent_config
from langchain_core.messages import HumanMessage

agent = build_agent()
config = get_agent_config('test-session')
result = agent.invoke(
    {'messages': [HumanMessage(content='Tengo pollo, ¿qué puedo cocinar?')]},
    config=config,
)
print(result['messages'][-1].content)
"
```

### Opciones de lanzamiento

```bash
# Puerto y host personalizados
python frontend/app.py  # editá launch() en frontend/app.py si necesitás server_name/share

# O desde Python:
python -c "from frontend.app import create_app, KITCHEN_CSS; demo, theme = create_app(); demo.queue().launch(server_name='0.0.0.0', server_port=7860, theme=theme, css=KITCHEN_CSS)"
```

### Solución de problemas

| Error | Qué hacer |
|---|---|
| `No se encontró la base vectorial en 'chroma_db'` | Ejecutá `data_preprocessing/preprocessing.ipynb` |
| `GEMINI_API_KEY no está configurada` | Creá o completá el archivo `.env` en la raíz |
| La app no responde / timeout | Revisá conexión a internet y que la API key de Gemini sea válida |
| `ModuleNotFoundError: backend` | Ejecutá siempre desde la raíz del proyecto, no desde `frontend/` |

## Modo estricto (sin alucinaciones)

El flujo principal usa `backend/pipeline.py`, **no** deja que el LLM elija ni redacte recetas:

1. Busca en ChromaDB con la consulta del usuario (+ filtros de tiempo/macros si aplica).
2. Si **no hay resultados relevantes** → responde con un mensaje fijo (sin LLM).
3. Si **hay resultados** → formatea la mejor coincidencia con una **plantilla** tomada literalmente de la base (`format_recipe_strict`). El LLM **no** genera ingredientes ni pasos.

Excepciones donde sí interviene el LLM:
- **Escalado de porciones** (sobre la última receta mostrada en la sesión).
- **Sustituciones** de ingredientes (no propone una receta nueva completa).

Ajustá la sensibilidad con `RETRIEVAL_MAX_DISTANCE` en `.env` (ver sección abajo).

## Verificar que la receta viene de la base de datos

No se puede **garantizar al 100%** solo con el prompt. En **modo estricto** la receta principal se arma por plantilla desde ChromaDB; la auditoría confirma el `csv_row_id`.

### Cómo funciona

1. **`csv_row_id` en metadata** — cada receta indexada en ChromaDB lleva el id de la fila en `data/recipes.csv`.
2. **`recipe_retriever` obligatorio** — el system prompt exige consultar la base antes de proponer una receta.
3. **Cita obligatoria** — al final de cada respuesta el modelo debe incluir:
   ```
   Fuente verificada: csv_row_id=123 | nombre=Nombre de la receta
   ```
4. **Auditoría automática** — al terminar el streaming, el frontend compara:
   - qué recetas devolvió `recipe_retriever` (desde el estado del agente)
   - qué `csv_row_id` citó el modelo
   - y muestra **✅ VERIFICADO** o **⚠️ NO VERIFICADO**

### Re-indexar (importante)

Si indexaste antes de este cambio, **volvé a ejecutar** `data_preprocessing/preprocessing.ipynb` para que `chroma_db/` incluya `csv_row_id`.

### Probar manualmente en consola

```bash
source .venv/bin/activate
python -c "
from backend.database import search_recipes, get_csv_row_preview
out = search_recipes('pollo y arroz', k=2)
print(out)
print('---')
# Reemplazá 123 por un csv_row_id que aparezca arriba
print(get_csv_row_preview(0))
"
```

### Probar auditoría de una consulta completa

```bash
source .venv/bin/activate
python -c "
from backend.agents import build_agent, get_agent_config
from backend.grounding import extract_retrieved_sources, validate_grounding, format_grounding_footer
from langchain_core.messages import HumanMessage

agent = build_agent()
config = get_agent_config('audit-test')
result = agent.invoke(
    {'messages': [HumanMessage(content='Tengo pollo, ¿qué puedo cocinar?')]},
    config=config,
)
answer = result['messages'][-1].content
state = agent.get_state(config)
validation = validate_grounding(answer, extract_retrieved_sources(state))
print(answer)
print('---')
print(format_grounding_footer(validation))
"
```

### Qué mirar al testear

| Señal | Significado |
|---|---|
| ✅ VERIFICADO | El `csv_row_id` citado estuvo entre los resultados de ChromaDB |
| ⚠️ NO VERIFICADO + sin retriever | El agente no llamó a la base (posible alucinación) |
| ⚠️ NO VERIFICADO + id no coincide | El modelo citó un id que no recuperó |

Contrastá siempre con `data/recipes.csv`: la fila cuya primera columna (`Unnamed: 0`) coincide con `csv_row_id` debe ser la misma receta.

### Umbral de relevancia (`RETRIEVAL_MAX_DISTANCE`)

**No es cosine similarity (−1 a 1).** Chroma devuelve una **distancia**: cuanto **más bajo**, más similar.

| Métrica en tu colección | Por defecto en Chroma (sin config explícita) |
|---|---|
| **L2** (distancia euclidiana entre embeddings) | Es la que usa tu `chroma_db` actual |

Ejemplos reales en este proyecto:

| Consulta | Distancia top-1 | ¿Útil? |
|---|---|---|
| `chicken rice` | ~0.93 | Sí |
| `pollo y arroz` | ~1.37 | Débil (match irrelevante) |
| `xyznonexistent999` | ~1.72 | No |

Default `RETRIEVAL_MAX_DISTANCE=1.35`: acepta buenos matches (~0.9) y rechaza consultas sin receta relevante (~1.4+).

- **Más estricto** → bajar (ej. `1.1`) → más respuestas "no hay recetas".
- **Más permisivo** → subir (ej. `1.5`) → más resultados, aunque a veces irrelevantes.

Si quisieras pensar en cosine similarity (vectores normalizados, métrica L2):

\[
\text{cos\_sim} \approx 1 - \frac{\text{L2}^2}{2}
\]

Ej.: L2 = 0.93 → cos_sim ≈ 0.57 | L2 = 1.35 → cos_sim ≈ 0.09 | L2 = 1.05 → cos_sim ≈ 0.45

El valor `1.05` anterior era un default arbitrario y el comentario de "cosine" era **incorrecto**. Usá `RETRIEVAL_MAX_DISTANCE` en `.env`.

## Herramientas del agente

| Herramienta | Uso |
|---|---|
| `recipe_retriever` | Busca recetas por ingredientes, tema, tiempo o macros |
| `scaling_expert` | Escala porciones (ej. 5 → 10) |
| `substitution_expert` | Sugiere reemplazos de ingredientes |

El agente ReAct enruta automáticamente según la intención del usuario.
