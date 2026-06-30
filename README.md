# Qué Cocinar IA

Asistente de cocina con RAG en modo estricto: búsqueda vectorial en ChromaDB, datos completos en SQLite, pipeline en **español**, y chat con streaming en Gradio.

El usuario escribe en **español**. Los embeddings y la metadata enriquecida están en **español** (`enriched_recipes_spanish.csv`). El LLM **no inventa** la receta principal: escala porciones y sugiere sustituciones cuando corresponde.

---

## Arquitectura dual

```
data/recipes.csv + data/enriched_recipes_spanish.csv
       │
       ├─ Fase 1 (ingest) ──► data/recipes.db     SQLite — fuente de verdad (fila completa + metadata ES)
       │
       └─ Fase 2 (ingest) ──► chroma_db/          Chroma — búsqueda semántica (documento en español)
```

| Capa | Archivo | Rol | Qué guarda |
|---|---|---|---|
| **SQLite** | `data/recipes.db` | Fuente de verdad | `recipe_name`, `ingredients`, **`directions`**, tiempos, rating, nutrition, metadata enriquecida en español… |
| **ChromaDB** | `chroma_db/` | Búsqueda semántica | `page_content` = documento enriquecido en español; `metadata` = `csv_row_id`, tiempos, macros… |

**Principio:** Chroma devuelve solo **IDs** (`csv_row_id`). Con ese ID, SQLite entrega la receta completa.

> **Nota:** Usá `data/recipes_spanish.csv` (generado con `scripts/translate_recipes_spanish.py`) para ingredientes y pasos en español. Si no existe, el ingest usa `recipes.csv` en inglés como respaldo.

---

## Estructura del proyecto

```
que-cocinar-IA/
├── data/
│   ├── recipes.csv                     # fuente original (inglés)
│   ├── recipes_spanish.csv             # recetas traducidas (generado)
│   ├── enriched_recipes_spanish.csv    # metadata en español para búsqueda
│   └── recipes.db                      # SQLite (generado por ingest, gitignored)
├── chroma_db/                          # índice vectorial (generado por ingest, gitignored)
├── data_preprocessing/
│   ├── ingest.py                       # CSV → SQLite + Chroma (recomendado)
│   └── preprocessing.ipynb             # alternativa interactiva
├── backend/
│   ├── config.py                       # .env, LLM/embeddings (Gemini o Hugging Face)
│   ├── recipe_db.py                    # SQLite: Recipe, get_recipe_by_id()
│   ├── vector_store.py                 # Chroma: search_recipe_ids()
│   ├── database.py                     # fachada (IDs + SQL)
│   ├── pipeline.py                     # orquestador principal (modo estricto)
│   ├── grounding.py                    # auditoría de fuente
│   ├── agents.py                       # herramientas scaling / substitution (LLM)
│   └── recipe_parsing.py               # parsers compartidos (tiempos, nutrition)
├── frontend/
│   └── app.py                          # UI Gradio con streaming
├── scripts/
│   ├── ingest_relational_db.py         # recipes_spanish + enriched → SQLite
│   ├── ingest_vectorial_db.py          # enriched → Chroma
│   ├── translate_recipes_spanish.py    # recipes.csv → recipes_spanish.csv
│   └── download_hf_model.py            # descarga modelo HF a models/
├── models/                             # modelos locales (gitignored)
├── requirements.txt
└── .env
```

---

## Flujo de una consulta

```
Usuario (ES)
    → frontend/app.py
    → pipeline.stream_query()
        1. Normalizar consulta (español)
        2. ¿Es cocina? → si no, mensaje fijo
        3. Chroma: search_recipe_ids() → [329, 305, ...]
        4. Si no hay IDs → "No encontré recetas..."
        5. SQLite: get_recipe_by_id(329) → fila completa (con directions)
        6. format_recipe_from_sql() → plantilla en español (sin LLM)
        7. Auditoría: verificar csv_row_id
    → Usuario (ES)
```

**Casos especiales:**

| Tipo de consulta | Comportamiento |
|---|---|
| Buscar receta | Chroma → ID → SQLite → plantilla |
| Escalado ("10 porciones") | `scaling_expert` sobre la última receta de la sesión |
| Sustitución ("no tengo huevo") | `substitution_expert` (no inventa receta nueva) |
| No es cocina | Mensaje fijo, sin LLM |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Variables de entorno (`.env`)

Copiá `.env.example` a `.env` o usá esta configuración para **Hugging Face local**:

```env
LLM_PROVIDER=huggingface
HF_BACKEND=local
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
HF_MODEL_CACHE_DIR=models
HF_DEVICE=auto              # auto | cpu | mps | cuda
HF_MAX_NEW_TOKENS=512
TEMPERATURE=0.3

EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHROMA_DIR=chroma_db
SQLITE_PATH=data/recipes.db
RETRIEVAL_MAX_DISTANCE=1.35
```

### Modelo Hugging Face local (recomendado)

**1. Instalar dependencias** (incluye PyTorch):

```bash
pip install -r requirements.txt
```

**2. Descargar el modelo** (~3 GB para Qwen2.5-1.5B):

```bash
python scripts/download_hf_model.py
# o con otro modelo:
python scripts/download_hf_model.py microsoft/Phi-3-mini-4k-instruct
```

El modelo se guarda en `models/<nombre>/`. La app lo detecta automáticamente.

**3. Iniciar la app** (carga el modelo al arrancar):

```bash
python frontend/app.py
```

Modelos sugeridos según tu hardware:

| Modelo | Tamaño aprox. | Notas |
|---|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | Default, liviano |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | ~3 GB | Muy liviano |
| `microsoft/Phi-3-mini-4k-instruct` | ~7 GB | Mejor calidad, más RAM |

**Alternativa: Gemini (cloud)**

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=tu_clave
```

**Alternativa: Hugging Face Inference API (cloud, sin descargar)**

```env
LLM_PROVIDER=huggingface
HF_BACKEND=inference_api
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
HUGGINGFACEHUB_API_TOKEN=tu_token
```

---

## Indexar recetas

Requisitos: `data/recipes.csv` y `data/enriched_recipes_spanish.csv` en el proyecto.

### Traducir recetas a español (opcional, recomendado)

```bash
pip install argostranslate   # si aún no está instalado
python scripts/translate_recipes_spanish.py
```

Genera `data/recipes_spanish.csv`. El ingest relacional lo usa automáticamente si existe.

Opciones:

```bash
python scripts/translate_recipes_spanish.py --limit 10    # prueba con 10 filas
python scripts/translate_recipes_spanish.py --resume    # continuar traducción parcial
```

### Indexar SQLite + Chroma

```bash
python data_preprocessing/ingest.py
```

| Fase | Qué hace |
|---|---|
| 1 | `recipes_spanish.csv` (o `recipes.csv`) + `enriched_recipes_spanish.csv` → `data/recipes.db` |
| 2 | `enriched_recipes_spanish.csv` → `chroma_db/` (documentos en español embeddeados) |

Opciones:

```bash
python data_preprocessing/ingest.py --relational-only   # solo SQLite
python data_preprocessing/ingest.py --vector-only       # solo Chroma (requiere SQLite previo)
```

> **Importante:** Si cambiás `EMBEDDING_MODEL`, reconstruí Chroma con `ingest.py --vector-only` o el ingest completo.

---

## Ejecutar la app

```bash
source .venv/bin/activate
python frontend/app.py
```

Al arrancar deberías ver:

```
✓ SQLite lista (.../data/recipes.db)
✓ Chroma lista (.../chroma_db)
```

Gradio abre en `http://127.0.0.1:7860`.

> No hay servidor backend separado. `backend/` se carga dentro del proceso de Gradio.

---

## Modo estricto (anti-alucinación)

Para la **receta principal**:

1. Chroma encuentra el mejor `csv_row_id` (con filtros opcionales de tiempo/macros).
2. SQLite devuelve la fila real.
3. La respuesta se arma con **plantilla fija** (`format_recipe_from_sql`) — el LLM no redacta ingredientes ni pasos.
4. Si no hay match relevante → mensaje fijo, sin inventar receta.

El LLM solo interviene en:

- Escalado de porciones (`scaling_expert`)
- Sustituciones de ingredientes (`substitution_expert`)

No hay traducción en runtime: consultas, embeddings, prompts y respuestas están en español.

---

## Auditoría de fuente

Cada respuesta incluye:

```
Fuente verificada: csv_row_id=329 | name=Chicken with Lemon-Caper Sauce
```

Y un bloque **Auditoría de fuente** con ✅ VERIFICADO o ⚠️ NO VERIFICADO.

Contrastá con `data/recipes.csv`: la columna `Unnamed: 0` debe coincidir con `csv_row_id`.

---

## Umbral de relevancia (`RETRIEVAL_MAX_DISTANCE`)

Chroma devuelve **distancia L2** (no cosine similarity). **Más bajo = más similar.**

| Consulta | Distancia típica | Resultado con umbral 1.35 |
|---|---|---|
| `pollo y arroz` | ~0.9–1.2 | ✅ Aceptada |
| consulta poco relacionada | ~1.5+ | ❌ Rechazada |

- **Más estricto** → bajar (ej. `1.1`)
- **Más permisivo** → subir (ej. `1.5`)

---

## Probar sin Gradio

**Búsqueda vectorial → ID:**

```bash
python -c "
from backend.vector_store import search_recipe_ids
print(search_recipe_ids('pollo y arroz', k=3))
"
```

**ID → receta completa desde SQLite:**

```bash
python -c "
from backend.recipe_db import get_recipe_by_id
from backend.pipeline import format_recipe_from_sql
r = get_recipe_by_id(329)
print(format_recipe_from_sql(r)[:600])
"
```

**Flujo integrado (búsqueda + SQL):**

```bash
python -c "
from backend.vector_store import search_recipe_ids
from backend.recipe_db import get_recipe_by_id
ids = search_recipe_ids('pollo', k=1)
r = get_recipe_by_id(ids[0])
print(f'ID={r.id} | {r.recipe_name} | directions={len(r.directions)} chars')
"
```

---

## Solución de problemas

| Error | Qué hacer |
|---|---|
| `No se encontró la base vectorial en 'chroma_db'` | `python data_preprocessing/ingest.py` |
| `No se encontró la base relacional` | `python data_preprocessing/ingest.py --relational-only` |
| `csv_row_id` missing / integridad | Re-ejecutar `ingest.py` completo y **reiniciar** la app |
| `GEMINI_API_KEY no está configurada` | Completar `.env` |
| Siempre "No encontré recetas" | Subir `RETRIEVAL_MAX_DISTANCE` en `.env` |
| Primera consulta lenta (~10s) | Normal: carga embeddings; el warmup al inicio lo mitiga |
| `ModuleNotFoundError: backend` | Ejecutar desde la raíz del proyecto |

---

## Herramientas LLM (`agents.py`)

Usadas por el pipeline en casos específicos (no para elegir la receta principal):

| Herramienta | Cuándo se usa |
|---|---|
| `scaling_expert` | "Adaptar para 10 porciones" |
| `substitution_expert` | "No tengo huevo / alternativa vegana" |
| `recipe_retriever` | Disponible para uso con agente ReAct; el flujo principal usa `pipeline.py` |

Prompts y respuestas en **español**.
