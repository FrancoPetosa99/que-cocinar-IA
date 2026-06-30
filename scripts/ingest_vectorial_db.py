from pathlib import Path
import shutil
import sys
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CSV_PATH = ROOT / "data" / "enriched_recipes_spanish.csv"
CHROMA_DIR = ROOT / "chroma_db" 
COLLECTION_NAME = "recipes"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 500

def recipe_to_embedding_text(recipe: pd.Series) -> str:
    """
    Construye el documento semántico que será utilizado para generar
    el embedding de la receta.
    """

    return f"""
Receta:
{recipe["recipe_name"]}

Tipo de comida:
{recipe["meal_type"]}

Sabor:
{recipe["taste_profile"]}

Temperatura de servicio:
{recipe["served_temperature"]}

Estación:
{recipe["season"]}

Dificultad:
{recipe["difficulty"]}

Características:
{recipe["characteristics"]}

Ingredientes principales:
{recipe["main_ingredients"]}

Resumen semántico:
{recipe["semantic_summary"]}
""".strip()

def recipe_to_vector_document(recipe: pd.Series) -> Document:
    return Document(
        page_content=recipe_to_embedding_text(recipe),
        metadata={
            "csv_row_id": int(recipe["recipe_id"])
        }
    )

def ingest_vector():
    print(f"Loading {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"Recipes loaded: {len(df):,}")

    documents = [
        recipe_to_vector_document(recipe)
        for _, recipe in df.iterrows()
    ]

    print(f"Documents: {len(documents):,}")

    print(
        f"Sample document size: "
        f"{len(documents[0].page_content)} characters"
    )

    if CHROMA_DIR.exists():

        print(f"Removing existing database ({CHROMA_DIR})")

        shutil.rmtree(CHROMA_DIR)

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = None

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i+BATCH_SIZE]

        if vectorstore is None:

            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_DIR),
            )

        else:
            vectorstore.add_documents(batch)

        print(
            f"Indexed "
            f"{min(i + BATCH_SIZE, len(documents)):,}"
            f" / {len(documents):,}"
        )

    sample = vectorstore.similarity_search(
        "postre frío refrescante con fruta",
        k=1,
    )[0]

    print("\nVerification")
    print("----------------------")
    print(f"Recipe ID: {sample.metadata['csv_row_id']}")
    print(sample.page_content)

if __name__ == "__main__":
    ingest_vector()
    print("\nVector database created successfully.")