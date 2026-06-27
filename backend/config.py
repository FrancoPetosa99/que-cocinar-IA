"""Environment variables and factory functions for LLM / embeddings."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "recipes")

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
HF_BACKEND = os.getenv("HF_BACKEND", "inference_api").lower()

# Embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def get_embeddings():
    """Return a HuggingFace embedding model (runs locally)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm(*, streaming: bool = True) -> BaseChatModel:
    """
    Return a LangChain chat model based on LLM_PROVIDER.

    Supported providers:
      - gemini: Google Gemini via langchain-google-genai
      - huggingface: any HF model via Inference API or local pipeline
    """
    if LLM_PROVIDER == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY no está configurada. Agregala a tu archivo .env"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=api_key,
            temperature=TEMPERATURE,
            streaming=streaming,
        )

    if LLM_PROVIDER == "huggingface":
        return _get_huggingface_llm(streaming=streaming)

    raise ValueError(
        f"LLM_PROVIDER '{LLM_PROVIDER}' no soportado. "
        "Usá 'gemini' o 'huggingface'."
    )


def _get_huggingface_llm(*, streaming: bool) -> BaseChatModel:
    """Build a HuggingFace-backed chat model (hosted API or local)."""
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if HF_BACKEND == "local":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map="auto",
            torch_dtype="auto",
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=TEMPERATURE,
            return_full_text=False,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return ChatHuggingFace(llm=llm, verbose=True)

    if not token:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN no está configurada. "
            "Necesaria para HF_BACKEND=inference_api."
        )

    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=token,
        temperature=TEMPERATURE,
        streaming=streaming,
    )
    return ChatHuggingFace(llm=endpoint, verbose=True)


def validate_chroma_exists() -> None:
    """Raise a clear error if the vector store has not been built yet."""
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        raise FileNotFoundError(
            f"No se encontró la base vectorial en '{CHROMA_DIR}'. "
            "Ejecutá primero data_preprocessing/preprocessing.ipynb "
            "para descargar e indexar las recetas."
        )
