"""Environment variables and factory functions for LLM / embeddings."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_chroma_env = os.getenv("CHROMA_DIR")
if _chroma_env:
    _chroma_path = Path(_chroma_env)
    CHROMA_DIR = str(
        _chroma_path if _chroma_path.is_absolute() else PROJECT_ROOT / _chroma_path
    )
else:
    CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "recipes")

_sqlite_env = os.getenv("SQLITE_PATH")
if _sqlite_env:
    _sqlite_path = Path(_sqlite_env)
    SQLITE_PATH = str(
        _sqlite_path if _sqlite_path.is_absolute() else PROJECT_ROOT / _sqlite_path
    )
else:
    SQLITE_PATH = str(PROJECT_ROOT / "data" / "recipes.db")

# LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct"
    if os.getenv("LLM_PROVIDER", "").lower() == "huggingface"
    else "gemini-2.5-flash",
)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
HF_BACKEND = os.getenv("HF_BACKEND", "local").lower()
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
HF_DEVICE = os.getenv("HF_DEVICE", "auto").lower()

_hf_cache_env = os.getenv("HF_MODEL_CACHE_DIR")
if _hf_cache_env:
    _hf_cache_path = Path(_hf_cache_env)
    HF_MODEL_CACHE_DIR = str(
        _hf_cache_path
        if _hf_cache_path.is_absolute()
        else PROJECT_ROOT / _hf_cache_path
    )
else:
    HF_MODEL_CACHE_DIR = str(PROJECT_ROOT / "models")

# Embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

_llm_singleton: BaseChatModel | None = None

def get_embeddings():
    """Return a HuggingFace embedding model (runs locally)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def resolve_llm_model_path() -> str:
    """
    Return path or Hub id for the LLM.

    If LLM_MODEL is a Hub id (org/name) and a matching folder exists under
    models/, use the local copy downloaded by scripts/download_hf_model.py.
    """
    model_path = Path(LLM_MODEL)
    if model_path.exists():
        return str(model_path.resolve())

    if "/" in LLM_MODEL:
        local_name = LLM_MODEL.split("/")[-1]
        local_dir = Path(HF_MODEL_CACHE_DIR) / local_name
        if local_dir.exists() and any(local_dir.iterdir()):
            return str(local_dir.resolve())

    return LLM_MODEL

def _detect_torch_device() -> str:
    """Return 'cuda', 'mps', or 'cpu'."""
    import torch

    if HF_DEVICE == "cpu":
        return "cpu"
    if HF_DEVICE == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if HF_DEVICE == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_llm(*, streaming: bool = True) -> BaseChatModel:
    """
    Return a LangChain chat model based on LLM_PROVIDER.

    Supported providers:
      - gemini: Google Gemini via langchain-google-genai
      - huggingface: local pipeline (HF_BACKEND=local) or Inference API
    """
    global _llm_singleton
    if _llm_singleton is not None and LLM_PROVIDER == "huggingface" and HF_BACKEND == "local":
        return _llm_singleton

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
        llm = _get_huggingface_llm(streaming=streaming)
        if HF_BACKEND == "local":
            _llm_singleton = llm
        return llm

    raise ValueError(
        f"LLM_PROVIDER '{LLM_PROVIDER}' no soportado. "
        "Usá 'gemini' o 'huggingface'."
    )

def preload_local_llm() -> None:
    """Load local HF model into memory (call at app startup)."""
    if LLM_PROVIDER == "huggingface" and HF_BACKEND == "local":
        get_llm(streaming=False)

def _get_huggingface_llm(*, streaming: bool) -> BaseChatModel:
    """Build a HuggingFace-backed chat model (hosted API or local pipeline)."""
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    if HF_BACKEND == "local":
        return _get_huggingface_local_llm()

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
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

def _get_huggingface_local_llm() -> BaseChatModel:
    """Load a causal LM from disk or Hugging Face Hub and run locally."""
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    model_id = resolve_llm_model_path()
    device = _detect_torch_device()
    print(f"Loading local HF model: {model_id} (device={device})")

    model_kwargs: dict = {"cache_dir": HF_MODEL_CACHE_DIR}
    pipeline_device: int | None = -1

    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
        pipeline_device = None
    elif device == "mps":
        model_kwargs["torch_dtype"] = torch.float16
        pipeline_device = None
    else:
        model_kwargs["torch_dtype"] = torch.float32
        pipeline_device = -1

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        device=pipeline_device,
        model_kwargs=model_kwargs,
        pipeline_kwargs={
            "max_new_tokens": HF_MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "do_sample": TEMPERATURE > 0,
            "return_full_text": False,
        },
    )

    if device == "mps":
        llm.pipeline.model.to("mps")

    return ChatHuggingFace(llm=llm, verbose=True)

def validate_chroma_exists() -> None:
    """Raise a clear error if the vector store has not been built yet."""
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        raise FileNotFoundError(
            f"No se encontró la base vectorial en '{CHROMA_DIR}'. "
            "Ejecutá: python data_preprocessing/ingest.py"
        )
