from __future__ import annotations

import os
from pathlib import Path


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
DEFAULT_CHROMA_PERSIST_DIR = PROJECT_ROOT / ".chroma" / "medical_kb"
DEFAULT_VECTOR_COLLECTION_NAME = "medical_knowledge"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RAG_TOP_K = 5
DEFAULT_RAG_COARSE_TOP_K = 20
DEFAULT_RAG_RERANK_TOP_K = 5


def load_dotenv(dotenv_path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    path = Path(dotenv_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_deepseek_api_key() -> str:
    load_dotenv()
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Please set it in your environment or in a local .env file."
        )
    return api_key


def get_bool_env(name: str, default: bool) -> bool:
    load_dotenv()
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_int_env(name: str, default: int) -> int:
    load_dotenv()
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value.strip())
    except ValueError:
        return default
