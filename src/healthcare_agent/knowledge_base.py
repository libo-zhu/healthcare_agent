from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field

from healthcare_agent.config import (
    DEFAULT_CHROMA_PERSIST_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_KNOWLEDGE_BASE_DIR,
    DEFAULT_RAG_COARSE_TOP_K,
    DEFAULT_RAG_RERANK_TOP_K,
    DEFAULT_RAG_TOP_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_VECTOR_COLLECTION_NAME,
    get_bool_env,
    get_int_env,
    load_dotenv,
)

SPECIALIST_AGENT_NAMES = {
    "sleep_activity_nicotine",
    "diet_bmi",
    "cardiometabolic_health",
    "mental_social_health",
}

_embedding_model = None
_reranker_model = None
_chroma_client = None
_embedding_model_lock = Lock()
_reranker_model_lock = Lock()
_chroma_client_lock = Lock()


class RetrievedKnowledgeChunk(BaseModel):
    source_file: str
    section_path: str
    content: str
    score: float | None = None
    vector_score: float | None = None
    rerank_score: float | None = None


class RetrievalResult(BaseModel):
    coarse_chunks: list[RetrievedKnowledgeChunk] = Field(default_factory=list)
    reranked_chunks: list[RetrievedKnowledgeChunk] = Field(default_factory=list)

    @property
    def final_chunks(self) -> list[RetrievedKnowledgeChunk]:
        return self.reranked_chunks


class KnowledgeBaseBuildResult(BaseModel):
    knowledge_base_dir: str
    persist_dir: str
    collection_name: str
    embedding_model: str
    indexed_files: int = 0
    indexed_chunks: int = 0


class KnowledgeBaseStatus(BaseModel):
    enabled: bool
    knowledge_base_dir: str
    persist_dir: str
    collection_name: str
    embedding_model: str
    index_exists: bool
    indexed_chunks: int = 0


def is_rag_enabled() -> bool:
    return get_bool_env("RAG_ENABLED", True)


def should_auto_build_index() -> bool:
    return get_bool_env("RAG_AUTO_BUILD", True)


def get_rag_top_k() -> int:
    return get_int_env("RAG_TOP_K", DEFAULT_RAG_TOP_K)


def get_rag_coarse_top_k() -> int:
    return get_int_env("RAG_COARSE_TOP_K", DEFAULT_RAG_COARSE_TOP_K)


def get_rag_rerank_top_k() -> int:
    return get_int_env("RAG_RERANK_TOP_K", DEFAULT_RAG_RERANK_TOP_K)


def get_embedding_model_name() -> str:
    load_dotenv()
    return os.environ.get("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)


def get_reranker_model_name() -> str:
    load_dotenv()
    return os.environ.get("RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL)


def get_knowledge_base_dir() -> Path:
    load_dotenv()
    return Path(os.environ.get("MEDICAL_KNOWLEDGE_BASE_DIR", str(DEFAULT_KNOWLEDGE_BASE_DIR))).expanduser()


def get_chroma_persist_dir() -> Path:
    load_dotenv()
    return Path(os.environ.get("CHROMA_PERSIST_DIR", str(DEFAULT_CHROMA_PERSIST_DIR))).expanduser()


def get_vector_collection_name() -> str:
    load_dotenv()
    return os.environ.get("VECTOR_COLLECTION_NAME", DEFAULT_VECTOR_COLLECTION_NAME)


def ensure_supported_dependencies() -> None:
    try:
        import chromadb  # noqa: F401
        from sentence_transformers import CrossEncoder  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "RAG dependencies are missing. Please install `chromadb` and `sentence-transformers` first."
        ) from exc


def get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    with _embedding_model_lock:
        if _embedding_model is not None:
            return _embedding_model

        ensure_supported_dependencies()
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(get_embedding_model_name())
        return _embedding_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    with _reranker_model_lock:
        if _reranker_model is not None:
            return _reranker_model

        ensure_supported_dependencies()
        from sentence_transformers import CrossEncoder

        _reranker_model = CrossEncoder(get_reranker_model_name())
        return _reranker_model


def get_chroma_client():
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    with _chroma_client_lock:
        if _chroma_client is not None:
            return _chroma_client

        ensure_supported_dependencies()
        import chromadb

        persist_dir = get_chroma_persist_dir()
        persist_dir.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        return _chroma_client


def clear_rag_caches() -> None:
    global _embedding_model, _reranker_model, _chroma_client

    with _embedding_model_lock:
        _embedding_model = None

    with _reranker_model_lock:
        _reranker_model = None

    with _chroma_client_lock:
        _chroma_client = None


def warmup_rag_dependencies() -> None:
    if not is_rag_enabled():
        return

    get_chroma_client()
    ensure_supported_dependencies()
    get_embedding_model()
    get_reranker_model()


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=get_vector_collection_name(),
        metadata={"hnsw:space": "cosine"},
    )


def build_knowledge_base_index(force_rebuild: bool = True) -> KnowledgeBaseBuildResult:
    if not is_rag_enabled():
        raise RuntimeError("RAG is disabled. Set RAG_ENABLED=true to build the index.")

    knowledge_base_dir = get_knowledge_base_dir()
    if not knowledge_base_dir.exists():
        raise RuntimeError(
            f"Knowledge base directory does not exist: {knowledge_base_dir}. "
            "Please put your JSON files there or set MEDICAL_KNOWLEDGE_BASE_DIR."
        )

    documents = load_json_documents(knowledge_base_dir)
    if not documents:
        raise RuntimeError(f"No JSON documents found under {knowledge_base_dir}.")

    clear_rag_caches()
    client = get_chroma_client()
    collection_name = get_vector_collection_name()
    if force_rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = get_collection()

    model = get_embedding_model()
    texts = [item["content"] for item in documents]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    metadatas = [
        {
            "source_file": item["source_file"],
            "section_path": item["section_path"],
            "agent_name": item["agent_name"],
        }
        for item in documents
    ]
    ids = [item["id"] for item in documents]

    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return KnowledgeBaseBuildResult(
        knowledge_base_dir=str(knowledge_base_dir),
        persist_dir=str(get_chroma_persist_dir()),
        collection_name=collection_name,
        embedding_model=get_embedding_model_name(),
        indexed_files=count_json_files(knowledge_base_dir),
        indexed_chunks=len(documents),
    )


def get_knowledge_base_status() -> KnowledgeBaseStatus:
    enabled = is_rag_enabled()
    knowledge_base_dir = get_knowledge_base_dir()
    persist_dir = get_chroma_persist_dir()
    collection_name = get_vector_collection_name()
    embedding_model = get_embedding_model_name()

    indexed_chunks = 0
    index_exists = False
    if enabled:
        try:
            collection = get_collection()
            indexed_chunks = collection.count()
            index_exists = indexed_chunks > 0
        except Exception:
            index_exists = False

    return KnowledgeBaseStatus(
        enabled=enabled,
        knowledge_base_dir=str(knowledge_base_dir),
        persist_dir=str(persist_dir),
        collection_name=collection_name,
        embedding_model=embedding_model,
        index_exists=index_exists,
        indexed_chunks=indexed_chunks,
    )


def retrieve_knowledge(
    query: str,
    agent_name: str | None = None,
    top_k: int | None = None,
) -> RetrievalResult:
    if not is_rag_enabled():
        return RetrievalResult()

    ensure_index_ready()
    collection = get_collection()
    if collection.count() == 0:
        return RetrievalResult()

    search_query = build_search_query(query=query, agent_name=agent_name)
    query_embedding = get_embedding_model().encode(
        [search_query], normalize_embeddings=True
    ).tolist()[0]
    coarse_top_k = min(collection.count(), max(top_k or get_rag_coarse_top_k(), 1))
    query_params: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": coarse_top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if agent_name:
        query_params["where"] = {"agent_name": agent_name}
    result = collection.query(**query_params)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    coarse_chunks: list[RetrievedKnowledgeChunk] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        vector_score = None if distance is None else max(0.0, 1.0 - float(distance))
        coarse_chunks.append(
            RetrievedKnowledgeChunk(
                source_file=str(metadata.get("source_file", "")),
                section_path=str(metadata.get("section_path", "")),
                content=str(document),
                score=vector_score,
                vector_score=vector_score,
            )
        )
    reranked_chunks = rerank_knowledge_chunks(
        query=search_query,
        chunks=coarse_chunks,
        top_k=min(len(coarse_chunks), max(top_k or get_rag_rerank_top_k(), 1)),
    )
    return RetrievalResult(
        coarse_chunks=coarse_chunks,
        reranked_chunks=reranked_chunks,
    )


def rerank_knowledge_chunks(
    query: str,
    chunks: list[RetrievedKnowledgeChunk],
    top_k: int,
) -> list[RetrievedKnowledgeChunk]:
    if not chunks or top_k <= 0:
        return []

    model = get_reranker_model()
    pairs = [(query, chunk.content) for chunk in chunks]
    scores = model.predict(pairs, batch_size=8, show_progress_bar=False)

    reranked = [
        chunk.model_copy(
            update={
                "score": float(score),
                "rerank_score": float(score),
            }
        )
        for chunk, score in zip(chunks, scores)
    ]
    reranked.sort(
        key=lambda chunk: chunk.rerank_score if chunk.rerank_score is not None else float("-inf"),
        reverse=True,
    )
    return reranked[:top_k]


def format_knowledge_context(chunks: list[RetrievedKnowledgeChunk]) -> str:
    if not chunks:
        return "未检索到可用知识库依据。若知识库没有覆盖该问题，请明确说明“知识库依据不足”，不要补充知识库外结论。"

    sections: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            "\n".join(
                [
                    f"[来源{index}] 文件: {chunk.source_file}",
                    f"[来源{index}] 路径: {chunk.section_path}",
                    f"[来源{index}] 内容: {chunk.content}",
                ]
            )
        )
    return "\n\n".join(sections)


def build_search_query(query: str, agent_name: str | None = None) -> str:
    if not agent_name:
        return query
    return f"评估方向: {agent_name}\n用户问题:\n{query}"


def ensure_index_ready() -> None:
    try:
        collection = get_collection()
        if collection.count() > 0:
            return
    except Exception:
        pass

    if should_auto_build_index():
        build_knowledge_base_index(force_rebuild=False)


def load_json_documents(base_dir: Path) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for json_path in sorted(base_dir.rglob("*.json")):
        content = json.loads(json_path.read_text(encoding="utf-8"))
        sanitized_content, agent_tags = extract_document_payload(content)
        relative_path = str(json_path.relative_to(base_dir))
        sections = extract_sections_from_json(sanitized_content)
        effective_tags = agent_tags or [""]
        for agent_tag in effective_tags:
            tag_suffix = agent_tag or "all_agents"
            for section_index, section in enumerate(sections):
                for chunk_index, chunk in enumerate(split_text(section["content"])):
                    documents.append(
                        {
                            "id": f"{relative_path}::{tag_suffix}::{section_index}::{chunk_index}",
                            "source_file": relative_path,
                            "section_path": section["section_path"],
                            "content": chunk,
                            "agent_name": agent_tag,
                        }
                    )
    return documents


def extract_document_payload(payload: Any) -> tuple[Any, list[str]]:
    if not isinstance(payload, dict):
        return payload, []

    sanitized_payload = dict(payload)
    raw_agent_tags = sanitized_payload.pop("agent_tags", [])
    return sanitized_payload, normalize_agent_tags(raw_agent_tags)


def normalize_agent_tags(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        candidates = [raw_value]
    elif isinstance(raw_value, list):
        candidates = [str(item) for item in raw_value]
    else:
        candidates = []

    normalized: list[str] = []
    for candidate in candidates:
        tag = candidate.strip()
        if tag in SPECIALIST_AGENT_NAMES and tag not in normalized:
            normalized.append(tag)
    return normalized


def extract_sections_from_json(payload: Any, path: str = "root") -> list[dict[str, str]]:
    sections: list[dict[str, str]] = []

    if is_scalar(payload):
        sections.append({"section_path": path, "content": f"{path}: {scalar_to_text(payload)}"})
        return sections

    flattened_lines = flatten_json(payload)
    if 0 < len(flattened_lines) <= 18:
        sections.append({"section_path": path, "content": "\n".join(flattened_lines)})
        return sections

    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}"
            sections.extend(extract_sections_from_json(value, child_path))
        return sections

    if isinstance(payload, list):
        for index, value in enumerate(payload):
            child_path = f"{path}[{index}]"
            sections.extend(extract_sections_from_json(value, child_path))
        return sections

    return sections


def flatten_json(payload: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            lines.extend(flatten_json(value, child_prefix))
        return lines

    if isinstance(payload, list):
        for index, value in enumerate(payload):
            child_prefix = f"{prefix}[{index}]"
            lines.extend(flatten_json(value, child_prefix))
        return lines

    if prefix:
        lines.append(f"{prefix}: {scalar_to_text(payload)}")
    return lines


def split_text(text: str, max_chars: int = 700, overlap: int = 120) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def count_json_files(base_dir: Path) -> int:
    return sum(1 for _ in base_dir.rglob("*.json"))


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def scalar_to_text(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
