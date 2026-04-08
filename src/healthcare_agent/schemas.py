from pydantic import BaseModel, Field


class AssessmentRequest(BaseModel):
    medical_data: str = Field(..., min_length=1, description="User-provided medical data.")


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AssessmentResponse(BaseModel):
    content: str
    agent_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    source_summary: str | None = None
    preprocessed_text: str | None = None
    preprocessing_notes: list[str] = Field(default_factory=list)
    knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)


class AssessmentResult(BaseModel):
    agent_name: str
    content: str
    usage: TokenUsage
    knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)


class PreprocessedInput(BaseModel):
    medical_data: str
    source_summary: str
    notes: list[str] = Field(default_factory=list)


class RouterDecision(BaseModel):
    agent_name: str
    reason: str = ""


class KnowledgeChunk(BaseModel):
    source_file: str
    section_path: str
    content: str
    score: float | None = None


class KnowledgeBaseRebuildRequest(BaseModel):
    force_rebuild: bool = True


class KnowledgeBaseStatusResponse(BaseModel):
    enabled: bool
    knowledge_base_dir: str
    persist_dir: str
    collection_name: str
    embedding_model: str
    index_exists: bool
    indexed_chunks: int = 0


class KnowledgeBaseRebuildResponse(BaseModel):
    message: str
    enabled: bool
    knowledge_base_dir: str
    persist_dir: str
    collection_name: str
    embedding_model: str
    index_exists: bool
    indexed_chunks: int = 0
