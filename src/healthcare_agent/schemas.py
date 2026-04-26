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
    routed_agent_names: list[str] = Field(default_factory=list)
    route_reason: str = ""
    rewritten_query: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_time_seconds: float = 0
    source_summary: str | None = None
    preprocessed_text: str | None = None
    preprocessing_notes: list[str] = Field(default_factory=list)
    knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)
    coarse_knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)
    reranked_knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)
    specialist_assessments: list["SpecialistAssessment"] = Field(default_factory=list)


class AssessmentResult(BaseModel):
    agent_name: str
    routed_agent_names: list[str] = Field(default_factory=list)
    route_reason: str = ""
    content: str
    rewritten_query: str | None = None
    usage: TokenUsage
    reasoning_time_seconds: float = 0
    knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)
    coarse_knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)
    reranked_knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)
    specialist_assessments: list["SpecialistAssessment"] = Field(default_factory=list)


class PreprocessedInput(BaseModel):
    medical_data: str
    source_summary: str
    notes: list[str] = Field(default_factory=list)


class RouterDecision(BaseModel):
    agent_names: list[str] = Field(default_factory=list)
    reason: str = ""


class SpecialistAssessment(BaseModel):
    agent_name: str
    agent_label: str
    content: str
    usage: TokenUsage
    reasoning_time_seconds: float = 0
    knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)
    coarse_knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)
    reranked_knowledge_chunks: list["KnowledgeChunk"] = Field(default_factory=list)


class KnowledgeChunk(BaseModel):
    source_file: str
    section_path: str
    content: str
    score: float | None = None
    vector_score: float | None = None
    rerank_score: float | None = None


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


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str | None = Field(default=None, max_length=100)


class UserLoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class UserProfile(BaseModel):
    id: int
    username: str
    display_name: str
    created_at: str | None = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="新的健康评估", max_length=180)
    mode: str = Field(default="specialist", pattern="^(specialist|general)$")


class ConversationUpdateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=180)
    mode: str | None = Field(default=None, pattern="^(specialist|general)$")


class ConversationSummary(BaseModel):
    id: int
    title: str
    mode: str
    created_at: str | None = None
    updated_at: str | None = None
    last_message: str | None = None


class StoredMessage(BaseModel):
    id: int
    role: str
    content: str
    metadata: dict | list | str | int | float | bool | None = None
    created_at: str | None = None


class ConversationDetail(BaseModel):
    conversation: ConversationSummary
    messages: list[StoredMessage] = Field(default_factory=list)


class ChatTurnRequest(BaseModel):
    content: str = Field(..., min_length=1)
    mode: str | None = Field(default=None, pattern="^(specialist|general)$")


class ChatTurnResponse(BaseModel):
    conversation: ConversationSummary
    user_message: StoredMessage
    assistant_message: StoredMessage
