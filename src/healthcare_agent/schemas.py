from pydantic import BaseModel, Field


class AssessmentRequest(BaseModel):
    medical_data: str = Field(..., min_length=1, description="User-provided medical data.")


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class AssessmentResponse(BaseModel):
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    source_summary: str | None = None
    preprocessed_text: str | None = None
    preprocessing_notes: list[str] = Field(default_factory=list)


class AssessmentResult(BaseModel):
    content: str
    usage: TokenUsage


class PreprocessedInput(BaseModel):
    medical_data: str
    source_summary: str
    notes: list[str] = Field(default_factory=list)
