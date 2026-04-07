from pydantic import BaseModel, Field


class AssessmentRequest(BaseModel):
    medical_data: str = Field(..., min_length=1, description="User-provided medical data.")


class AssessmentResponse(BaseModel):
    content: str

