from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from healthcare_agent.agent import run_healthcare_assessment, stream_healthcare_assessment
from healthcare_agent.schemas import AssessmentRequest, AssessmentResponse


app = FastAPI(
    title="Healthcare Agent API",
    version="0.1.0",
    description="A small FastAPI service for healthcare assessment powered by LangChain + DeepSeek.",
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "Healthcare Agent API is running.",
        "docs": "/docs",
        "health_assessment": "/api/v1/health-assessment",
        "health_assessment_stream": "/api/v1/health-assessment/stream",
    }


@app.post("/api/v1/health-assessment", response_model=AssessmentResponse)
def create_health_assessment(request: AssessmentRequest) -> AssessmentResponse:
    try:
        content = run_healthcare_assessment(request.medical_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AssessmentResponse(content=content)


async def sse_event_generator(medical_data: str) -> AsyncIterator[str]:
    try:
        async for chunk in stream_healthcare_assessment(medical_data):
            payload = {"type": "token", "content": chunk}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except Exception as exc:
        payload = {"type": "error", "content": str(exc)}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return

    payload = {"type": "done"}
    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/v1/health-assessment/stream")
async def create_streaming_health_assessment(
    request: AssessmentRequest,
) -> StreamingResponse:
    return StreamingResponse(
        sse_event_generator(request.medical_data),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

