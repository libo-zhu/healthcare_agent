from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from healthcare_agent.agent import run_healthcare_assessment, stream_healthcare_assessment
from healthcare_agent.schemas import AssessmentRequest, AssessmentResponse
from healthcare_agent.preprocessing import build_preprocessed_input


app = FastAPI(
    title="Healthcare Agent API",
    version="0.1.0",
    description="A small FastAPI service for healthcare assessment powered by LangChain + DeepSeek.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "Healthcare Agent API is running.",
        "docs": "/docs",
        "health_assessment": "/api/v1/health-assessment",
        "health_assessment_stream": "/api/v1/health-assessment/stream",
        "health_assessment_files": "/api/v1/health-assessment/files",
        "health_assessment_stream_files": "/api/v1/health-assessment/files/stream",
    }


@app.post("/api/v1/health-assessment", response_model=AssessmentResponse)
def create_health_assessment(request: AssessmentRequest) -> AssessmentResponse:
    try:
        result = run_healthcare_assessment(request.medical_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AssessmentResponse(
        content=result.content,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        total_tokens=result.usage.total_tokens,
        source_summary="inline_text",
        preprocessed_text=request.medical_data,
        preprocessing_notes=[],
    )


async def sse_event_generator(medical_data: str) -> AsyncIterator[str]:
    try:
        async for chunk in stream_healthcare_assessment(medical_data):
            payload = chunk
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except Exception as exc:
        payload = {"type": "error", "content": str(exc)}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return


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


@app.post("/api/v1/health-assessment/files", response_model=AssessmentResponse)
async def create_health_assessment_from_files(
    medical_data: str | None = Form(default=None),
    files: list[UploadFile] = File(default=[]),
) -> AssessmentResponse:
    try:
        preprocessed = await build_preprocessed_input(medical_data=medical_data, files=files)
        result = run_healthcare_assessment(preprocessed.medical_data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AssessmentResponse(
        content=result.content,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        total_tokens=result.usage.total_tokens,
        source_summary=preprocessed.source_summary,
        preprocessed_text=preprocessed.medical_data,
        preprocessing_notes=preprocessed.notes,
    )


@app.post("/api/v1/health-assessment/files/stream")
async def create_streaming_health_assessment_from_files(
    medical_data: str | None = Form(default=None),
    files: list[UploadFile] = File(default=[]),
) -> StreamingResponse:
    try:
        preprocessed = await build_preprocessed_input(medical_data=medical_data, files=files)
    except HTTPException as exc:
        payload = {"type": "error", "content": exc.detail}
        return StreamingResponse(
            iter([f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"]),
            media_type="text/event-stream",
        )

    async def file_sse_event_generator() -> AsyncIterator[str]:
        start_payload = {
            "type": "source",
            "source_summary": preprocessed.source_summary,
            "preprocessed_text": preprocessed.medical_data,
            "preprocessing_notes": preprocessed.notes,
        }
        yield f"data: {json.dumps(start_payload, ensure_ascii=False)}\n\n"
        async for event in sse_event_generator(preprocessed.medical_data):
            yield event

    return StreamingResponse(
        file_sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
