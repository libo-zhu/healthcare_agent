from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from healthcare_agent.agent import (
    arun_general_health_assessment,
    arun_healthcare_assessment,
    run_general_health_assessment,
    run_healthcare_assessment,
    stream_general_health_assessment,
    stream_healthcare_assessment,
)
from healthcare_agent.knowledge_base import build_knowledge_base_index, get_knowledge_base_status
from healthcare_agent.schemas import AssessmentRequest, AssessmentResponse
from healthcare_agent.schemas import (
    KnowledgeBaseRebuildRequest,
    KnowledgeBaseRebuildResponse,
    KnowledgeBaseStatusResponse,
)
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
        "specialist_assessment": "/api/v1/specialist/assessment",
        "specialist_assessment_stream": "/api/v1/specialist/assessment/stream",
        "specialist_assessment_files": "/api/v1/specialist/assessment/files",
        "specialist_assessment_stream_files": "/api/v1/specialist/assessment/files/stream",
        "general_assessment": "/api/v1/general/assessment",
        "general_assessment_stream": "/api/v1/general/assessment/stream",
        "general_assessment_files": "/api/v1/general/assessment/files",
        "general_assessment_stream_files": "/api/v1/general/assessment/files/stream",
        "knowledge_base_status": "/api/v1/knowledge-base/status",
        "knowledge_base_rebuild": "/api/v1/knowledge-base/rebuild",
    }


def build_assessment_response(
    result,
    source_summary: str,
    preprocessed_text: str,
    preprocessing_notes: list[str],
) -> AssessmentResponse:
    return AssessmentResponse(
        content=result.content,
        agent_name=result.agent_name,
        routed_agent_names=result.routed_agent_names,
        route_reason=result.route_reason,
        rewritten_query=result.rewritten_query,
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
        total_tokens=result.usage.total_tokens,
        reasoning_time_seconds=result.reasoning_time_seconds,
        source_summary=source_summary,
        preprocessed_text=preprocessed_text,
        preprocessing_notes=preprocessing_notes,
        knowledge_chunks=result.knowledge_chunks,
        coarse_knowledge_chunks=result.coarse_knowledge_chunks,
        reranked_knowledge_chunks=result.reranked_knowledge_chunks,
        specialist_assessments=result.specialist_assessments,
    )


async def sse_event_generator(
    medical_data: str,
    stream_handler,
) -> AsyncIterator[str]:
    try:
        async for chunk in stream_handler(medical_data):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    except Exception as exc:
        payload = {"type": "error", "content": str(exc)}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return


def build_streaming_response(event_generator: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        event_generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/v1/specialist/assessment", response_model=AssessmentResponse)
def create_specialist_health_assessment(request: AssessmentRequest) -> AssessmentResponse:
    try:
        result = run_healthcare_assessment(request.medical_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return build_assessment_response(
        result=result,
        source_summary="inline_text",
        preprocessed_text=request.medical_data,
        preprocessing_notes=[],
    )


@app.post("/api/v1/specialist/assessment/stream")
async def create_streaming_specialist_health_assessment(
    request: AssessmentRequest,
) -> StreamingResponse:
    return build_streaming_response(
        sse_event_generator(request.medical_data, stream_healthcare_assessment),
    )


@app.post("/api/v1/specialist/assessment/files", response_model=AssessmentResponse)
async def create_specialist_health_assessment_from_files(
    medical_data: str | None = Form(default=None),
    files: list[UploadFile] = File(default=[]),
) -> AssessmentResponse:
    try:
        preprocessed = await build_preprocessed_input(medical_data=medical_data, files=files)
        result = await arun_healthcare_assessment(preprocessed.medical_data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return build_assessment_response(
        result=result,
        source_summary=preprocessed.source_summary,
        preprocessed_text=preprocessed.medical_data,
        preprocessing_notes=preprocessed.notes,
    )


@app.post("/api/v1/specialist/assessment/files/stream")
async def create_streaming_specialist_health_assessment_from_files(
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
        async for event in sse_event_generator(preprocessed.medical_data, stream_healthcare_assessment):
            yield event

    return build_streaming_response(
        file_sse_event_generator(),
    )


@app.get("/api/v1/knowledge-base/status", response_model=KnowledgeBaseStatusResponse)
def read_knowledge_base_status() -> KnowledgeBaseStatusResponse:
    try:
        status = get_knowledge_base_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return KnowledgeBaseStatusResponse(**status.model_dump())


@app.post("/api/v1/knowledge-base/rebuild", response_model=KnowledgeBaseRebuildResponse)
def rebuild_knowledge_base(
    request: KnowledgeBaseRebuildRequest,
) -> KnowledgeBaseRebuildResponse:
    try:
        build_result = build_knowledge_base_index(force_rebuild=request.force_rebuild)
        status = get_knowledge_base_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return KnowledgeBaseRebuildResponse(
        message=(
            f"Knowledge base index rebuilt successfully. Indexed {build_result.indexed_files} files "
            f"and {build_result.indexed_chunks} chunks."
        ),
        **status.model_dump(),
    )


@app.post("/api/v1/general/assessment", response_model=AssessmentResponse)
def create_general_health_assessment(request: AssessmentRequest) -> AssessmentResponse:
    try:
        result = run_general_health_assessment(request.medical_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return build_assessment_response(
        result=result,
        source_summary="inline_text",
        preprocessed_text=request.medical_data,
        preprocessing_notes=[],
    )


@app.post("/api/v1/general/assessment/stream")
async def create_streaming_general_health_assessment(
    request: AssessmentRequest,
) -> StreamingResponse:
    return build_streaming_response(
        sse_event_generator(request.medical_data, stream_general_health_assessment),
    )


@app.post("/api/v1/general/assessment/files", response_model=AssessmentResponse)
async def create_general_health_assessment_from_files(
    medical_data: str | None = Form(default=None),
    files: list[UploadFile] = File(default=[]),
) -> AssessmentResponse:
    try:
        preprocessed = await build_preprocessed_input(medical_data=medical_data, files=files)
        result = await arun_general_health_assessment(preprocessed.medical_data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return build_assessment_response(
        result=result,
        source_summary=preprocessed.source_summary,
        preprocessed_text=preprocessed.medical_data,
        preprocessing_notes=preprocessed.notes,
    )


@app.post("/api/v1/general/assessment/files/stream")
async def create_streaming_general_health_assessment_from_files(
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
        async for event in sse_event_generator(
            preprocessed.medical_data,
            stream_general_health_assessment,
        ):
            yield event

    return build_streaming_response(
        file_sse_event_generator(),
    )
