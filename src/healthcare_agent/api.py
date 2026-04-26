from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pymysql
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from healthcare_agent.auth import create_access_token, get_current_user, hash_password, verify_password
from healthcare_agent.chat_service import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    list_messages,
    run_chat_turn,
    update_conversation,
)
from healthcare_agent.database import ensure_database_schema, get_connection
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
    AuthResponse,
    ChatTurnRequest,
    ChatTurnResponse,
    ConversationCreateRequest,
    ConversationDetail,
    ConversationSummary,
    ConversationUpdateRequest,
    KnowledgeBaseRebuildRequest,
    KnowledgeBaseRebuildResponse,
    KnowledgeBaseStatusResponse,
    UserCreateRequest,
    UserLoginRequest,
    UserProfile,
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


@app.on_event("startup")
def startup() -> None:
    ensure_database_schema()


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


def to_user_profile(row: dict) -> UserProfile:
    return UserProfile(
        id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        created_at=str(row.get("created_at")) if row.get("created_at") is not None else None,
    )


def normalize_conversation(row: dict) -> dict:
    normalized = dict(row)
    if normalized.get("created_at") is not None:
        normalized["created_at"] = str(normalized["created_at"])
    if normalized.get("updated_at") is not None:
        normalized["updated_at"] = str(normalized["updated_at"])
    return normalized


def normalize_message(row: dict) -> dict:
    normalized = dict(row)
    if normalized.get("created_at") is not None:
        normalized["created_at"] = str(normalized["created_at"])
    if "metadata" not in normalized:
        normalized["metadata"] = None
    return normalized


@app.post("/api/v1/auth/register", response_model=AuthResponse)
def register_user(request: UserCreateRequest) -> AuthResponse:
    username = request.username.strip()
    display_name = (request.display_name or username).strip() or username
    try:
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (username, password_hash, display_name)
                    VALUES (%s, %s, %s)
                    """,
                    (username, hash_password(request.password), display_name),
                )
                user_id = cursor.lastrowid
                cursor.execute(
                    "SELECT id, username, display_name, created_at FROM users WHERE id=%s",
                    (user_id,),
                )
                user = cursor.fetchone()
    except pymysql.err.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Username already exists.") from exc

    return AuthResponse(
        access_token=create_access_token(user["id"], user["username"]),
        user=to_user_profile(user),
    )


@app.post("/api/v1/auth/login", response_model=AuthResponse)
def login_user(request: UserLoginRequest) -> AuthResponse:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, username, password_hash, display_name, created_at FROM users WHERE username=%s",
                (request.username.strip(),),
            )
            user = cursor.fetchone()

    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    return AuthResponse(
        access_token=create_access_token(user["id"], user["username"]),
        user=to_user_profile(user),
    )


@app.get("/api/v1/auth/me", response_model=UserProfile)
def read_current_user(current_user: dict = Depends(get_current_user)) -> UserProfile:
    return to_user_profile(current_user)


@app.get("/api/v1/conversations", response_model=list[ConversationSummary])
def read_conversations(current_user: dict = Depends(get_current_user)) -> list[dict]:
    return [normalize_conversation(row) for row in list_conversations(current_user["id"])]


@app.post("/api/v1/conversations", response_model=ConversationSummary)
def create_user_conversation(
    request: ConversationCreateRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    conversation = create_conversation(
        current_user["id"],
        request.title,
        request.mode,  # type: ignore[arg-type]
    )
    return normalize_conversation(conversation)


@app.get("/api/v1/conversations/{conversation_id}", response_model=ConversationDetail)
def read_conversation_detail(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
) -> dict:
    conversation = normalize_conversation(get_conversation(current_user["id"], conversation_id))
    messages = [normalize_message(row) for row in list_messages(current_user["id"], conversation_id)]
    return {"conversation": conversation, "messages": messages}


@app.patch("/api/v1/conversations/{conversation_id}", response_model=ConversationSummary)
def update_user_conversation(
    conversation_id: int,
    request: ConversationUpdateRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    conversation = update_conversation(
        current_user["id"],
        conversation_id,
        title=request.title,
        mode=request.mode,  # type: ignore[arg-type]
    )
    return normalize_conversation(conversation)


@app.delete("/api/v1/conversations/{conversation_id}")
def delete_user_conversation(
    conversation_id: int,
    current_user: dict = Depends(get_current_user),
) -> dict[str, bool]:
    delete_conversation(current_user["id"], conversation_id)
    return {"deleted": True}


@app.post("/api/v1/conversations/{conversation_id}/messages", response_model=ChatTurnResponse)
async def create_conversation_message(
    conversation_id: int,
    request: ChatTurnRequest,
    current_user: dict = Depends(get_current_user),
) -> dict:
    result = await run_chat_turn(
        current_user["id"],
        conversation_id,
        request.content,
        mode=request.mode,  # type: ignore[arg-type]
    )
    return {
        "conversation": normalize_conversation(result["conversation"]),
        "user_message": normalize_message(result["user_message"]),
        "assistant_message": normalize_message(result["assistant_message"]),
    }


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
