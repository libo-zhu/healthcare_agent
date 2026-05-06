from __future__ import annotations

import json
import logging
from typing import Any, Literal

from fastapi import HTTPException, status

from healthcare_agent.agent import arun_general_health_assessment, arun_healthcare_assessment
from healthcare_agent.database import get_connection


ConversationMode = Literal["specialist", "general"]
MAX_CONTEXT_MESSAGES = 8
logger = logging.getLogger(__name__)


def serialize_metadata(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def deserialize_metadata(value: str | bytes | None) -> Any:
    if not value:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def create_conversation(user_id: int, title: str, mode: ConversationMode) -> dict[str, Any]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO conversations (user_id, title, mode) VALUES (%s, %s, %s)",
                (user_id, title[:180] or "新的健康评估", mode),
            )
            conversation_id = cursor.lastrowid
            cursor.execute(
                """
                SELECT id, title, mode, created_at, updated_at
                FROM conversations
                WHERE id=%s AND user_id=%s
                """,
                (conversation_id, user_id),
            )
            return cursor.fetchone()


def list_conversations(user_id: int) -> list[dict[str, Any]]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.id, c.title, c.mode, c.created_at, c.updated_at,
                       (
                         SELECT content FROM messages m
                         WHERE m.conversation_id = c.id
                         ORDER BY m.id DESC LIMIT 1
                       ) AS last_message
                FROM conversations c
                WHERE c.user_id=%s
                ORDER BY c.updated_at DESC, c.id DESC
                """,
                (user_id,),
            )
            return list(cursor.fetchall())


def get_conversation(user_id: int, conversation_id: int) -> dict[str, Any]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, title, mode, created_at, updated_at
                FROM conversations
                WHERE id=%s AND user_id=%s
                """,
                (conversation_id, user_id),
            )
            conversation = cursor.fetchone()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return conversation


def list_messages(user_id: int, conversation_id: int) -> list[dict[str, Any]]:
    get_conversation(user_id, conversation_id)
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, role, content, metadata_json, created_at
                FROM messages
                WHERE conversation_id=%s
                ORDER BY id ASC
                """,
                (conversation_id,),
            )
            rows = list(cursor.fetchall())
    for row in rows:
        row["metadata"] = deserialize_metadata(row.pop("metadata_json", None))
    return rows


def list_recent_context_messages(
    user_id: int,
    conversation_id: int,
    limit: int = MAX_CONTEXT_MESSAGES,
) -> list[dict[str, Any]]:
    get_conversation(user_id, conversation_id)
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, role, content, metadata_json, created_at
                FROM (
                    SELECT id, role, content, metadata_json, created_at
                    FROM messages
                    WHERE conversation_id=%s
                    ORDER BY id DESC
                    LIMIT %s
                ) recent_messages
                ORDER BY id ASC
                """,
                (conversation_id, max(limit, 1)),
            )
            rows = list(cursor.fetchall())
    for row in rows:
        row["metadata"] = deserialize_metadata(row.pop("metadata_json", None))
    return rows


def update_conversation(
    user_id: int,
    conversation_id: int,
    *,
    title: str | None = None,
    mode: ConversationMode | None = None,
) -> dict[str, Any]:
    get_conversation(user_id, conversation_id)
    updates: list[str] = []
    params: list[Any] = []
    if title is not None:
        updates.append("title=%s")
        params.append(title[:180] or "新的健康评估")
    if mode is not None:
        updates.append("mode=%s")
        params.append(mode)
    if updates:
        params.extend([conversation_id, user_id])
        with get_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE conversations SET {', '.join(updates)} WHERE id=%s AND user_id=%s",
                    params,
                )
    return get_conversation(user_id, conversation_id)


def delete_conversation(user_id: int, conversation_id: int) -> None:
    get_conversation(user_id, conversation_id)
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM conversations WHERE id=%s AND user_id=%s",
                (conversation_id, user_id),
            )


def insert_message(
    conversation_id: int,
    role: Literal["user", "assistant", "system"],
    content: str,
    metadata: Any | None = None,
) -> int:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO messages (conversation_id, role, content, metadata_json)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    None if metadata is None else serialize_metadata(metadata),
                ),
            )
            cursor.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=%s", (conversation_id,))
            return int(cursor.lastrowid)


def format_message_for_context(message: dict[str, Any]) -> str:
    role_label = "用户原始输入" if message["role"] == "user" else "Agent最终答复"
    content = str(message.get("content", "")).strip()
    return f"{role_label}: {content}"


def build_contextual_medical_data(history: list[dict[str, Any]], user_message: str) -> str:
    if not history:
        return user_message

    formatted_messages = [format_message_for_context(message) for message in history[-MAX_CONTEXT_MESSAGES:]]

    return "\n".join(
        [
            "以下是同一用户在当前健康评估对话中的历史上下文。请只把它作为理解用户当前问题的背景，不要虚构新的检查结果或病史。",
            "历史上下文只包含用户原始输入和 Agent 最终答复，不包含 rewrite、路由、分诊、检索或其他中间过程。",
            "",
            "历史对话：",
            "\n\n".join(formatted_messages),
            "",
            "用户当前最新输入：",
            user_message,
        ]
    )


def context_diagnostics(history: list[dict[str, Any]], contextual_input: str) -> dict[str, int]:
    user_chars = sum(len(str(message.get("content", ""))) for message in history if message.get("role") == "user")
    assistant_chars = sum(
        len(str(message.get("content", ""))) for message in history if message.get("role") == "assistant"
    )
    return {
        "history_messages": len(history),
        "history_user_chars": user_chars,
        "history_assistant_chars": assistant_chars,
        "contextual_input_chars": len(contextual_input),
    }


def result_to_metadata(result: Any, mode: ConversationMode) -> dict[str, Any]:
    return {
        "mode": mode,
        "agent_name": result.agent_name,
        "routed_agent_names": result.routed_agent_names,
        "route_reason": result.route_reason,
        "rewritten_query": result.rewritten_query,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "total_tokens": result.usage.total_tokens,
        "reasoning_time_seconds": result.reasoning_time_seconds,
        "knowledge_chunks": [chunk.model_dump() for chunk in result.knowledge_chunks],
        "coarse_knowledge_chunks": [chunk.model_dump() for chunk in result.coarse_knowledge_chunks],
        "reranked_knowledge_chunks": [chunk.model_dump() for chunk in result.reranked_knowledge_chunks],
        "specialist_assessments": [
            assessment.model_dump() for assessment in result.specialist_assessments
        ],
    }


async def run_chat_turn(
    user_id: int,
    conversation_id: int,
    user_message: str,
    mode: ConversationMode | None = None,
) -> dict[str, Any]:
    conversation = get_conversation(user_id, conversation_id)
    active_mode: ConversationMode = mode or conversation["mode"]
    if active_mode != conversation["mode"]:
        conversation = update_conversation(user_id, conversation_id, mode=active_mode)

    history = list_recent_context_messages(user_id, conversation_id)
    contextual_input = build_contextual_medical_data(history, user_message)
    logger.warning(
        "TOKEN_DIAG chat_context user_id=%s conversation_id=%s mode=%s %s",
        user_id,
        conversation_id,
        active_mode,
        context_diagnostics(history, contextual_input),
    )
    user_message_id = insert_message(conversation_id, "user", user_message)

    if len(history) == 0 and conversation["title"] == "新的健康评估":
        update_conversation(user_id, conversation_id, title=user_message[:32])

    if active_mode == "general":
        result = await arun_general_health_assessment(contextual_input)
    else:
        result = await arun_healthcare_assessment(contextual_input)

    metadata = result_to_metadata(result, active_mode)
    assistant_message_id = insert_message(
        conversation_id,
        "assistant",
        result.content,
        metadata=metadata,
    )
    return {
        "conversation": get_conversation(user_id, conversation_id),
        "user_message": {
            "id": user_message_id,
            "role": "user",
            "content": user_message,
            "metadata": None,
        },
        "assistant_message": {
            "id": assistant_message_id,
            "role": "assistant",
            "content": result.content,
            "metadata": metadata,
        },
    }
