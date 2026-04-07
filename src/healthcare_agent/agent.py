from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_deepseek import ChatDeepSeek

from healthcare_agent.config import DEFAULT_BASE_URL, DEFAULT_MODEL, get_deepseek_api_key
from healthcare_agent.prompts import build_healthcare_prompt
from healthcare_agent.schemas import AssessmentResult, TokenUsage


def build_chat_model() -> ChatDeepSeek:
    return ChatDeepSeek(
        model=DEFAULT_MODEL,
        api_key=get_deepseek_api_key(),
        base_url=DEFAULT_BASE_URL,
        temperature=0.2,
    )


def build_agent_chain():
    model = build_chat_model()
    prompt = build_healthcare_prompt()
    return prompt | model


def run_healthcare_assessment(medical_data: str) -> AssessmentResult:
    chain = build_agent_chain()
    message = chain.invoke({"medical_data": medical_data})
    return AssessmentResult(
        content=extract_message_text(message),
        usage=extract_token_usage(message),
    )


async def stream_healthcare_assessment(medical_data: str) -> AsyncIterator[dict[str, Any]]:
    chain = build_agent_chain()
    final_usage = TokenUsage()
    async for chunk in chain.astream({"medical_data": medical_data}):
        chunk_text = extract_message_text(chunk)
        if chunk_text:
            yield {"type": "token", "content": chunk_text}

        chunk_usage = extract_token_usage(chunk)
        if chunk_usage.total_tokens:
            final_usage = chunk_usage

    yield {
        "type": "done",
        "input_tokens": final_usage.input_tokens,
        "output_tokens": final_usage.output_tokens,
        "total_tokens": final_usage.total_tokens,
    }


def extract_message_text(message: BaseMessage | AIMessageChunk | str) -> str:
    if isinstance(message, str):
        return message

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        return "".join(text_parts)
    return str(content)


def extract_token_usage(message: BaseMessage | AIMessage | AIMessageChunk) -> TokenUsage:
    usage_metadata = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {})

    input_tokens = int(
        usage_metadata.get("input_tokens")
        or token_usage.get("prompt_tokens")
        or 0
    )
    output_tokens = int(
        usage_metadata.get("output_tokens")
        or token_usage.get("completion_tokens")
        or 0
    )
    total_tokens = int(
        usage_metadata.get("total_tokens")
        or token_usage.get("total_tokens")
        or input_tokens + output_tokens
    )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
