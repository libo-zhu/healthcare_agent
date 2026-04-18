from __future__ import annotations

import json
from collections.abc import AsyncIterator
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_deepseek import ChatDeepSeek

from healthcare_agent.config import DEFAULT_BASE_URL, DEFAULT_MODEL, get_deepseek_api_key
from healthcare_agent.knowledge_base import (
    format_knowledge_context,
    retrieve_knowledge,
)
from healthcare_agent.prompts import (
    ROUTER_AGENT_NAMES,
    build_general_health_prompt,
    build_query_rewrite_prompt,
    build_router_prompt,
    build_specialist_prompt,
)
from healthcare_agent.schemas import AssessmentResult, KnowledgeChunk, RouterDecision, TokenUsage


def build_chat_model() -> ChatDeepSeek:
    return ChatDeepSeek(
        model=DEFAULT_MODEL,
        api_key=get_deepseek_api_key(),
        base_url=DEFAULT_BASE_URL,
        temperature=0.2,
    )


def build_router_chain():
    model = build_chat_model()
    prompt = build_router_prompt()
    return prompt | model


def build_query_rewrite_chain():
    model = build_chat_model()
    prompt = build_query_rewrite_prompt()
    return prompt | model


def build_specialist_chain(agent_name: str):
    model = build_chat_model()
    prompt = build_specialist_prompt(agent_name)
    return prompt | model


def build_general_health_chain():
    model = build_chat_model()
    prompt = build_general_health_prompt()
    return prompt | model


def run_healthcare_assessment(medical_data: str) -> AssessmentResult:
    start_time = perf_counter()
    rewritten_query, rewrite_usage = rewrite_medical_data(medical_data)
    decision, router_usage = route_to_specialist(rewritten_query)
    retrieval = retrieve_knowledge(rewritten_query, agent_name=decision.agent_name)
    chain = build_specialist_chain(decision.agent_name)
    message = chain.invoke(
        {
            "medical_data": rewritten_query,
            "knowledge_context": format_knowledge_context(retrieval.final_chunks),
        }
    )
    return AssessmentResult(
        agent_name=decision.agent_name,
        content=extract_message_text(message),
        rewritten_query=rewritten_query,
        usage=add_token_usage(
            rewrite_usage,
            add_token_usage(router_usage, extract_token_usage(message)),
        ),
        reasoning_time_seconds=round(perf_counter() - start_time, 3),
        knowledge_chunks=to_schema_knowledge_chunks(retrieval.final_chunks),
        coarse_knowledge_chunks=to_schema_knowledge_chunks(retrieval.coarse_chunks),
        reranked_knowledge_chunks=to_schema_knowledge_chunks(retrieval.reranked_chunks),
    )


def run_general_health_assessment(medical_data: str) -> AssessmentResult:
    start_time = perf_counter()
    rewritten_query, rewrite_usage = rewrite_medical_data(medical_data)
    retrieval = retrieve_knowledge(rewritten_query)
    chain = build_general_health_chain()
    message = chain.invoke(
        {
            "medical_data": rewritten_query,
            "knowledge_context": format_knowledge_context(retrieval.final_chunks),
        }
    )
    return AssessmentResult(
        agent_name="general_health_overview",
        content=extract_message_text(message),
        rewritten_query=rewritten_query,
        usage=add_token_usage(rewrite_usage, extract_token_usage(message)),
        reasoning_time_seconds=round(perf_counter() - start_time, 3),
        knowledge_chunks=to_schema_knowledge_chunks(retrieval.final_chunks),
        coarse_knowledge_chunks=to_schema_knowledge_chunks(retrieval.coarse_chunks),
        reranked_knowledge_chunks=to_schema_knowledge_chunks(retrieval.reranked_chunks),
    )


async def stream_healthcare_assessment(medical_data: str) -> AsyncIterator[dict[str, Any]]:
    start_time = perf_counter()
    rewritten_query, rewrite_usage = rewrite_medical_data(medical_data)
    decision, router_usage = route_to_specialist(rewritten_query)
    retrieval = retrieve_knowledge(rewritten_query, agent_name=decision.agent_name)
    yield {
        "type": "rewrite",
        "rewritten_query": rewritten_query,
    }
    yield {
        "type": "route",
        "agent_name": decision.agent_name,
        "agent_label": ROUTER_AGENT_NAMES[decision.agent_name],
        "reason": decision.reason,
    }
    yield {
        "type": "knowledge",
        "chunks": [chunk.model_dump() for chunk in to_schema_knowledge_chunks(retrieval.final_chunks)],
        "coarse_chunks": [chunk.model_dump() for chunk in to_schema_knowledge_chunks(retrieval.coarse_chunks)],
        "reranked_chunks": [chunk.model_dump() for chunk in to_schema_knowledge_chunks(retrieval.reranked_chunks)],
    }

    chain = build_specialist_chain(decision.agent_name)
    final_usage = add_token_usage(rewrite_usage, router_usage)
    async for chunk in chain.astream(
        {
            "medical_data": rewritten_query,
            "knowledge_context": format_knowledge_context(retrieval.final_chunks),
        }
    ):
        chunk_text = extract_message_text(chunk)
        if chunk_text:
            yield {"type": "token", "content": chunk_text}

        chunk_usage = extract_token_usage(chunk)
        if chunk_usage.total_tokens:
            final_usage = add_token_usage(
                rewrite_usage,
                add_token_usage(router_usage, chunk_usage),
            )

    yield {
        "type": "done",
        "agent_name": decision.agent_name,
        "rewritten_query": rewritten_query,
        "input_tokens": final_usage.input_tokens,
        "output_tokens": final_usage.output_tokens,
        "total_tokens": final_usage.total_tokens,
        "reasoning_time_seconds": round(perf_counter() - start_time, 3),
    }


async def stream_general_health_assessment(medical_data: str) -> AsyncIterator[dict[str, Any]]:
    start_time = perf_counter()
    rewritten_query, rewrite_usage = rewrite_medical_data(medical_data)
    retrieval = retrieve_knowledge(rewritten_query)
    chain = build_general_health_chain()
    yield {
        "type": "rewrite",
        "rewritten_query": rewritten_query,
    }
    yield {
        "type": "route",
        "agent_name": "general_health_overview",
        "agent_label": "General Health Overview",
        "reason": "direct generalist assessment",
    }
    yield {
        "type": "knowledge",
        "chunks": [chunk.model_dump() for chunk in to_schema_knowledge_chunks(retrieval.final_chunks)],
        "coarse_chunks": [chunk.model_dump() for chunk in to_schema_knowledge_chunks(retrieval.coarse_chunks)],
        "reranked_chunks": [chunk.model_dump() for chunk in to_schema_knowledge_chunks(retrieval.reranked_chunks)],
    }

    final_usage = rewrite_usage
    async for chunk in chain.astream(
        {
            "medical_data": rewritten_query,
            "knowledge_context": format_knowledge_context(retrieval.final_chunks),
        }
    ):
        chunk_text = extract_message_text(chunk)
        if chunk_text:
            yield {"type": "token", "content": chunk_text}

        chunk_usage = extract_token_usage(chunk)
        if chunk_usage.total_tokens:
            final_usage = add_token_usage(rewrite_usage, chunk_usage)

    yield {
        "type": "done",
        "agent_name": "general_health_overview",
        "rewritten_query": rewritten_query,
        "input_tokens": final_usage.input_tokens,
        "output_tokens": final_usage.output_tokens,
        "total_tokens": final_usage.total_tokens,
        "reasoning_time_seconds": round(perf_counter() - start_time, 3),
    }


def route_to_specialist(medical_data: str) -> tuple[RouterDecision, TokenUsage]:
    chain = build_router_chain()
    message = chain.invoke({"medical_data": medical_data})
    decision = parse_router_decision(extract_message_text(message))
    usage = extract_token_usage(message)
    return decision, usage


def rewrite_medical_data(medical_data: str) -> tuple[str, TokenUsage]:
    chain = build_query_rewrite_chain()
    message = chain.invoke({"medical_data": medical_data})
    rewritten_query = extract_message_text(message).strip() or medical_data.strip()
    usage = extract_token_usage(message)
    return rewritten_query, usage


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


def add_token_usage(first: TokenUsage, second: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
    )


def parse_router_decision(raw_text: str) -> RouterDecision:
    normalized = raw_text.strip()
    try:
        payload = json.loads(extract_json_object(normalized))
    except json.JSONDecodeError:
        return fallback_router_decision(normalized)

    agent_name = payload.get("agent_name", "").strip()
    reason = str(payload.get("reason", "")).strip()
    if agent_name not in ROUTER_AGENT_NAMES:
        return fallback_router_decision(normalized)
    return RouterDecision(agent_name=agent_name, reason=reason)


def extract_json_object(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return raw_text
    return raw_text[start : end + 1]


def fallback_router_decision(raw_text: str) -> RouterDecision:
    lowered = raw_text.lower()
    for agent_name in ROUTER_AGENT_NAMES:
        if agent_name in lowered:
            return RouterDecision(agent_name=agent_name, reason="fallback parser matched agent name")
    keyword_groups = {
        "mental_social_health": [
            "焦虑", "抑郁", "压力", "情绪", "崩溃", "紧张", "失眠", "无助", "答辩", "毕业设计",
            "经济", "贫困", "家庭", "支持", "医保", "医疗保障", "工作压力", "学习压力",
        ],
        "cardiometabolic_health": [
            "血压", "高压", "低压", "血糖", "空腹血糖", "糖化", "hba1c", "fpg", "血脂",
            "胆固醇", "甘油三酯", "non-hdl", "高血压", "糖尿病", "内分泌", "甲状腺",
        ],
        "diet_bmi": [
            "bmi", "体重", "肥胖", "超重", "消瘦", "减肥", "减重", "增重", "饮食", "外卖",
            "热量", "营养", "高盐", "高糖", "高脂", "腰围",
        ],
        "sleep_activity_nicotine": [
            "睡眠", "熬夜", "早醒", "入睡", "打鼾", "运动", "久坐", "活动", "锻炼", "体能",
            "吸烟", "抽烟", "电子烟", "尼古丁", "二手烟",
        ],
    }

    for agent_name, keywords in keyword_groups.items():
        if any(keyword in lowered for keyword in keywords):
            return RouterDecision(agent_name=agent_name, reason="fallback keyword router matched input")

    return RouterDecision(
        agent_name="cardiometabolic_health",
        reason="fallback defaulted to cardiometabolic_health",
    )


def to_schema_knowledge_chunks(chunks: list[Any]) -> list[KnowledgeChunk]:
    return [
        KnowledgeChunk(
            source_file=chunk.source_file,
            section_path=chunk.section_path,
            content=chunk.content,
            score=chunk.score,
            vector_score=getattr(chunk, "vector_score", None),
            rerank_score=getattr(chunk, "rerank_score", None),
        )
        for chunk in chunks
    ]
