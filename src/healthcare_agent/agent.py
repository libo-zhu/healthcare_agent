from __future__ import annotations

import asyncio
import json
import logging
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
    build_specialist_summary_prompt,
)
from healthcare_agent.schemas import (
    AssessmentResult,
    KnowledgeChunk,
    RouterDecision,
    SpecialistAssessment,
    TokenUsage,
)


SUMMARY_AGENT_NAME = "specialist_summary"
SUMMARY_AGENT_LABEL = "Specialist Summary"
logger = logging.getLogger(__name__)


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


def build_specialist_summary_chain():
    model = build_chat_model()
    prompt = build_specialist_summary_prompt()
    return prompt | model


def build_general_health_chain():
    model = build_chat_model()
    prompt = build_general_health_prompt()
    return prompt | model


def run_healthcare_assessment(medical_data: str) -> AssessmentResult:
    return asyncio.run(arun_healthcare_assessment(medical_data))


async def arun_healthcare_assessment(medical_data: str) -> AssessmentResult:
    start_time = perf_counter()
    rewritten_query, rewrite_usage = await rewrite_medical_data(medical_data)
    logger.warning(
        "TOKEN_DIAG specialist_direct phase=rewrite input_chars=%s rewritten_chars=%s usage=%s",
        len(medical_data),
        len(rewritten_query),
        rewrite_usage.model_dump(),
    )
    decision, router_usage = await route_to_specialists(rewritten_query)
    logger.warning(
        "TOKEN_DIAG specialist_direct phase=router rewritten_chars=%s agents=%s usage=%s",
        len(rewritten_query),
        decision.agent_names,
        router_usage.model_dump(),
    )
    specialist_assessments = await run_parallel_specialist_assessments(
        rewritten_query,
        decision.agent_names,
    )
    specialist_assessments = order_specialist_assessments(
        specialist_assessments,
        decision.agent_names,
    )
    summary_content, summary_usage = await summarize_specialist_assessments(
        rewritten_query,
        specialist_assessments,
    )
    for assessment in specialist_assessments:
        logger.warning(
            "TOKEN_DIAG specialist_direct phase=specialist agent=%s usage=%s",
            assessment.agent_name,
            assessment.usage.model_dump(),
        )
    logger.warning(
        "TOKEN_DIAG specialist_direct phase=summary specialist_count=%s usage=%s",
        len(specialist_assessments),
        summary_usage.model_dump(),
    )
    combined_chunks = combine_knowledge_chunks(
        specialist_assessments,
        chunk_attr="knowledge_chunks",
    )
    combined_coarse_chunks = combine_knowledge_chunks(
        specialist_assessments,
        chunk_attr="coarse_knowledge_chunks",
    )
    combined_reranked_chunks = combine_knowledge_chunks(
        specialist_assessments,
        chunk_attr="reranked_knowledge_chunks",
    )
    total_usage = add_token_usages(
        rewrite_usage,
        router_usage,
        summary_usage,
        *(assessment.usage for assessment in specialist_assessments),
    )
    return AssessmentResult(
        agent_name=SUMMARY_AGENT_NAME,
        routed_agent_names=decision.agent_names,
        route_reason=decision.reason,
        content=summary_content,
        rewritten_query=rewritten_query,
        usage=total_usage,
        reasoning_time_seconds=round(perf_counter() - start_time, 3),
        knowledge_chunks=combined_chunks,
        coarse_knowledge_chunks=combined_coarse_chunks,
        reranked_knowledge_chunks=combined_reranked_chunks,
        specialist_assessments=specialist_assessments,
    )


def run_general_health_assessment(medical_data: str) -> AssessmentResult:
    return asyncio.run(arun_general_health_assessment(medical_data))


async def arun_general_health_assessment(medical_data: str) -> AssessmentResult:
    start_time = perf_counter()
    rewritten_query, rewrite_usage = await rewrite_medical_data(medical_data)
    logger.warning(
        "TOKEN_DIAG general_direct phase=rewrite input_chars=%s rewritten_chars=%s usage=%s",
        len(medical_data),
        len(rewritten_query),
        rewrite_usage.model_dump(),
    )
    retrieval = await asyncio.to_thread(retrieve_knowledge, rewritten_query)
    chain = build_general_health_chain()
    message = await chain.ainvoke(
        {
            "medical_data": rewritten_query,
            "knowledge_context": format_knowledge_context(retrieval.final_chunks),
        }
    )
    message_usage = extract_token_usage(message)
    logger.warning(
        "TOKEN_DIAG general_direct phase=answer rewritten_chars=%s chunks=%s usage=%s",
        len(rewritten_query),
        len(retrieval.final_chunks),
        message_usage.model_dump(),
    )
    return AssessmentResult(
        agent_name="general_health_overview",
        routed_agent_names=["general_health_overview"],
        route_reason="direct generalist assessment",
        content=extract_message_text(message),
        rewritten_query=rewritten_query,
        usage=add_token_usage(rewrite_usage, message_usage),
        reasoning_time_seconds=round(perf_counter() - start_time, 3),
        knowledge_chunks=to_schema_knowledge_chunks(retrieval.final_chunks),
        coarse_knowledge_chunks=to_schema_knowledge_chunks(retrieval.coarse_chunks),
        reranked_knowledge_chunks=to_schema_knowledge_chunks(retrieval.reranked_chunks),
    )


async def stream_healthcare_assessment(medical_data: str) -> AsyncIterator[dict[str, Any]]:
    start_time = perf_counter()
    rewritten_query, rewrite_usage = await rewrite_medical_data(medical_data)
    logger.warning(
        "TOKEN_DIAG specialist_stream phase=rewrite input_chars=%s rewritten_chars=%s usage=%s",
        len(medical_data),
        len(rewritten_query),
        rewrite_usage.model_dump(),
    )
    decision, router_usage = await route_to_specialists(rewritten_query)
    logger.warning(
        "TOKEN_DIAG specialist_stream phase=router rewritten_chars=%s agents=%s usage=%s",
        len(rewritten_query),
        decision.agent_names,
        router_usage.model_dump(),
    )
    yield {
        "type": "rewrite",
        "rewritten_query": rewritten_query,
    }
    yield {
        "type": "route",
        "agent_name": SUMMARY_AGENT_NAME,
        "agent_label": SUMMARY_AGENT_LABEL,
        "agent_names": decision.agent_names,
        "agent_labels": [ROUTER_AGENT_NAMES[name] for name in decision.agent_names],
        "reason": decision.reason,
    }

    specialist_tasks = [
        asyncio.create_task(run_specialist_assessment(rewritten_query, agent_name))
        for agent_name in decision.agent_names
    ]

    specialist_assessments: list[SpecialistAssessment] = []
    specialist_usage = TokenUsage()
    for task in asyncio.as_completed(specialist_tasks):
        assessment = await task
        specialist_assessments.append(assessment)
        specialist_usage = add_token_usage(specialist_usage, assessment.usage)
        logger.warning(
            "TOKEN_DIAG specialist_stream phase=specialist agent=%s usage=%s",
            assessment.agent_name,
            assessment.usage.model_dump(),
        )
        yield {
            "type": "knowledge",
            "agent_name": assessment.agent_name,
            "agent_label": assessment.agent_label,
            "chunks": [chunk.model_dump() for chunk in assessment.knowledge_chunks],
            "coarse_chunks": [chunk.model_dump() for chunk in assessment.coarse_knowledge_chunks],
            "reranked_chunks": [chunk.model_dump() for chunk in assessment.reranked_knowledge_chunks],
        }
        yield {
            "type": "specialist_result",
            "agent_name": assessment.agent_name,
            "agent_label": assessment.agent_label,
            "content": assessment.content,
            "input_tokens": assessment.usage.input_tokens,
            "output_tokens": assessment.usage.output_tokens,
            "total_tokens": assessment.usage.total_tokens,
            "reasoning_time_seconds": assessment.reasoning_time_seconds,
        }

    specialist_assessments = order_specialist_assessments(
        specialist_assessments,
        decision.agent_names,
    )
    combined_chunks = combine_knowledge_chunks(
        specialist_assessments,
        chunk_attr="knowledge_chunks",
    )
    combined_coarse_chunks = combine_knowledge_chunks(
        specialist_assessments,
        chunk_attr="coarse_knowledge_chunks",
    )
    combined_reranked_chunks = combine_knowledge_chunks(
        specialist_assessments,
        chunk_attr="reranked_knowledge_chunks",
    )
    summary_chain = build_specialist_summary_chain()
    summary_input = build_specialist_summary_input(rewritten_query, specialist_assessments)
    final_usage = add_token_usages(rewrite_usage, router_usage, specialist_usage)
    summary_text_parts: list[str] = []
    async for chunk in summary_chain.astream(summary_input):
        chunk_text = extract_message_text(chunk)
        if chunk_text:
            summary_text_parts.append(chunk_text)
            yield {"type": "token", "content": chunk_text}

        chunk_usage = extract_token_usage(chunk)
        if chunk_usage.total_tokens:
            final_usage = add_token_usages(
                rewrite_usage,
                router_usage,
                specialist_usage,
                chunk_usage,
            )

    logger.warning(
        "TOKEN_DIAG specialist_stream phase=summary specialist_count=%s total_usage_so_far=%s output_chars=%s",
        len(specialist_assessments),
        final_usage.model_dump(),
        len("".join(summary_text_parts)),
    )
    yield {
        "type": "done",
        "agent_name": SUMMARY_AGENT_NAME,
        "agent_label": SUMMARY_AGENT_LABEL,
        "routed_agent_names": decision.agent_names,
        "rewritten_query": rewritten_query,
        "content": "".join(summary_text_parts),
        "knowledge_chunks": [chunk.model_dump() for chunk in combined_chunks],
        "coarse_knowledge_chunks": [chunk.model_dump() for chunk in combined_coarse_chunks],
        "reranked_knowledge_chunks": [chunk.model_dump() for chunk in combined_reranked_chunks],
        "input_tokens": final_usage.input_tokens,
        "output_tokens": final_usage.output_tokens,
        "total_tokens": final_usage.total_tokens,
        "reasoning_time_seconds": round(perf_counter() - start_time, 3),
    }


async def stream_general_health_assessment(medical_data: str) -> AsyncIterator[dict[str, Any]]:
    start_time = perf_counter()
    rewritten_query, rewrite_usage = await rewrite_medical_data(medical_data)
    logger.warning(
        "TOKEN_DIAG general_stream phase=rewrite input_chars=%s rewritten_chars=%s usage=%s",
        len(medical_data),
        len(rewritten_query),
        rewrite_usage.model_dump(),
    )
    retrieval = await asyncio.to_thread(retrieve_knowledge, rewritten_query)
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

    logger.warning(
        "TOKEN_DIAG general_stream phase=answer rewritten_chars=%s chunks=%s total_usage=%s",
        len(rewritten_query),
        len(retrieval.final_chunks),
        final_usage.model_dump(),
    )
    yield {
        "type": "done",
        "agent_name": "general_health_overview",
        "rewritten_query": rewritten_query,
        "input_tokens": final_usage.input_tokens,
        "output_tokens": final_usage.output_tokens,
        "total_tokens": final_usage.total_tokens,
        "reasoning_time_seconds": round(perf_counter() - start_time, 3),
    }


async def route_to_specialists(medical_data: str) -> tuple[RouterDecision, TokenUsage]:
    chain = build_router_chain()
    message = await chain.ainvoke({"medical_data": medical_data})
    decision = parse_router_decision(extract_message_text(message))
    usage = extract_token_usage(message)
    return decision, usage


async def rewrite_medical_data(medical_data: str) -> tuple[str, TokenUsage]:
    chain = build_query_rewrite_chain()
    message = await chain.ainvoke({"medical_data": medical_data})
    rewritten_query = extract_message_text(message).strip() or medical_data.strip()
    usage = extract_token_usage(message)
    return rewritten_query, usage


def rewrite_medical_data_sync(medical_data: str) -> tuple[str, TokenUsage]:
    chain = build_query_rewrite_chain()
    message = chain.invoke({"medical_data": medical_data})
    rewritten_query = extract_message_text(message).strip() or medical_data.strip()
    usage = extract_token_usage(message)
    return rewritten_query, usage


async def run_parallel_specialist_assessments(
    rewritten_query: str,
    agent_names: list[str],
) -> list[SpecialistAssessment]:
    tasks = [
        run_specialist_assessment(rewritten_query, agent_name)
        for agent_name in agent_names
    ]
    return await asyncio.gather(*tasks)


async def run_specialist_assessment(
    rewritten_query: str,
    agent_name: str,
) -> SpecialistAssessment:
    start_time = perf_counter()
    retrieval = await asyncio.to_thread(
        retrieve_knowledge,
        rewritten_query,
        agent_name,
    )
    chain = build_specialist_chain(agent_name)
    message = await chain.ainvoke(
        {
            "medical_data": rewritten_query,
            "knowledge_context": format_knowledge_context(retrieval.final_chunks),
        }
    )
    return SpecialistAssessment(
        agent_name=agent_name,
        agent_label=ROUTER_AGENT_NAMES[agent_name],
        content=extract_message_text(message),
        usage=extract_token_usage(message),
        reasoning_time_seconds=round(perf_counter() - start_time, 3),
        knowledge_chunks=to_schema_knowledge_chunks(retrieval.final_chunks),
        coarse_knowledge_chunks=to_schema_knowledge_chunks(retrieval.coarse_chunks),
        reranked_knowledge_chunks=to_schema_knowledge_chunks(retrieval.reranked_chunks),
    )


async def summarize_specialist_assessments(
    rewritten_query: str,
    specialist_assessments: list[SpecialistAssessment],
) -> tuple[str, TokenUsage]:
    summary_chain = build_specialist_summary_chain()
    message = await summary_chain.ainvoke(
        build_specialist_summary_input(rewritten_query, specialist_assessments)
    )
    return extract_message_text(message), extract_token_usage(message)


def build_specialist_summary_input(
    rewritten_query: str,
    specialist_assessments: list[SpecialistAssessment],
) -> dict[str, str]:
    combined_chunks = combine_knowledge_chunks(
        specialist_assessments,
        chunk_attr="knowledge_chunks",
    )
    return {
        "medical_data": rewritten_query,
        "specialist_assessments": format_specialist_assessments(specialist_assessments),
        "knowledge_context": format_knowledge_context(from_schema_knowledge_chunks(combined_chunks)),
    }


def format_specialist_assessments(
    specialist_assessments: list[SpecialistAssessment],
) -> str:
    sections: list[str] = []
    for index, assessment in enumerate(specialist_assessments, start=1):
        sections.append(
            "\n".join(
                [
                    f"[专科{index}] agent_name: {assessment.agent_name}",
                    f"[专科{index}] agent_label: {assessment.agent_label}",
                    f"[专科{index}] 分析结果:",
                    assessment.content,
                ]
            )
        )
    return "\n\n".join(sections) if sections else "无专科分析结果。"


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


def add_token_usages(*usages: TokenUsage) -> TokenUsage:
    total = TokenUsage()
    for usage in usages:
        total = add_token_usage(total, usage)
    return total


def parse_router_decision(raw_text: str) -> RouterDecision:
    normalized = raw_text.strip()
    try:
        payload = json.loads(extract_json_object(normalized))
    except json.JSONDecodeError:
        return fallback_router_decision(normalized)

    reason = str(payload.get("reason", "")).strip()
    agent_names = normalize_agent_names(
        payload.get("agent_names", payload.get("agent_name", []))
    )
    if not agent_names:
        return fallback_router_decision(normalized)
    return RouterDecision(agent_names=agent_names, reason=reason)


def extract_json_object(raw_text: str) -> str:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return raw_text
    return raw_text[start : end + 1]


def normalize_agent_names(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        candidates = [raw_value]
    elif isinstance(raw_value, list):
        candidates = [str(item) for item in raw_value]
    else:
        candidates = []

    normalized: list[str] = []
    for candidate in candidates:
        agent_name = candidate.strip()
        if agent_name in ROUTER_AGENT_NAMES and agent_name not in normalized:
            normalized.append(agent_name)
    return normalized


def fallback_router_decision(raw_text: str) -> RouterDecision:
    lowered = raw_text.lower()
    mentioned_agents = [
        agent_name
        for agent_name in ROUTER_AGENT_NAMES
        if agent_name in lowered
    ]
    if mentioned_agents:
        return RouterDecision(
            agent_names=mentioned_agents,
            reason="fallback parser matched agent name",
        )

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

    matched_agents = [
        agent_name
        for agent_name, keywords in keyword_groups.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    if matched_agents:
        return RouterDecision(
            agent_names=matched_agents,
            reason="fallback keyword router matched input",
        )

    return RouterDecision(
        agent_names=["cardiometabolic_health"],
        reason="fallback defaulted to cardiometabolic_health",
    )


def order_specialist_assessments(
    specialist_assessments: list[SpecialistAssessment],
    agent_names: list[str],
) -> list[SpecialistAssessment]:
    order_map = {agent_name: index for index, agent_name in enumerate(agent_names)}
    return sorted(
        specialist_assessments,
        key=lambda assessment: order_map.get(assessment.agent_name, len(order_map)),
    )


def combine_knowledge_chunks(
    specialist_assessments: list[SpecialistAssessment],
    chunk_attr: str,
) -> list[KnowledgeChunk]:
    merged: list[KnowledgeChunk] = []
    seen: set[tuple[str, str, str]] = set()
    for assessment in specialist_assessments:
        chunks = getattr(assessment, chunk_attr, [])
        for chunk in chunks:
            key = (chunk.source_file, chunk.section_path, chunk.content)
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
    return merged


def from_schema_knowledge_chunks(chunks: list[KnowledgeChunk]) -> list[Any]:
    return [
        type(
            "KnowledgeContextChunk",
            (),
            {
                "source_file": chunk.source_file,
                "section_path": chunk.section_path,
                "content": chunk.content,
            },
        )()
        for chunk in chunks
    ]


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
