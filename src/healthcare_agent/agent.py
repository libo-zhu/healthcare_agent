from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

from healthcare_agent.config import DEFAULT_BASE_URL, DEFAULT_MODEL, get_deepseek_api_key
from healthcare_agent.prompts import build_healthcare_prompt


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
    parser = StrOutputParser()
    return prompt | model | parser


def run_healthcare_assessment(medical_data: str) -> str:
    chain = build_agent_chain()
    return chain.invoke({"medical_data": medical_data})


async def stream_healthcare_assessment(medical_data: str) -> AsyncIterator[str]:
    chain = build_agent_chain()
    async for chunk in chain.astream({"medical_data": medical_data}):
        if chunk:
            yield chunk

