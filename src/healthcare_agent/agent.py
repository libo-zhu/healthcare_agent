from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_deepseek import ChatDeepSeek

from healthcare_agent.config import DEFAULT_BASE_URL, DEFAULT_MODEL, get_deepseek_api_key
from healthcare_agent.prompts import build_healthcare_prompt


def build_agent_chain():
    model = ChatDeepSeek(
        model=DEFAULT_MODEL,
        api_key=get_deepseek_api_key(),
        base_url=DEFAULT_BASE_URL,
        temperature=0.2,
    )
    prompt = build_healthcare_prompt()
    parser = StrOutputParser()
    return prompt | model | parser


def run_healthcare_assessment(medical_data: str) -> str:
    chain = build_agent_chain()
    return chain.invoke({"medical_data": medical_data})

