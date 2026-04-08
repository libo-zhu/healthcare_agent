from __future__ import annotations

import argparse

from healthcare_agent.agent import run_healthcare_assessment
from healthcare_agent.knowledge_base import build_knowledge_base_index, get_knowledge_base_status


DEFAULT_MEDICAL_DATA = """
患者，45岁，男性。
近1个月体检数据如下：
- 身高 175cm，体重 82kg
- 血压 148/95 mmHg
- 空腹血糖 6.8 mmol/L
- 总胆固醇 6.3 mmol/L
- 低密度脂蛋白胆固醇 4.2 mmol/L
- 甘油三酯 2.1 mmol/L
- 尿酸 468 umol/L
- ALT 62 U/L

主诉：
- 最近经常熬夜，工作压力较大
- 偶尔头晕
- 饭后容易犯困
- 平时缺乏运动
- 每周饮酒 2 到 3 次
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcare agent CLI")
    parser.add_argument(
        "--rebuild-kb",
        action="store_true",
        help="Build or rebuild the local medical knowledge-base index from JSON files.",
    )
    parser.add_argument(
        "--kb-status",
        action="store_true",
        help="Show current knowledge-base status.",
    )
    args = parser.parse_args()

    if args.rebuild_kb:
        result = build_knowledge_base_index(force_rebuild=True)
        print("\n===== Knowledge Base Rebuilt =====\n")
        print(result.model_dump_json(indent=2))
        return

    if args.kb_status:
        status = get_knowledge_base_status()
        print("\n===== Knowledge Base Status =====\n")
        print(status.model_dump_json(indent=2))
        return

    print("请输入用户医疗数据，直接回车则使用默认测试样例：")
    user_input = input("> ").strip()
    medical_data = user_input or DEFAULT_MEDICAL_DATA

    result = run_healthcare_assessment(medical_data)

    print("\n===== DeepSeek 返回结果 =====\n")
    print(result)
