from __future__ import annotations

from healthcare_agent.agent import run_healthcare_assessment


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
    print("请输入用户医疗数据，直接回车则使用默认测试样例：")
    user_input = input("> ").strip()
    medical_data = user_input or DEFAULT_MEDICAL_DATA

    result = run_healthcare_assessment(medical_data)

    print("\n===== DeepSeek 返回结果 =====\n")
    print(result)

