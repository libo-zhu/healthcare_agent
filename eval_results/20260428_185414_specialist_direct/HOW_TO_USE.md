# Evaluation Output Guide

本次评测已经自动生成了这些文件：

- `summary.json`: 自动聚合后的总体指标
- `per_case_results.jsonl`: 每条样本的详细结果，适合进一步脚本分析
- `manual_review.csv`: 人工复核或 LLM-as-a-judge 的模板
- `run_config.json`: 本次运行配置

## 已自动计算的指标

- 路由指标：`macro_precision`、`macro_recall`、`macro_f1`、`exact_match_rate`
- RAG 文件级指标：`macro_precision`、`macro_recall`、`macro_f1`、`exact_match_rate`
- 性能指标：平均耗时、平均 reasoning time、平均 token、总 token
- 安全提示启发式指标：
  - `urgency_signal_rate_in_high_risk_cases`
  - `delay_signal_rate_in_high_risk_cases`

## 建议你手动或半自动继续做的实验

### 1. 最终回答质量

最推荐的方法是使用 `manual_review.csv` 做人工评估或 LLM-as-a-judge。

建议评分维度：

- 正确性：1-5
- 覆盖度：1-5
- 依据充分性：1-5
- 安全性：1-5
- 是否有幻觉：yes/no

你可以让另一个 LLM 读取以下字段后打分：

- `input_text`
- `reference_keypoints`
- `safety_expected`
- `model_output`

### 2. LLM-as-a-judge 的推荐提示词

可以让 judge 模型完成下面任务：

1. 判断输出是否覆盖 `reference_keypoints` 的核心点
2. 判断是否出现明显知识外扩展或幻觉
3. 对高风险样本判断是否给出了足够明确的就医/急诊建议
4. 输出结构化 JSON 评分，字段与你的 `manual_review.csv` 保持一致

### 3. 更细粒度的 RAG 实验

当前脚本做的是“文件级命中评估”。
如果你后面想做 chunk 级 Recall@k / Precision@k，可以：

1. 给测试集里的 `gold_relevant_chunks` 改成更细的 chunk 标识
2. 在 agent 返回的 `knowledge_chunks` 中保留 `source_file + section_path`
3. 用同样的集合比较函数计算 chunk 级指标

### 4. 消融实验建议

你可以固定同一份 `test_data.json`，分别运行：

- Base：单模型直接问答
- +Router
- +Router+Rewrite
- +Router+Rewrite+RAG（当前完整系统）

然后把每轮的 `summary.json` 拿出来汇总成论文表格。

### 5. Agent vs DeepSeek baseline 对比

直接调用 DeepSeek API，不经过你构建的 agent/router/RAG：

```bash
python evaluate_test_data.py --runner deepseek --mode specialist --concurrency 2
```

拿 agent 结果和 DeepSeek baseline 结果做成对 LLM-as-a-judge：

```bash
python judge_pairwise_comparison.py   --agent-results-jsonl path/to/agent/per_case_results.jsonl   --deepseek-results-jsonl path/to/deepseek/per_case_results.jsonl   --output-csv path/to/pairwise_comparison.csv   --output-jsonl path/to/pairwise_comparison.jsonl   --summary-json path/to/pairwise_summary.json   --overwrite
```

## 本次运行摘要

```json
{
  "dataset_path": "/Users/zhulibo/data/university/graduation project/healthcare_agent/test_data.json",
  "mode": "specialist",
  "runner": "direct",
  "total_cases": 80,
  "ok_cases": 80,
  "error_cases": 0,
  "success_rate": 1.0,
  "timing": {
    "mean_elapsed_seconds": 31.8515,
    "mean_reasoning_time_seconds": 31.8513
  },
  "tokens": {
    "mean_input_tokens": 6706.1,
    "mean_output_tokens": 2775.2875,
    "mean_total_tokens": 9481.3875,
    "sum_total_tokens": 758511
  },
  "routing_metrics": {
    "evaluated_cases": 80,
    "macro_precision": 0.8729,
    "macro_recall": 0.9875,
    "macro_f1": 0.9058,
    "exact_match_rate": 0.725,
    "mean_jaccard": 0.8604
  },
  "rag_file_level_metrics": {
    "evaluated_cases": 80,
    "macro_precision": 0.486,
    "macro_recall": 0.8152,
    "macro_f1": 0.5805,
    "exact_match_rate": 0.1375
  },
  "safety_metrics": {
    "high_risk_cases": 30,
    "urgency_signal_rate_in_high_risk_cases": 1.0,
    "delay_signal_rate_in_high_risk_cases": 0.0
  },
  "error_case_ids": []
}
```
