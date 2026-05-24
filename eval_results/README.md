# Final Evaluation Artifacts

本目录的最终保留口径以 `../../thesis/本科毕设论文-朱力波.docx` 第 4 章为准。

## 定稿采用的结果

| 路径 | 用途 |
| --- | --- |
| `20260428_185414_specialist_direct/` | Healthcare Agent 在 80 条测试样本上的最终运行结果 |
| `20260428_190657_specialist_deepseek/` | DeepSeek V4 直接调用基线运行结果 |
| `gpt55_high_kb_aware_judge/` | GPT-5.5 high KB-aware 成对评审结果 |
| `EXPERIMENT_ANALYSIS.md` | 与论文第 4 章一致的实验分析汇总 |

## Word 定稿中的关键数字

| 项目 | 结果 |
| --- | ---: |
| 测试样本数 | 80 |
| 路由准确率 | 0.8729 |
| 路由召回率 | 0.9875 |
| 路由 F1 值 | 0.9058 |
| RAG 检索准确率 | 0.4860 |
| RAG 检索召回率 | 0.8152 |
| RAG 检索 F1 值 | 0.5805 |
| 高风险样本数 | 30 |
| 紧急提示覆盖率 | 100% |
| 延误提示率 | 0% |
| Agent 更优 | 35 |
| DeepSeek 更优 | 3 |
| 二者相当 | 42 |
| Agent 不劣于基线比例 | 96.25% |

四维评分：

| 评价维度 | Healthcare Agent | DeepSeek V4 | 差值 |
| --- | ---: | ---: | ---: |
| 正确性 | 4.3000 | 4.4875 | -0.1875 |
| 覆盖度 | 4.9500 | 4.5375 | +0.4125 |
| 依据充分性 | 4.6875 | 4.0125 | +0.6750 |
| 安全性 | 4.8750 | 4.8500 | +0.0250 |

Token 资源消耗：

| 指标 | Healthcare Agent | DeepSeek V4 |
| --- | ---: | ---: |
| 平均输入 Token | 6706.1000 | 123.1646 |
| 平均输出 Token | 2775.2875 | 897.2025 |
| 平均总 Token | 9481.3875 | 1020.3671 |

## 可清理的旧结果

以下内容不属于 Word 定稿采用结果，可在最终整理仓库时删除：

```text
20260423_221848_specialist_direct/
gpt55_high_judge/
pairwise_comparison.csv
pairwise_comparison.jsonl
pairwise_summary.json
.DS_Store
```

`figures/` 中的图不是 Word 定稿直接嵌入的最终图片。Word 第 4 章实际使用的图片对应外层论文素材目录：

```text
../../thesis/assets/chapter4_redrawn/fig4-1-rag-metrics.png
../../thesis/assets/chapter4_redrawn/fig4-2-radar-scores.png
../../thesis/assets/chapter4_redrawn/fig4-3-grouped-pairwise.png
```
