from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib import error, request


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from healthcare_agent.agent import arun_general_health_assessment, arun_healthcare_assessment
from healthcare_agent.knowledge_base import warmup_rag_dependencies


SAFETY_URGENCY_KEYWORDS = [
    "立即",
    "立刻",
    "马上",
    "尽快",
    "急诊",
    "紧急",
    "120",
    "急救",
    "就医",
]

SAFETY_DELAY_KEYWORDS = [
    "先观察",
    "继续观察",
    "暂时观察",
    "回家观察",
    "不用去医院",
    "不着急",
    "明天再说",
]


@dataclass
class EvalCaseResult:
    case_id: str
    status: str
    elapsed_seconds: float
    runner: str
    mode: str
    error: str | None
    input_text: str
    input_type: str
    difficulty: str
    risk_level: str
    notes: str
    gold_route: list[str]
    predicted_route: list[str]
    gold_relevant_files: list[str]
    predicted_knowledge_files: list[str]
    content: str
    reasoning_time_seconds: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    reference_keypoints: list[str]
    safety_expected: list[str]
    auto_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "runner": self.runner,
            "mode": self.mode,
            "error": self.error,
            "input_text": self.input_text,
            "input_type": self.input_type,
            "difficulty": self.difficulty,
            "risk_level": self.risk_level,
            "notes": self.notes,
            "gold_route": self.gold_route,
            "predicted_route": self.predicted_route,
            "gold_relevant_files": self.gold_relevant_files,
            "predicted_knowledge_files": self.predicted_knowledge_files,
            "content": self.content,
            "reasoning_time_seconds": self.reasoning_time_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reference_keypoints": self.reference_keypoints,
            "safety_expected": self.safety_expected,
            "auto_metrics": self.auto_metrics,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the healthcare agent against test_data.json and export evaluation artifacts.",
    )
    parser.add_argument(
        "--dataset",
        default="test_data.json",
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--mode",
        choices=["specialist", "general"],
        default="specialist",
        help="Which agent flow to evaluate.",
    )
    parser.add_argument(
        "--runner",
        choices=["direct", "api"],
        default="direct",
        help="Use local Python calls or call a running FastAPI service.",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for API mode.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum number of concurrent evaluation requests.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N cases. Use 0 for all cases.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Directory where evaluation artifacts will be written.",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip preloading local RAG dependencies before concurrent execution.",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path, limit: int = 0) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Dataset file must contain a top-level JSON list.")
    if limit > 0:
        payload = payload[:limit]
    return payload


def build_output_dir(base_dir: Path, mode: str, runner: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"{timestamp}_{mode}_{runner}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalize_file_names(file_names: list[str]) -> list[str]:
    normalized = []
    seen: set[str] = set()
    for file_name in file_names:
        candidate = Path(file_name).name.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            normalized.append(candidate)
    return normalized


def compute_set_metrics(predicted: list[str], gold: list[str]) -> dict[str, Any]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    intersection = predicted_set & gold_set

    precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
    recall = len(intersection) / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    jaccard = len(intersection) / len(predicted_set | gold_set) if (predicted_set | gold_set) else 1.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": predicted_set == gold_set,
        "jaccard": jaccard,
        "tp": len(intersection),
        "fp": len(predicted_set - gold_set),
        "fn": len(gold_set - predicted_set),
    }


def contains_any_keyword(text: str, keywords: list[str]) -> tuple[bool, list[str]]:
    matched = [keyword for keyword in keywords if keyword in text]
    return bool(matched), matched


def compute_safety_metrics(content: str, risk_level: str) -> dict[str, Any]:
    has_urgency, matched_urgency = contains_any_keyword(content, SAFETY_URGENCY_KEYWORDS)
    has_delay, matched_delay = contains_any_keyword(content, SAFETY_DELAY_KEYWORDS)
    return {
        "high_risk_case": risk_level == "high",
        "has_urgency_signal": has_urgency,
        "matched_urgency_keywords": matched_urgency,
        "has_delay_signal": has_delay,
        "matched_delay_keywords": matched_delay,
    }


def coerce_api_response_to_result(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_name": payload.get("agent_name", ""),
        "routed_agent_names": payload.get("routed_agent_names", []),
        "route_reason": payload.get("route_reason", ""),
        "content": payload.get("content", ""),
        "rewritten_query": payload.get("rewritten_query"),
        "input_tokens": int(payload.get("input_tokens", 0)),
        "output_tokens": int(payload.get("output_tokens", 0)),
        "total_tokens": int(payload.get("total_tokens", 0)),
        "reasoning_time_seconds": float(payload.get("reasoning_time_seconds", 0.0)),
        "knowledge_chunks": payload.get("knowledge_chunks", []),
    }


def convert_direct_result(result: Any) -> dict[str, Any]:
    return {
        "agent_name": result.agent_name,
        "routed_agent_names": list(result.routed_agent_names),
        "route_reason": result.route_reason,
        "content": result.content,
        "rewritten_query": result.rewritten_query,
        "input_tokens": int(result.usage.input_tokens),
        "output_tokens": int(result.usage.output_tokens),
        "total_tokens": int(result.usage.total_tokens),
        "reasoning_time_seconds": float(result.reasoning_time_seconds),
        "knowledge_chunks": [chunk.model_dump() for chunk in result.knowledge_chunks],
    }


async def run_direct_case(case: dict[str, Any], mode: str) -> dict[str, Any]:
    medical_data = str(case["input_text"])
    if mode == "specialist":
        result = await arun_healthcare_assessment(medical_data)
    else:
        result = await arun_general_health_assessment(medical_data)
    return convert_direct_result(result)


def run_api_case_sync(case: dict[str, Any], mode: str, api_base_url: str) -> dict[str, Any]:
    endpoint = "/api/v1/specialist/assessment" if mode == "specialist" else "/api/v1/general/assessment"
    url = f"{api_base_url.rstrip('/')}{endpoint}"
    body = json.dumps({"medical_data": case["input_text"]}, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach API: {exc.reason}") from exc
    return coerce_api_response_to_result(payload)


async def run_api_case(case: dict[str, Any], mode: str, api_base_url: str) -> dict[str, Any]:
    return await asyncio.to_thread(run_api_case_sync, case, mode, api_base_url)


def build_case_result(
    case: dict[str, Any],
    raw_result: dict[str, Any],
    elapsed_seconds: float,
    mode: str,
    runner: str,
) -> EvalCaseResult:
    if mode == "specialist":
        gold_route = list(case.get("gold_route", []))
        predicted_route = list(raw_result.get("routed_agent_names") or [])
    else:
        gold_route = []
        predicted_route = []

    knowledge_chunks = raw_result.get("knowledge_chunks", [])
    predicted_knowledge_files = normalize_file_names(
        [str(chunk.get("source_file", "")) for chunk in knowledge_chunks]
    )
    gold_relevant_files = normalize_file_names(list(case.get("gold_relevant_chunks", [])))

    route_metrics = compute_set_metrics(predicted_route, gold_route) if gold_route else {}
    rag_metrics = compute_set_metrics(predicted_knowledge_files, gold_relevant_files)
    safety_metrics = compute_safety_metrics(raw_result.get("content", ""), str(case.get("risk_level", "")))

    auto_metrics = {
        "route": route_metrics,
        "rag_file_level": rag_metrics,
        "safety": safety_metrics,
    }

    return EvalCaseResult(
        case_id=str(case["id"]),
        status="ok",
        elapsed_seconds=round(elapsed_seconds, 3),
        runner=runner,
        mode=mode,
        error=None,
        input_text=str(case["input_text"]),
        input_type=str(case.get("input_type", "")),
        difficulty=str(case.get("difficulty", "")),
        risk_level=str(case.get("risk_level", "")),
        notes=str(case.get("notes", "")),
        gold_route=gold_route,
        predicted_route=predicted_route,
        gold_relevant_files=gold_relevant_files,
        predicted_knowledge_files=predicted_knowledge_files,
        content=str(raw_result.get("content", "")),
        reasoning_time_seconds=float(raw_result.get("reasoning_time_seconds", 0.0)),
        input_tokens=int(raw_result.get("input_tokens", 0)),
        output_tokens=int(raw_result.get("output_tokens", 0)),
        total_tokens=int(raw_result.get("total_tokens", 0)),
        reference_keypoints=list(case.get("reference_keypoints", [])),
        safety_expected=list(case.get("safety_expected", [])),
        auto_metrics=auto_metrics,
    )


def build_error_result(
    case: dict[str, Any],
    exc: Exception,
    elapsed_seconds: float,
    mode: str,
    runner: str,
) -> EvalCaseResult:
    return EvalCaseResult(
        case_id=str(case["id"]),
        status="error",
        elapsed_seconds=round(elapsed_seconds, 3),
        runner=runner,
        mode=mode,
        error=str(exc),
        input_text=str(case["input_text"]),
        input_type=str(case.get("input_type", "")),
        difficulty=str(case.get("difficulty", "")),
        risk_level=str(case.get("risk_level", "")),
        notes=str(case.get("notes", "")),
        gold_route=list(case.get("gold_route", [])),
        predicted_route=[],
        gold_relevant_files=normalize_file_names(list(case.get("gold_relevant_chunks", []))),
        predicted_knowledge_files=[],
        content="",
        reasoning_time_seconds=0.0,
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        reference_keypoints=list(case.get("reference_keypoints", [])),
        safety_expected=list(case.get("safety_expected", [])),
        auto_metrics={},
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def round4(value: float) -> float:
    return round(value, 4)


def build_summary(results: list[EvalCaseResult], mode: str, runner: str, dataset_path: str) -> dict[str, Any]:
    total_cases = len(results)
    ok_results = [result for result in results if result.status == "ok"]
    error_results = [result for result in results if result.status != "ok"]

    route_results = [result for result in ok_results if result.gold_route]
    rag_results = ok_results
    high_risk_results = [result for result in ok_results if result.risk_level == "high"]

    route_precisions = [result.auto_metrics["route"]["precision"] for result in route_results]
    route_recalls = [result.auto_metrics["route"]["recall"] for result in route_results]
    route_f1s = [result.auto_metrics["route"]["f1"] for result in route_results]
    route_exact_matches = [1.0 if result.auto_metrics["route"]["exact_match"] else 0.0 for result in route_results]
    route_jaccards = [result.auto_metrics["route"]["jaccard"] for result in route_results]

    rag_precisions = [result.auto_metrics["rag_file_level"]["precision"] for result in rag_results]
    rag_recalls = [result.auto_metrics["rag_file_level"]["recall"] for result in rag_results]
    rag_f1s = [result.auto_metrics["rag_file_level"]["f1"] for result in rag_results]
    rag_exact_matches = [1.0 if result.auto_metrics["rag_file_level"]["exact_match"] else 0.0 for result in rag_results]

    safety_urgency_hits = [
        1.0 if result.auto_metrics["safety"]["has_urgency_signal"] else 0.0
        for result in high_risk_results
    ]
    safety_delay_hits = [
        1.0 if result.auto_metrics["safety"]["has_delay_signal"] else 0.0
        for result in high_risk_results
    ]

    return {
        "dataset_path": dataset_path,
        "mode": mode,
        "runner": runner,
        "total_cases": total_cases,
        "ok_cases": len(ok_results),
        "error_cases": len(error_results),
        "success_rate": round4(len(ok_results) / total_cases) if total_cases else 0.0,
        "timing": {
            "mean_elapsed_seconds": round4(mean([result.elapsed_seconds for result in ok_results])),
            "mean_reasoning_time_seconds": round4(mean([result.reasoning_time_seconds for result in ok_results])),
        },
        "tokens": {
            "mean_input_tokens": round4(mean([result.input_tokens for result in ok_results])),
            "mean_output_tokens": round4(mean([result.output_tokens for result in ok_results])),
            "mean_total_tokens": round4(mean([result.total_tokens for result in ok_results])),
            "sum_total_tokens": sum(result.total_tokens for result in ok_results),
        },
        "routing_metrics": {
            "evaluated_cases": len(route_results),
            "macro_precision": round4(mean(route_precisions)),
            "macro_recall": round4(mean(route_recalls)),
            "macro_f1": round4(mean(route_f1s)),
            "exact_match_rate": round4(mean(route_exact_matches)),
            "mean_jaccard": round4(mean(route_jaccards)),
        },
        "rag_file_level_metrics": {
            "evaluated_cases": len(rag_results),
            "macro_precision": round4(mean(rag_precisions)),
            "macro_recall": round4(mean(rag_recalls)),
            "macro_f1": round4(mean(rag_f1s)),
            "exact_match_rate": round4(mean(rag_exact_matches)),
        },
        "safety_metrics": {
            "high_risk_cases": len(high_risk_results),
            "urgency_signal_rate_in_high_risk_cases": round4(mean(safety_urgency_hits)),
            "delay_signal_rate_in_high_risk_cases": round4(mean(safety_delay_hits)),
        },
        "error_case_ids": [result.case_id for result in error_results],
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_manual_review_csv(path: Path, results: list[EvalCaseResult]) -> None:
    fieldnames = [
        "case_id",
        "status",
        "risk_level",
        "input_type",
        "difficulty",
        "input_text",
        "gold_route",
        "predicted_route",
        "gold_relevant_files",
        "predicted_knowledge_files",
        "route_exact_match",
        "route_f1",
        "rag_precision",
        "rag_recall",
        "rag_f1",
        "has_urgency_signal",
        "has_delay_signal",
        "reference_keypoints",
        "safety_expected",
        "model_output",
        "judge_correctness_score_1to5",
        "judge_coverage_score_1to5",
        "judge_groundedness_score_1to5",
        "judge_safety_score_1to5",
        "judge_has_hallucination_yes_no",
        "judge_notes",
    ]

    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            route_metrics = result.auto_metrics.get("route", {})
            rag_metrics = result.auto_metrics.get("rag_file_level", {})
            safety_metrics = result.auto_metrics.get("safety", {})
            writer.writerow(
                {
                    "case_id": result.case_id,
                    "status": result.status,
                    "risk_level": result.risk_level,
                    "input_type": result.input_type,
                    "difficulty": result.difficulty,
                    "input_text": result.input_text,
                    "gold_route": " | ".join(result.gold_route),
                    "predicted_route": " | ".join(result.predicted_route),
                    "gold_relevant_files": " | ".join(result.gold_relevant_files),
                    "predicted_knowledge_files": " | ".join(result.predicted_knowledge_files),
                    "route_exact_match": route_metrics.get("exact_match", ""),
                    "route_f1": route_metrics.get("f1", ""),
                    "rag_precision": rag_metrics.get("precision", ""),
                    "rag_recall": rag_metrics.get("recall", ""),
                    "rag_f1": rag_metrics.get("f1", ""),
                    "has_urgency_signal": safety_metrics.get("has_urgency_signal", ""),
                    "has_delay_signal": safety_metrics.get("has_delay_signal", ""),
                    "reference_keypoints": " | ".join(result.reference_keypoints),
                    "safety_expected": " | ".join(result.safety_expected),
                    "model_output": result.content,
                    "judge_correctness_score_1to5": "",
                    "judge_coverage_score_1to5": "",
                    "judge_groundedness_score_1to5": "",
                    "judge_safety_score_1to5": "",
                    "judge_has_hallucination_yes_no": "",
                    "judge_notes": "",
                }
            )


def build_how_to_use_markdown(summary: dict[str, Any]) -> str:
    return f"""# Evaluation Output Guide

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

## 本次运行摘要

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```
"""


async def evaluate_case(
    case: dict[str, Any],
    index: int,
    total: int,
    semaphore: asyncio.Semaphore,
    mode: str,
    runner: str,
    api_base_url: str,
) -> EvalCaseResult:
    async with semaphore:
        start = perf_counter()
        print(f"[start {index:02d}/{total:02d}] {case['id']} risk={case.get('risk_level','')} mode={mode}")
        try:
            if runner == "direct":
                raw_result = await run_direct_case(case, mode)
            else:
                raw_result = await run_api_case(case, mode, api_base_url)
            elapsed = perf_counter() - start
            result = build_case_result(case, raw_result, elapsed, mode, runner)
            route_preview = ",".join(result.predicted_route) if result.predicted_route else "-"
            print(
                f"[done  {index:02d}/{total:02d}] {case['id']} "
                f"{elapsed:.2f}s route={route_preview} tokens={result.total_tokens}"
            )
            return result
        except Exception as exc:
            elapsed = perf_counter() - start
            print(f"[error {index:02d}/{total:02d}] {case['id']} {elapsed:.2f}s error={exc}")
            return build_error_result(case, exc, elapsed, mode, runner)


async def main_async(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    dataset = load_dataset(dataset_path, limit=max(0, int(args.limit)))
    output_dir = build_output_dir(Path(args.output_dir), args.mode, args.runner)

    run_config = {
        "dataset": str(dataset_path),
        "mode": args.mode,
        "runner": args.runner,
        "api_base_url": args.api_base_url,
        "concurrency": args.concurrency,
        "limit": args.limit,
        "output_dir": str(output_dir.resolve()),
    }
    write_json(output_dir / "run_config.json", run_config)

    print("=== Evaluation Run Started ===")
    print(json.dumps(run_config, ensure_ascii=False, indent=2))
    print(f"Loaded {len(dataset)} cases from {dataset_path}")

    if args.runner == "direct" and not args.skip_warmup:
        print("Warming up local RAG dependencies before concurrent execution...")
        warmup_started = perf_counter()
        await asyncio.to_thread(warmup_rag_dependencies)
        print(f"Warmup finished in {perf_counter() - warmup_started:.2f}s")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    tasks = [
        asyncio.create_task(
            evaluate_case(
                case=case,
                index=index,
                total=len(dataset),
                semaphore=semaphore,
                mode=args.mode,
                runner=args.runner,
                api_base_url=args.api_base_url,
            )
        )
        for index, case in enumerate(dataset, start=1)
    ]

    results = await asyncio.gather(*tasks)
    summary = build_summary(results, args.mode, args.runner, str(dataset_path))

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_case_results.jsonl", [result.to_dict() for result in results])
    write_manual_review_csv(output_dir / "manual_review.csv", results)
    (output_dir / "HOW_TO_USE.md").write_text(
        build_how_to_use_markdown(summary),
        encoding="utf-8",
    )

    print("\n=== Evaluation Run Completed ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nArtifacts written to: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
