from __future__ import annotations

"""Module 4: RAGAS evaluation and diagnostic failure analysis."""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH

METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load the evaluation questions and ground-truth answers from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Evaluate answers with RAGAS' four standard metrics."""
    zero_results = _build_zero_results(
        questions,
        answers,
        contexts,
        ground_truths,
    )
    empty_result = {metric: 0.0 for metric in METRICS}
    empty_result["per_question"] = zero_results

    if not questions:
        return empty_result

    try:
        if not (
            len(questions)
            == len(answers)
            == len(contexts)
            == len(ground_truths)
        ):
            raise ValueError("Evaluation inputs must have equal lengths")

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": list(questions),
                "answer": list(answers),
                "contexts": [list(item) for item in contexts],
                "ground_truth": list(ground_truths),
            }
        )
        evaluation = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
        )
        records = _evaluation_records(evaluation)
        per_question = _records_to_results(
            records,
            questions,
            answers,
            contexts,
            ground_truths,
        )
        return {
            metric: _mean(getattr(result, metric) for result in per_question)
            for metric in METRICS
        } | {"per_question": per_question}
    except Exception as exc:  # noqa: BLE001 - RAGAS has optional runtime dependencies
        print(f"  Warning: RAGAS evaluation failed: {exc}")
        return empty_result


def _evaluation_records(evaluation: Any) -> list[dict]:
    """Convert common RAGAS result representations into row dictionaries."""
    if hasattr(evaluation, "to_pandas"):
        dataframe = evaluation.to_pandas()
        if hasattr(dataframe, "to_dict"):
            return list(dataframe.to_dict(orient="records"))

    if isinstance(evaluation, dict):
        columns = {
            key: value
            for key, value in evaluation.items()
            if key in {"question", "answer", "contexts", "ground_truth", *METRICS}
        }
        row_count = max(
            (len(value) for value in columns.values() if isinstance(value, list)),
            default=1,
        )
        records = []
        for index in range(row_count):
            records.append(
                {
                    key: (
                        value[index]
                        if isinstance(value, list) and index < len(value)
                        else value
                    )
                    for key, value in columns.items()
                }
            )
        return records

    records = []
    for index in range(len(evaluation)):
        row = evaluation[index]
        records.append(dict(row) if isinstance(row, dict) else {})
    return records


def _records_to_results(
    records: list[dict],
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> list[EvalResult]:
    results = []
    for index in range(len(questions)):
        record = records[index] if index < len(records) else {}
        results.append(
            EvalResult(
                question=str(record.get("question", questions[index])),
                answer=str(record.get("answer", answers[index])),
                contexts=_as_contexts(record.get("contexts", contexts[index])),
                ground_truth=str(
                    record.get("ground_truth", ground_truths[index])
                ),
                **{
                    metric: _as_score(record.get(metric, 0.0))
                    for metric in METRICS
                },
            )
        )
    return results


def _build_zero_results(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> list[EvalResult]:
    count = min(len(questions), len(answers), len(contexts), len(ground_truths))
    return [
        EvalResult(
            question=str(questions[index]),
            answer=str(answers[index]),
            contexts=list(contexts[index]),
            ground_truth=str(ground_truths[index]),
            **{metric: 0.0 for metric in METRICS},
        )
        for index in range(count)
    ]


def _as_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if math.isfinite(score) else 0.0


def _as_contexts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def failure_analysis(
    eval_results: list[EvalResult], bottom_n: int = 10
) -> list[dict]:
    """Return the lowest-scoring questions with diagnostic recommendations."""
    if bottom_n <= 0:
        return []

    diagnostic_tree = {
        "faithfulness": (
            "LLM hallucinating",
            "Tighten prompt, lower temperature",
        ),
        "context_recall": (
            "Missing relevant chunks",
            "Improve chunking or add BM25",
        ),
        "context_precision": (
            "Too many irrelevant chunks",
            "Add reranking or metadata filter",
        ),
        "answer_relevancy": (
            "Answer does not match question",
            "Improve prompt template",
        ),
    }

    analyzed = []
    for result in eval_results:
        metric_scores = {
            metric: _as_score(getattr(result, metric, 0.0))
            for metric in METRICS
        }
        average_score = _mean(metric_scores.values())
        worst_metric = min(
            METRICS,
            key=lambda metric: (metric_scores[metric], METRICS.index(metric)),
        )
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append(
            {
                "question": result.question,
                "worst_metric": worst_metric,
                "score": average_score,
                "worst_score": metric_scores[worst_metric],
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )

    analyzed.sort(key=lambda item: (item["score"], item["question"]))
    return analyzed[:bottom_n]


def save_report(
    results: dict,
    failures: list[dict],
    path: str = "reports/ragas_report.json",
):
    """Save aggregate metrics and failure analysis to JSON."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {
            key: value for key, value in results.items() if key != "per_question"
        },
        "num_questions": len(results.get("per_question", [])),
        "per_question": [
            asdict(item) if isinstance(item, EvalResult) else item
            for item in results.get("per_question", [])
        ],
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
