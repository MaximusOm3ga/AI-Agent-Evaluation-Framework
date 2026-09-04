from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any


@dataclass(slots=True)
class EvaluationResult:
    evaluator: str
    score: float | None
    passed: bool | None
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    status: str = "ok"
    error: str | None = None


@dataclass(slots=True)
class EvaluationSummary:
    case_count: int
    results: list[EvaluationResult]

    def by_evaluator(self) -> dict[str, list[EvaluationResult]]:
        grouped: dict[str, list[EvaluationResult]] = {}
        for result in self.results:
            grouped.setdefault(result.evaluator, []).append(result)
        return grouped

    def metric(self, evaluator: str) -> float | None:
        scores = [result.score for result in self.by_evaluator().get(evaluator, []) if result.score is not None]
        return mean(scores) if scores else None

    def summary_lines(self) -> list[str]:
        lines = [f"Cases: {self.case_count}"]
        for evaluator, items in sorted(self.by_evaluator().items()):
            scores = [item.score for item in items if item.score is not None]
            if not scores:
                continue
            value = mean(scores)
            if 0 <= value <= 1:
                lines.append(f"{evaluator:24} {value:.1%}")
            else:
                lines.append(f"{evaluator:24} {value:.3f}")
        return lines

    def summary(self) -> str:
        return "\n".join(self.summary_lines())
