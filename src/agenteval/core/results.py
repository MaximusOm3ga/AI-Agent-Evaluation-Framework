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

    def detailed_lines(self) -> list[str]:
        lines = [f"Cases: {self.case_count}"]
        for evaluator, items in sorted(self.by_evaluator().items()):
            scores = [item.score for item in items if item.score is not None]
            if not scores:
                lines.append(f"{evaluator:24} no numeric score")
                continue
            avg_score = mean(scores)
            pass_count = sum(1 for item in items if item.passed is True)
            fail_count = sum(1 for item in items if item.passed is False)
            lines.append(
                f"{evaluator:24} avg={avg_score:.1%} | pass={pass_count} | fail={fail_count} | "
                f"scores={[round(score, 3) for score in scores]}"
            )
            for item in items:
                status = "PASS" if item.passed is True else "FAIL" if item.passed is False else "N/A"
                score_text = "n/a" if item.score is None else f"{item.score:.3f}"
                lines.append(f"  - {status} | score={score_text} | {item.explanation or 'no explanation'}")
                if item.metadata:
                    lines.append(f"    metadata={item.metadata}")
        return lines

    def summary(self) -> str:
        return "\n".join(self.summary_lines())

    def detailed_summary(self) -> str:
        return "\n".join(self.detailed_lines())
