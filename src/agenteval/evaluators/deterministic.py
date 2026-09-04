from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from ..core.evaluator import Evaluator
from ..core.models import DatasetCase, EventType, Run, Trace
from ..core.results import EvaluationResult


def _normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _selected_tools(trace: Trace) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for event in trace.filter(EventType.TOOL_CALL):
        tools.append(
            {
                "name": event.name or event.metadata.get("tool_name") or event.metadata.get("name"),
                "arguments": event.input,
                "event_id": event.id,
            }
        )
    return tools


class TaskSuccess(Evaluator):
    name = "TaskSuccess"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        expected = (case.expected or {}).get("answer")
        if expected is None:
            passed = run.status == "succeeded" and run.output not in (None, "")
            return EvaluationResult(
                evaluator=self.name,
                score=1.0 if passed else 0.0,
                passed=passed,
                explanation="Agent produced a final output." if passed else "Agent did not produce a final output.",
                evidence=[event.id for event in trace.ordered_events()[-1:]],
            )
        passed = _normalize_text(run.output) == _normalize_text(expected)
        return EvaluationResult(
            evaluator=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation="Final answer matched the expected answer." if passed else "Final answer did not match the expected answer.",
            evidence=[event.id for event in trace.filter(EventType.FINAL_OUTPUT)[:1]],
        )


class ExactMatch(Evaluator):
    name = "ExactMatch"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        expected = (case.expected or {}).get("output", (case.expected or {}).get("answer"))
        if expected is None:
            return EvaluationResult(
                evaluator=self.name,
                score=None,
                passed=None,
                explanation="No expected output was provided.",
            )
        passed = run.output == expected
        return EvaluationResult(
            evaluator=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation="Output matched exactly." if passed else "Output did not match exactly.",
            evidence=[event.id for event in trace.filter(EventType.FINAL_OUTPUT)[:1]],
        )


class AnswerCorrectness(Evaluator):
    name = "AnswerCorrectness"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        expected = case.expected or {}
        answer = expected.get("answer")
        facts = expected.get("facts") or []
        score = 0.0
        evidence: list[str] = []
        explanation = "No expected answer or facts were provided."
        if answer is not None:
            passed = _normalize_text(run.output) == _normalize_text(answer)
            score = 1.0 if passed else 0.0
            explanation = "Answer matched the reference." if passed else "Answer did not match the reference."
        elif facts:
            output_text = _normalize_text(run.output)
            hits = sum(1 for fact in facts if _normalize_text(fact).lower() in output_text.lower())
            score = hits / len(facts)
            explanation = f"{hits}/{len(facts)} reference facts appeared in the output."
            evidence = [event.id for event in trace.filter(EventType.FINAL_OUTPUT)[:1]]
        return EvaluationResult(
            evaluator=self.name,
            score=score,
            passed=score == 1.0 if answer is not None or facts else None,
            explanation=explanation,
            evidence=evidence or [event.id for event in trace.filter(EventType.FINAL_OUTPUT)[:1]],
        )


class ToolSelection(Evaluator):
    name = "ToolSelection"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        expected = (case.expected or {}).get("tools")
        selected = [tool["name"] for tool in _selected_tools(trace) if tool["name"]]
        if not expected:
            return EvaluationResult(
                evaluator=self.name,
                score=1.0 if not selected else 0.5,
                passed=None,
                explanation="No expected tools were provided.",
                evidence=[tool["event_id"] for tool in _selected_tools(trace)],
                metadata={"selected_tools": selected},
            )
        expected_set = set(expected)
        selected_set = set(selected)
        score = len(expected_set & selected_set) / len(expected_set) if expected_set else 1.0
        return EvaluationResult(
            evaluator=self.name,
            score=score,
            passed=score == 1.0,
            explanation="Expected tools were selected." if score == 1.0 else "Selected tools did not fully match the expectation.",
            evidence=[tool["event_id"] for tool in _selected_tools(trace)],
            metadata={"selected_tools": selected, "expected_tools": list(expected_set)},
        )


class ToolArgumentCorrectness(Evaluator):
    name = "ToolArgumentCorrectness"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        expectations = (case.expected or {}).get("tool_arguments") or {}
        calls = _selected_tools(trace)
        if not calls or not expectations:
            return EvaluationResult(
                evaluator=self.name,
                score=None,
                passed=None,
                explanation="No tool argument expectations or tool calls were available.",
                evidence=[tool["event_id"] for tool in calls],
            )
        matches = 0
        for call in calls:
            expected_args = expectations.get(call["name"])
            if expected_args is None:
                continue
            if call["arguments"] == expected_args:
                matches += 1
        score = matches / max(1, len(expectations))
        return EvaluationResult(
            evaluator=self.name,
            score=score,
            passed=score == 1.0,
            explanation="Tool arguments matched expectations." if score == 1.0 else "Some tool arguments differed from expectations.",
            evidence=[tool["event_id"] for tool in calls],
        )


class ToolErrorHandling(Evaluator):
    name = "ToolErrorHandling"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        errors = trace.filter(EventType.TOOL_ERROR)
        if not errors:
            return EvaluationResult(
                evaluator=self.name,
                score=1.0,
                passed=True,
                explanation="No tool errors occurred.",
            )
        score = 0.0
        evidence = [event.id for event in errors]
        for error in errors:
            following = [event for event in trace.ordered_events() if event.timestamp_start > error.timestamp_start]
            if any(event.type == EventType.TOOL_CALL and event.name != error.name for event in following):
                score = 1.0
                break
            if any(event.type == EventType.FINAL_OUTPUT for event in following):
                score = 1.0
                break
        return EvaluationResult(
            evaluator=self.name,
            score=score,
            passed=score == 1.0,
            explanation="The agent recovered after a tool error." if score == 1.0 else "The agent did not recover from tool errors.",
            evidence=evidence,
        )


class Latency(Evaluator):
    name = "Latency"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        return EvaluationResult(
            evaluator=self.name,
            score=run.duration or 0.0,
            passed=None,
            explanation="Elapsed wall-clock time in seconds.",
            metadata={"seconds": run.duration or 0.0},
        )


class TokenUsage(Evaluator):
    name = "TokenUsage"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        total = 0
        for value in run.token_usage.values():
            if isinstance(value, (int, float)):
                total += int(value)
        return EvaluationResult(
            evaluator=self.name,
            score=float(total),
            passed=None,
            explanation="Total token usage recorded on the run.",
            metadata={"tokens": total},
        )


class Cost(Evaluator):
    name = "Cost"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        return EvaluationResult(
            evaluator=self.name,
            score=float(run.estimated_cost or 0.0),
            passed=None,
            explanation="Estimated run cost.",
            metadata={"cost": run.estimated_cost or 0.0},
        )


class LoopDetection(Evaluator):
    name = "LoopDetection"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        calls = _selected_tools(trace)
        if len(calls) < 2:
            return EvaluationResult(
                evaluator=self.name,
                score=1.0,
                passed=True,
                explanation="Not enough tool calls to form a loop.",
            )
        pairs = [(call["name"], repr(call["arguments"])) for call in calls]
        repeated = any(a == b for a, b in zip(pairs, pairs[1:])) or len(pairs) != len(set(pairs))
        return EvaluationResult(
            evaluator=self.name,
            score=0.0 if repeated else 1.0,
            passed=not repeated,
            explanation="Repeated tool calls detected." if repeated else "No tool-call loops detected.",
            evidence=[tool["event_id"] for tool in calls],
            metadata={"tool_calls": pairs},
        )
