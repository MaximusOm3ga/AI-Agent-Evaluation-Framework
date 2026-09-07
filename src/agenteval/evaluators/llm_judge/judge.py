from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from ...core.evaluator import Evaluator
from ...core.models import DatasetCase, Run, Trace
from ...core.results import EvaluationResult


@dataclass(slots=True)
class JudgeConfig:
    model: str = "llama-3.1-8b-instant"
    provider: str = "mock"
    system_prompt: str = "You are a careful evaluator. Assess the answer against the system instructions and provide structured JSON."
    evaluation_prompt: str = "Assess the agent output for correctness and completeness."
    score_scale: str = "0 = incorrect, 1 = weak, 2 = fair, 3 = good, 4 = excellent"
    criteria: list[str] = field(default_factory=lambda: ["correctness", "completeness", "relevance"])
    temperature: float = 0.0
    structured_output_schema: dict[str, Any] = field(
        default_factory=lambda: {"score": "int", "reason": "string", "passed": "bool"}
    )
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        if self.provider.lower() == "groq":
            override = os.getenv("GROQ_MODEL")
            if override:
                self.model = override


class LLMJudge(Evaluator):
    name = "LLMJudge"

    def __init__(self, config: JudgeConfig | None = None, judge_fn: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None):
        self.config = config or JudgeConfig()
        self.judge_fn = judge_fn

    def _prompt_for_case(self, case: DatasetCase, run: Run, trace: Trace) -> str:
        task = case.input if isinstance(case.input, (dict, list, str)) else str(case.input)
        answer = run.output if run.output is not None else "<no final answer>"
        trace_summary = "\n".join(
            f"[{event.type}] {event.name or 'event'}: {event.output if event.output is not None else event.input}"
            for event in trace.ordered_events()[-10:]
        )
        expected = case.expected or {}
        return (
            f"{self.config.evaluation_prompt}\n\n"
            f"Task:\n{task}\n\n"
            f"Expected:\n{json.dumps(expected, ensure_ascii=True)}\n\n"
            f"Agent final output:\n{answer}\n\n"
            f"Relevant trace:\n{trace_summary}\n\n"
            f"Criteria:\n{', '.join(self.config.criteria)}\n\n"
            f"Score scale:\n{self.config.score_scale}"
        )

    def _mock_judge(self, prompt: str) -> dict[str, Any]:
        lower = prompt.lower()
        if "42" in lower or "correct" in lower:
            score = 4
            passed = True
            reason = "The answer is correct and aligns with the expected output."
        elif "no final answer" in lower:
            score = 0
            passed = False
            reason = "The agent did not provide a usable final answer."
        else:
            score = 2
            passed = False
            reason = "The answer is only partially aligned with the task requirements."
        return {"score": score, "reason": reason, "passed": passed}

    def _resolve_api_credentials(self) -> tuple[str | None, str | None]:
        provider = self.config.provider.lower()
        if provider == "groq":
            api_key = self.config.api_key or os.getenv("GROQ_API_KEY")
            base_url = self.config.base_url or os.getenv("GROQ_BASE_URL") or "https://api.groq.com/openai/v1"
            return api_key, base_url
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.config.base_url or os.getenv("OPENAI_BASE_URL")
        return api_key, base_url

    def _call_openai(self, prompt: str) -> dict[str, Any]:
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is not installed") from exc

        api_key, base_url = self._resolve_api_credentials()
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        response = client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Judge returned empty content")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Judge output was not a JSON object")
        return parsed

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        prompt = self._prompt_for_case(case, run, trace)
        try:
            if self.judge_fn is not None:
                payload = self.judge_fn(prompt, self.config.model, {"criteria": self.config.criteria})
            elif self.config.provider.lower() in {"openai", "groq"}:
                payload = self._call_openai(prompt)
            else:
                payload = self._mock_judge(prompt)
        except Exception as exc:  # noqa: BLE001
            return EvaluationResult(
                evaluator=self.name,
                score=None,
                passed=False,
                explanation=f"LLM judge failed: {exc}",
                metadata={"judge_model": self.config.model, "judge_provider": self.config.provider},
                status="error",
                error=str(exc),
            )

        score = payload.get("score")
        reason = payload.get("reason")
        passed = payload.get("passed")
        if score is None:
            try:
                score = float(payload.get("rating", 0))
            except (TypeError, ValueError):
                score = 0
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        normalized = max(0.0, min(1.0, score / 4.0)) if self.config.score_scale.lower().startswith("0 = incorrect") else score
        return EvaluationResult(
            evaluator=self.name,
            score=normalized,
            passed=bool(passed) if passed is not None else normalized >= 0.5,
            explanation=str(reason or "LLM judge score generated."),
            metadata={
                "judge_model": self.config.model,
                "judge_provider": self.config.provider,
                "criteria": self.config.criteria,
                "raw_output": payload,
            },
            evidence=[event.id for event in trace.ordered_events()[-5:]],
        )
