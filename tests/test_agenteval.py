from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import types

from agenteval import (
    AnswerCorrectness,
    Dataset,
    DatasetCase,
    EvaluationSuite,
    ExactMatch,
    EventType,
    JudgeConfig,
    LLMJudge,
    LoopDetection,
    Run,
    TaskSuccess,
    ToolArgumentCorrectness,
    ToolSelection,
    Tracer,
    evaluate,
    evaluate_async,
    trace,
)
from agenteval.storage.database import SQLiteDatabase
from agenteval.storage.repositories import Repository


def build_dataset() -> Dataset:
    return Dataset(
        id="research",
        name="research",
        cases=[
            DatasetCase(
                id="case-1",
                input={"task": "find answer"},
                expected={
                    "answer": "42",
                    "tools": ["calculator"],
                    "tool_arguments": {"calculator": {"expression": "6 * 7"}},
                },
            )
        ],
    )


def test_sync_evaluate_records_trace_and_metrics():
    tracer = Tracer()

    @trace(tracer=tracer)
    def agent(task, tracer=None):
        with tracer.span(EventType.LLM_CALL, name="plan", input=task) as outer:
            with tracer.span(EventType.TOOL_CALL, name="calculator", input={"expression": "6 * 7"}) as inner:
                inner.finish(output={"value": 42}, status="succeeded")
            outer.finish(output={"strategy": "use calculator"}, status="succeeded")
        return "42"

    suite = EvaluationSuite(
        [
            TaskSuccess(),
            ExactMatch(),
            AnswerCorrectness(),
            ToolSelection(),
            ToolArgumentCorrectness(),
            LoopDetection(),
        ]
    )
    result = evaluate(agent=agent, dataset=build_dataset(), suite=suite, tracer=tracer)
    summary = result.summary()

    assert summary.metric("TaskSuccess") == 1.0
    assert summary.metric("ExactMatch") == 1.0
    assert summary.metric("ToolSelection") == 1.0
    assert summary.metric("LoopDetection") == 1.0

    trace_obj = result.cases[0].trace
    assert [event.type for event in trace_obj.ordered_events()].count(EventType.TOOL_CALL) == 1
    assert trace_obj.filter(EventType.FINAL_OUTPUT)[0].output == "42"


def test_async_evaluate_works():
    async def agent(task, tracer=None):
        with tracer.span(EventType.TOOL_CALL, name="calculator", input={"expression": "6 * 7"}) as event:
            event.finish(output={"value": 42}, status="succeeded")
        return "42"

    suite = EvaluationSuite([TaskSuccess(), ExactMatch(), ToolSelection()])
    result = asyncio.run(evaluate_async(agent=agent, dataset=build_dataset(), suite=suite))
    assert result.summary().metric("TaskSuccess") == 1.0
    assert result.summary().metric("ExactMatch") == 1.0


def test_storage_round_trip(tmp_path: Path):
    tracer = Tracer()

    @trace(tracer=tracer)
    def agent(task, tracer=None):
        return "42"

    suite = EvaluationSuite([TaskSuccess(), ExactMatch()])
    result = evaluate(agent=agent, dataset=build_dataset(), suite=suite, tracer=tracer)

    db_path = tmp_path / "agenteval.sqlite3"
    repository = Repository(SQLiteDatabase(db_path))
    case_result = result.cases[0]
    repository.save_run(case_result.run, case_result.trace)
    repository.save_evaluation_results(case_result.run.id, case_result.results)

    loaded_run = repository.load_run(case_result.run.id)
    loaded_trace = repository.load_trace(case_result.run.id)
    loaded_results = repository.list_evaluation_results(case_result.run.id)

    assert loaded_run.output == "42"
    assert loaded_trace.filter(EventType.FINAL_OUTPUT)[0].output == "42"
    assert {item.evaluator for item in loaded_results} == {"TaskSuccess", "ExactMatch"}


def test_llm_judge_mock_evaluation():
    judge = LLMJudge(config=JudgeConfig(provider="mock"))
    dataset = build_dataset()
    result = evaluate(
        agent=lambda task, tracer=None: "42",
        dataset=dataset,
        suite=EvaluationSuite([judge]),
    )
    score = result.summary().metric("LLMJudge")
    assert score == 1.0
    assert result.cases[0].results[0].passed is True


def test_llm_judge_groq_provider_uses_groq_env(monkeypatch):
    captured: dict[str, object] = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            content = json.dumps({"score": 4, "reason": "Great answer", "passed": True})
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
            )

    class _FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    fake_openai = types.SimpleNamespace(OpenAI=_FakeOpenAI)
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)

    judge = LLMJudge(
        config=JudgeConfig(
            provider="groq",
            model="llama-3.3-70b-versatile",
            system_prompt="judge system",
            evaluation_prompt="judge eval prompt",
        )
    )
    result = evaluate(
        agent=lambda task, tracer=None: "42",
        dataset=build_dataset(),
        suite=EvaluationSuite([judge]),
    )

    assert captured["api_key"] == "groq-test-key"
    assert captured["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["request"]["model"] == "llama-3.3-70b-versatile"
    assert result.summary().metric("LLMJudge") == 1.0
