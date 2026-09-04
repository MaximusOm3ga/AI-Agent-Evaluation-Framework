from __future__ import annotations

from dataclasses import dataclass, field
from inspect import isawaitable, signature
from typing import Any, Callable
from uuid import uuid4
import asyncio
import inspect

from .evaluator import Evaluator
from .models import Dataset, DatasetCase, Event, EventType, Run, Trace
from .results import EvaluationResult, EvaluationSummary
from ..tracing.tracer import Tracer, default_tracer


async def _call_agent(agent: Callable[..., Any], case_input: Any, tracer: Tracer) -> Any:
    call_kwargs: dict[str, Any] = {}
    try:
        sig = signature(agent)
        if "tracer" in sig.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
        ):
            call_kwargs["tracer"] = tracer
    except (TypeError, ValueError):
        pass

    result = agent(case_input, **call_kwargs)
    if isawaitable(result):
        return await result
    return result


@dataclass(slots=True)
class EvaluationSuite:
    evaluators: list[Evaluator] = field(default_factory=list)

    async def evaluate_case(self, case: DatasetCase, run: Run, trace: Trace) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for evaluator in self.evaluators:
            try:
                evaluated = evaluator.evaluate(case, run, trace)
                if isawaitable(evaluated):
                    evaluated = await evaluated
                results.append(evaluated)
            except Exception as exc:  # noqa: BLE001
                results.append(
                    EvaluationResult(
                        evaluator=getattr(evaluator, "name", evaluator.__class__.__name__),
                        score=None,
                        passed=False,
                        explanation=f"Evaluator failed: {exc}",
                        status="error",
                        error=str(exc),
                    )
                )
        return results


@dataclass(slots=True)
class EvaluationCaseResult:
    case: DatasetCase
    run: Run
    trace: Trace
    results: list[EvaluationResult]


@dataclass(slots=True)
class EvaluationRunResult:
    dataset: Dataset
    cases: list[EvaluationCaseResult]

    def all_results(self) -> list[EvaluationResult]:
        collected: list[EvaluationResult] = []
        for case in self.cases:
            collected.extend(case.results)
        return collected

    def summary(self) -> EvaluationSummary:
        return EvaluationSummary(case_count=len(self.cases), results=self.all_results())


async def evaluate_async(
    agent: Callable[..., Any],
    dataset: Dataset,
    suite: EvaluationSuite,
    tracer: Tracer | None = None,
    experiment_id: str | None = None,
) -> EvaluationRunResult:
    tracer = tracer or default_tracer
    cases: list[EvaluationCaseResult] = []
    for case in dataset.cases:
        run = Run(
            id=uuid4().hex,
            experiment_id=experiment_id,
            dataset_case_id=case.id,
            input=case.input,
            status="running",
        )
        trace = Trace(run_id=run.id)
        context = tracer.start_run(run, trace)
        context.__enter__()
        error: Exception | None = None
        try:
            output = await _call_agent(agent, case.input, tracer)
            run.finish(output=output, status="succeeded")
            final_output = Event(
                id=uuid4().hex,
                run_id=run.id,
                type=EventType.FINAL_OUTPUT,
                input=case.input,
                output=output,
                status="succeeded",
            )
            final_output.finish(output=output, status="succeeded")
            trace.add_event(final_output)
        except Exception as exc:  # noqa: BLE001
            error = exc
            run.fail(str(exc))
            trace.add_event(
                tracer.error_event(run_id=run.id, parent_id=None, error=str(exc), input=case.input)
            )
        finally:
            context.__exit__(type(error) if error else None, error, error.__traceback__ if error else None)
        results = await suite.evaluate_case(case, run, trace)
        cases.append(EvaluationCaseResult(case=case, run=run, trace=trace, results=results))
    return EvaluationRunResult(dataset=dataset, cases=cases)


def evaluate(
    agent: Callable[..., Any],
    dataset: Dataset,
    suite: EvaluationSuite,
    tracer: Tracer | None = None,
    experiment_id: str | None = None,
) -> EvaluationRunResult:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(evaluate_async(agent, dataset, suite, tracer=tracer, experiment_id=experiment_id))
    raise RuntimeError("evaluate() cannot run inside an active event loop; use evaluate_async() instead")
