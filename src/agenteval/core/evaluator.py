from __future__ import annotations

from abc import ABC
from typing import Any
import inspect

from .models import DatasetCase, Run, Trace
from .results import EvaluationResult


class Evaluator(ABC):
    name = "evaluator"

    def evaluate(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult | Any:
        raise NotImplementedError

    async def evaluate_async(self, case: DatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        result = self.evaluate(case, run, trace)
        if inspect.isawaitable(result):
            return await result
        return result
