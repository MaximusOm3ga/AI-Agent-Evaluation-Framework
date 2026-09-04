from __future__ import annotations

from functools import wraps
from inspect import isawaitable, iscoroutinefunction, signature
from typing import Any, Callable
from uuid import uuid4

from ..core.models import Event, EventType, Run, Trace
from .tracer import Tracer, default_tracer


def trace(func: Callable[..., Any] | None = None, *, tracer: Tracer | None = None):
    tracer = tracer or default_tracer

    def decorator(callable_obj: Callable[..., Any]):
        def _call_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
            call_kwargs = dict(kwargs)
            try:
                sig = signature(callable_obj)
                if "tracer" in sig.parameters or any(
                    param.kind.name == "VAR_KEYWORD" for param in sig.parameters.values()
                ):
                    call_kwargs.setdefault("tracer", tracer)
            except (TypeError, ValueError):
                call_kwargs.setdefault("tracer", tracer)
            return call_kwargs

        @wraps(callable_obj)
        def sync_wrapper(*args, **kwargs):
            if tracer.current_run is not None:
                return callable_obj(*args, **_call_kwargs(kwargs))
            run = Run(id=uuid4().hex, experiment_id=None, dataset_case_id="manual", input=args[0] if args else None)
            trace_obj = Trace(run_id=run.id)
            context = tracer.start_run(run, trace_obj)
            context.__enter__()
            error: Exception | None = None
            try:
                result = callable_obj(*args, **_call_kwargs(kwargs))
                run.finish(output=result)
                final_output = Event(
                    id=uuid4().hex,
                    run_id=run.id,
                    type=EventType.FINAL_OUTPUT,
                    input=args[0] if args else None,
                    output=result,
                    status="succeeded",
                )
                final_output.finish(output=result, status="succeeded")
                trace_obj.add_event(final_output)
            except Exception as exc:  # noqa: BLE001
                error = exc
                run.fail(str(exc))
                raise
            finally:
                context.__exit__(type(error) if error else None, error, error.__traceback__ if error else None)
            return result

        @wraps(callable_obj)
        async def async_wrapper(*args, **kwargs):
            if tracer.current_run is not None:
                result = callable_obj(*args, **_call_kwargs(kwargs))
                if isawaitable(result):
                    return await result
                return result
            run = Run(id=uuid4().hex, experiment_id=None, dataset_case_id="manual", input=args[0] if args else None)
            trace_obj = Trace(run_id=run.id)
            context = tracer.start_run(run, trace_obj)
            context.__enter__()
            error: Exception | None = None
            try:
                result = callable_obj(*args, **_call_kwargs(kwargs))
                if isawaitable(result):
                    result = await result
                run.finish(output=result)
                final_output = Event(
                    id=uuid4().hex,
                    run_id=run.id,
                    type=EventType.FINAL_OUTPUT,
                    input=args[0] if args else None,
                    output=result,
                    status="succeeded",
                )
                final_output.finish(output=result, status="succeeded")
                trace_obj.add_event(final_output)
            except Exception as exc:  # noqa: BLE001
                error = exc
                run.fail(str(exc))
                raise
            finally:
                context.__exit__(type(error) if error else None, error, error.__traceback__ if error else None)
            return result

        return async_wrapper if iscoroutinefunction(callable_obj) else sync_wrapper

    return decorator(func) if func is not None else decorator
