from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator
from uuid import uuid4

from ..core.models import Event, EventType, Run, Trace


_current_tracer: ContextVar["Tracer | None"] = ContextVar("agenteval_current_tracer", default=None)
_event_stack: ContextVar[tuple[Event, ...]] = ContextVar("agenteval_event_stack", default=())


def default_event_name(event_type: EventType | str) -> str:
    return event_type.value if isinstance(event_type, EventType) else str(event_type)


@dataclass(slots=True)
class RunContext:
    tracer: "Tracer"
    run: Run
    trace: Trace
    _tracer_token: Any = field(init=False, repr=False, default=None)
    _stack_token: Any = field(init=False, repr=False, default=None)

    def __enter__(self) -> "RunContext":
        self._tracer_token = _current_tracer.set(self.tracer)
        self._stack_token = _event_stack.set(())
        started = Event(
            id=uuid4().hex,
            run_id=self.run.id,
            type=EventType.AGENT_STARTED,
            input=self.run.input,
            status="succeeded",
        )
        started.finish(output=None, status="succeeded")
        self.trace.add_event(started)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            failed = Event(
                id=uuid4().hex,
                run_id=self.run.id,
                type=EventType.AGENT_FAILED,
                input=self.run.input,
                status="failed",
                error=str(exc),
            )
            failed.finish(status="failed", error=str(exc))
            self.trace.add_event(failed)
        else:
            finished = Event(
                id=uuid4().hex,
                run_id=self.run.id,
                type=EventType.AGENT_FINISHED,
                output=self.run.output,
                status="succeeded",
            )
            finished.finish(output=self.run.output, status=self.run.status)
            self.trace.add_event(finished)
        _event_stack.reset(self._stack_token)
        _current_tracer.reset(self._tracer_token)
        return False


@dataclass(slots=True)
class SpanContext:
    tracer: "Tracer"
    run_id: str
    type: EventType | str
    name: str | None = None
    input: Any = None
    metadata: dict[str, Any] | None = None
    _event: Event = field(init=False, repr=False, default=None)

    def __enter__(self) -> Event:
        event = Event(
            id=uuid4().hex,
            run_id=self.run_id,
            type=self.type,
            parent_id=_event_stack.get()[-1].id if _event_stack.get() else None,
            name=self.name,
            input=self.input,
            metadata=self.metadata or {},
        )
        stack = _event_stack.get()
        _event_stack.set(stack + (event,))
        self._event = event
        return event

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self._event.finish(status="failed", error=str(exc))
        elif self._event.timestamp_end is None:
            self._event.finish(status="succeeded")
        stack = _event_stack.get()
        _event_stack.set(stack[:-1])
        self.tracer.current_trace.add_event(self._event)
        return False


class Tracer:
    def __init__(self) -> None:
        self.current_run: Run | None = None
        self.current_trace: Trace | None = None

    def start_run(self, run: Run, trace: Trace | None = None) -> RunContext:
        self.current_run = run
        self.current_trace = trace or Trace(run_id=run.id)
        return RunContext(self, run, self.current_trace)

    @contextmanager
    def run(self, run: Run, trace: Trace | None = None) -> Iterator[RunContext]:
        context = self.start_run(run, trace)
        try:
            yield context.__enter__()
        except Exception as exc:  # noqa: BLE001
            context.__exit__(type(exc), exc, exc.__traceback__)
            raise
        else:
            context.__exit__(None, None, None)

    def span(
        self,
        event_type: EventType | str,
        *,
        name: str | None = None,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> SpanContext:
        if self.current_run is None:
            raise RuntimeError("Tracer.span() requires an active run")
        return SpanContext(self, self.current_run.id, event_type, name=name, input=input, metadata=metadata)

    def error_event(
        self,
        *,
        run_id: str,
        parent_id: str | None,
        error: str,
        input: Any = None,
    ) -> Event:
        event = Event(
            id=uuid4().hex,
            run_id=run_id,
            parent_id=parent_id,
            type=EventType.ERROR,
            input=input,
            error=error,
            status="failed",
        )
        event.finish(status="failed", error=error)
        return event


default_tracer = Tracer()


def get_current_tracer() -> Tracer | None:
    return _current_tracer.get()
