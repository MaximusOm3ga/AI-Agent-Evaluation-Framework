from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any
import json
from uuid import uuid4


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def from_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def dumps(value: Any) -> str:
    return json.dumps(value, default=json_default, ensure_ascii=True)


def loads(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


class EventType(StrEnum):
    AGENT_STARTED = "AGENT_STARTED"
    AGENT_FINISHED = "AGENT_FINISHED"
    AGENT_FAILED = "AGENT_FAILED"
    LLM_CALL = "LLM_CALL"
    LLM_RESPONSE = "LLM_RESPONSE"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    TOOL_ERROR = "TOOL_ERROR"
    RETRIEVAL = "RETRIEVAL"
    EMBEDDING = "EMBEDDING"
    RERANK = "RERANK"
    STATE_CHANGE = "STATE_CHANGE"
    FINAL_OUTPUT = "FINAL_OUTPUT"
    ERROR = "ERROR"


@dataclass(slots=True)
class DatasetCase:
    id: str
    input: Any
    expected: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetCase":
        return cls(
            id=data["id"],
            input=data.get("input"),
            expected=data.get("expected"),
            metadata=data.get("metadata") or {},
        )


@dataclass(slots=True)
class Dataset:
    id: str
    name: str
    cases: list[DatasetCase]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            cases = [DatasetCase.from_dict(item) for item in raw]
            dataset_id = Path(path).stem
            return cls(id=dataset_id, name=dataset_id, cases=cases)
        cases = [DatasetCase.from_dict(item) for item in raw.get("cases", [])]
        dataset_id = raw.get("id") or Path(path).stem
        return cls(
            id=dataset_id,
            name=raw.get("name") or dataset_id,
            cases=cases,
            metadata=raw.get("metadata") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "cases": [case.to_dict() for case in self.cases],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Run:
    id: str
    experiment_id: str | None
    dataset_case_id: str
    input: Any
    output: Any = None
    status: str = "pending"
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None
    duration: float | None = None
    token_usage: dict[str, Any] = field(default_factory=dict)
    estimated_cost: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None

    def finish(self, output: Any = None, status: str = "succeeded") -> None:
        self.output = output
        self.status = status
        self.finished_at = utcnow()
        self.duration = (self.finished_at - self.started_at).total_seconds()

    def fail(self, error: str) -> None:
        self.status = "failed"
        self.metadata["error"] = error
        self.finished_at = utcnow()
        self.duration = (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "dataset_case_id": self.dataset_case_id,
            "input": self.input,
            "output": self.output,
            "status": self.status,
            "started_at": to_iso(self.started_at),
            "finished_at": to_iso(self.finished_at),
            "duration": self.duration,
            "token_usage": self.token_usage,
            "estimated_cost": self.estimated_cost,
            "metadata": self.metadata,
            "trace_id": self.trace_id,
        }


@dataclass(slots=True)
class Event:
    id: str
    run_id: str
    type: EventType | str
    parent_id: str | None = None
    name: str | None = None
    timestamp_start: datetime = field(default_factory=utcnow)
    timestamp_end: datetime | None = None
    input: Any = None
    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    error: str | None = None

    def finish(self, output: Any = None, status: str = "succeeded", error: str | None = None) -> None:
        self.output = output
        self.status = status
        self.error = error
        self.timestamp_end = utcnow()

    @property
    def duration(self) -> float | None:
        if not self.timestamp_end:
            return None
        return (self.timestamp_end - self.timestamp_start).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "parent_id": self.parent_id,
            "name": self.name,
            "timestamp_start": to_iso(self.timestamp_start),
            "timestamp_end": to_iso(self.timestamp_end),
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(
            id=data["id"],
            run_id=data["run_id"],
            type=data["type"],
            parent_id=data.get("parent_id"),
            name=data.get("name"),
            timestamp_start=from_iso(data.get("timestamp_start")) or utcnow(),
            timestamp_end=from_iso(data.get("timestamp_end")),
            input=data.get("input"),
            output=data.get("output"),
            metadata=data.get("metadata") or {},
            status=data.get("status") or "running",
            error=data.get("error"),
        )


@dataclass(slots=True)
class Trace:
    run_id: str
    events: list[Event] = field(default_factory=list)

    def add_event(self, event: Event) -> None:
        self.events.append(event)
        self.events.sort(key=lambda item: item.timestamp_start)

    def ordered_events(self) -> list[Event]:
        return sorted(self.events, key=lambda item: item.timestamp_start)

    def children_of(self, parent_id: str | None) -> list[Event]:
        return [event for event in self.events if event.parent_id == parent_id]

    def filter(self, event_type: EventType | str) -> list[Event]:
        wanted = event_type.value if isinstance(event_type, EventType) else event_type
        return [event for event in self.events if (event.type.value if isinstance(event.type, EventType) else event.type) == wanted]

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "events": [event.to_dict() for event in self.events]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        return cls(run_id=data["run_id"], events=[Event.from_dict(item) for item in data.get("events", [])])

    @classmethod
    def empty(cls, run_id: str | None = None) -> "Trace":
        return cls(run_id=run_id or uuid4().hex)
