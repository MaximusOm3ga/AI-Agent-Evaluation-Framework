from __future__ import annotations

from dataclasses import asdict
from typing import Any
from uuid import uuid4
import json

from ..core.models import Dataset, DatasetCase, Event, Run, Trace, dumps, loads
from ..core.results import EvaluationResult
from .database import SQLiteDatabase


def _dump(value: Any) -> str | None:
    return None if value is None else dumps(value)


class Repository:
    def __init__(self, database: SQLiteDatabase) -> None:
        self.database = database
        self.database.initialize()

    def save_dataset(self, dataset: Dataset) -> None:
        with self.database.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO datasets (id, name, metadata_json) VALUES (?, ?, ?)",
                (dataset.id, dataset.name, _dump(dataset.metadata)),
            )
            connection.execute("DELETE FROM dataset_cases WHERE dataset_id = ?", (dataset.id,))
            for case in dataset.cases:
                connection.execute(
                    "INSERT OR REPLACE INTO dataset_cases (id, dataset_id, input_json, expected_json, metadata_json) VALUES (?, ?, ?, ?, ?)",
                    (
                        case.id,
                        dataset.id,
                        _dump(case.input),
                        _dump(case.expected),
                        _dump(case.metadata),
                    ),
                )

    def save_run(self, run: Run, trace: Trace) -> None:
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO runs
                (id, experiment_id, dataset_case_id, input_json, output_json, status, started_at, finished_at, duration, token_usage_json, estimated_cost, metadata_json, trace_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.experiment_id,
                    run.dataset_case_id,
                    _dump(run.input),
                    _dump(run.output),
                    run.status,
                    run.started_at.isoformat(),
                    run.finished_at.isoformat() if run.finished_at else None,
                    run.duration,
                    _dump(run.token_usage),
                    run.estimated_cost,
                    _dump(run.metadata),
                    run.trace_id,
                ),
            )
            connection.execute("DELETE FROM events WHERE run_id = ?", (run.id,))
            for event in trace.ordered_events():
                connection.execute(
                    """
                    INSERT OR REPLACE INTO events
                    (id, run_id, parent_id, type, name, timestamp_start, timestamp_end, input_json, output_json, metadata_json, status, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.id,
                        event.run_id,
                        event.parent_id,
                        event.type.value if hasattr(event.type, "value") else event.type,
                        event.name,
                        event.timestamp_start.isoformat(),
                        event.timestamp_end.isoformat() if event.timestamp_end else None,
                        _dump(event.input),
                        _dump(event.output),
                        _dump(event.metadata),
                        event.status,
                        event.error,
                    ),
                )

    def save_evaluation_results(self, run_id: str, results: list[EvaluationResult]) -> None:
        with self.database.connect() as connection:
            connection.execute("DELETE FROM evaluation_results WHERE run_id = ?", (run_id,))
            for result in results:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO evaluation_results
                    (id, run_id, evaluator, score, passed, explanation, status, metadata_json, evidence_json, error)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid4().hex,
                        run_id,
                        result.evaluator,
                        result.score,
                        1 if result.passed else 0 if result.passed is not None else None,
                        result.explanation,
                        result.status,
                        _dump(result.metadata),
                        _dump(result.evidence),
                        result.error,
                    ),
                )

    def load_run(self, run_id: str) -> Run:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            return Run(
                id=row["id"],
                experiment_id=row["experiment_id"],
                dataset_case_id=row["dataset_case_id"],
                input=loads(row["input_json"]),
                output=loads(row["output_json"]),
                status=row["status"],
                started_at=__import__("datetime").datetime.fromisoformat(row["started_at"]),
                finished_at=__import__("datetime").datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
                duration=row["duration"],
                token_usage=loads(row["token_usage_json"]) or {},
                estimated_cost=row["estimated_cost"],
                metadata=loads(row["metadata_json"]) or {},
                trace_id=row["trace_id"],
            )

    def load_trace(self, run_id: str) -> Trace:
        with self.database.connect() as connection:
            rows = connection.execute("SELECT * FROM events WHERE run_id = ? ORDER BY timestamp_start", (run_id,)).fetchall()
            return Trace(
                run_id=run_id,
                events=[
                    Event(
                        id=row["id"],
                        run_id=row["run_id"],
                        parent_id=row["parent_id"],
                        type=row["type"],
                        name=row["name"],
                        timestamp_start=__import__("datetime").datetime.fromisoformat(row["timestamp_start"]),
                        timestamp_end=__import__("datetime").datetime.fromisoformat(row["timestamp_end"]) if row["timestamp_end"] else None,
                        input=loads(row["input_json"]),
                        output=loads(row["output_json"]),
                        metadata=loads(row["metadata_json"]) or {},
                        status=row["status"],
                        error=row["error"],
                    )
                    for row in rows
                ],
            )

    def list_runs(self, experiment_id: str | None = None) -> list[Run]:
        query = "SELECT * FROM runs"
        params: tuple[Any, ...] = ()
        if experiment_id is not None:
            query += " WHERE experiment_id = ?"
            params = (experiment_id,)
        query += " ORDER BY started_at"
        with self.database.connect() as connection:
            rows = connection.execute(query, params).fetchall()
            return [self.load_run(row["id"]) for row in rows]

    def list_evaluation_results(self, run_id: str) -> list[EvaluationResult]:
        with self.database.connect() as connection:
            rows = connection.execute("SELECT * FROM evaluation_results WHERE run_id = ? ORDER BY evaluator", (run_id,)).fetchall()
            results: list[EvaluationResult] = []
            for row in rows:
                results.append(
                    EvaluationResult(
                        evaluator=row["evaluator"],
                        score=row["score"],
                        passed=bool(row["passed"]) if row["passed"] is not None else None,
                        explanation=row["explanation"],
                        status=row["status"],
                        metadata=loads(row["metadata_json"]) or {},
                        evidence=loads(row["evidence_json"]) or [],
                        error=row["error"],
                    )
                )
            return results
