from __future__ import annotations

import json
import os
import runpy
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agenteval import EvaluationSuite, evaluate_inbox_entry_sync
from agenteval.core.evaluator import Evaluator
from agenteval.core.results import EvaluationResult
from agenteval.env import load_env
from agenteval.storage.database import SQLiteDatabase
from agenteval.storage.repositories import Repository

load_env(ROOT / ".env")
load_env(Path(r"C:\Users\sauri\PycharmProjects\data-analyst-agent\.env"))


class PolicyInvariantEvaluator(Evaluator):
    name = "PolicyInvariant"

    def evaluate(self, case, run, trace) -> EvaluationResult:
        output = run.output if isinstance(run.output, dict) else {}
        decision = output.get("decision") or {}
        violations: list[str] = []

        if output.get("guardrail_triggered") and output.get("action") != "force_security_route":
            violations.append("guardrail triggered but action wasn't force_security_route")

        if (
            decision.get("priority") in {"P1-Critical", "P2-High"}
            and output.get("action") in {"auto_route", "auto_resolve"}
            and not output.get("requires_approval")
        ):
            violations.append("high-priority auto action skipped required approval")

        allowed_actions = {"auto_resolve", "auto_route", "auto_route_spotcheck", "force_security_route", "human_review"}
        action = output.get("action")
        if action not in allowed_actions:
            violations.append(f"unknown action: {action}")

        return EvaluationResult(
            evaluator=self.name,
            score=1.0 if not violations else 0.0,
            passed=not violations,
            explanation="; ".join(violations) if violations else "No policy violations detected.",
            metadata={"violations": violations, "output": output},
        )


def _load_data_analyst_evaluator():
    eval_file = ROOT / "eval_data_analyst_agent.py"
    if not eval_file.exists():
        return None
    module = runpy.run_path(str(eval_file))
    cls = module.get("DataAnalystAgentEvaluator")
    return None if cls is None else cls()


def _golden_expected(ticket_id_source: str) -> dict[str, Any] | None:
    golden_path = Path(os.getenv("EVAL_GOLDEN_SET", "")).expanduser().resolve(strict=False)
    if not golden_path.exists():
        return None
    try:
        data = json.loads(golden_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        return data.get(ticket_id_source)
    if isinstance(data, list):
        for item in data:
            if str(item.get("ticket_id_source")) == str(ticket_id_source):
                return item.get("expected")
    return None


def _default_suite_for_payload(payload: dict[str, Any]) -> EvaluationSuite:
    evaluators = [PolicyInvariantEvaluator()]
    expected = _golden_expected(str(payload.get("ticket_id_source") or payload.get("id") or ""))
    if expected:
        custom = _load_data_analyst_evaluator()
        if custom is not None:
            evaluators.append(custom)
    return EvaluationSuite(evaluators)


def _agent_for_resolved_ticket(payload: dict[str, Any]):
    def agent(case: dict[str, Any], tracer=None):
        decision = payload.get("decision") or {}
        guardrail = payload.get("guardrail") or {}
        return {
            "status": payload.get("status") or ("completed" if payload.get("resolved") else "unknown"),
            "action": payload.get("action") or decision.get("recommended_action"),
            "decision": {
                "category": decision.get("category") or payload.get("category"),
                "queue": decision.get("queue") or payload.get("queue"),
                "priority": decision.get("priority") or payload.get("priority"),
                "recommended_action": decision.get("recommended_action"),
                "summary": decision.get("summary") or payload.get("resolution_summary"),
            },
            "guardrail_triggered": bool(payload.get("guardrail_triggered") or guardrail.get("triggered")),
            "requires_approval": bool(payload.get("approval_required") or payload.get("requires_approval")),
            "tool_result": payload.get("tool_result"),
            "classifier_mode_used": payload.get("classifier_mode_used", "unknown"),
        }

    return agent


def _path_from_env() -> Path:
    value = os.getenv("EVAL_MIRROR_FILE", "").strip()
    if value:
        return Path(value).expanduser().resolve(strict=False)
    return ROOT / "mirror_inbox.log"


def _db_path() -> str:
    value = os.getenv("AGENTEVAL_DB", "").strip()
    if value:
        return value
    return str(ROOT / "agenteval.sqlite3")


def _already_processed(payload: dict[str, Any]) -> bool:
    ticket_key = str(payload.get("ticket_id_source") or payload.get("id") or payload.get("kind") or "")
    if not ticket_key:
        return False
    with sqlite3.connect(_db_path()) as conn:
        row = conn.execute(
            "SELECT 1 FROM inbox_entries WHERE payload_json LIKE ? LIMIT 1",
            (f'%"ticket_id_source": "{ticket_key}"%',),
        ).fetchone()
    return row is not None


def main() -> int:
    mirror_path = _path_from_env()
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    if not mirror_path.exists():
        mirror_path.touch()

    repository = Repository(SQLiteDatabase(_db_path()))
    while True:
        try:
            with mirror_path.open("r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("kind") != "resolved_ticket":
                    continue
                if _already_processed(payload):
                    continue
                repository.save_mirror_inbox_entry(payload)

                payload_for_eval = {**payload}
                expected = _golden_expected(str(payload.get("ticket_id_source") or payload.get("id") or ""))
                if expected is not None:
                    payload_for_eval["expected"] = expected

                result = evaluate_inbox_entry_sync(
                    payload_for_eval,
                    agent=_agent_for_resolved_ticket(payload),
                    suite=_default_suite_for_payload(payload),
                )
                for case in result.cases:
                    repository.save_run(case.run, case.trace)
                    repository.save_evaluation_results(case.run.id, case.results)
                print(f"Evaluated mirrored ticket {payload.get('ticket_id_source')} -> {result.summary().detailed_summary()}")
        except Exception as exc:
            print(f"mirror worker warning: {exc}")
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
