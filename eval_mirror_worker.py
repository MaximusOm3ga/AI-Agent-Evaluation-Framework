from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agenteval import EvaluationSuite, evaluate_inbox_entry_sync
from agenteval.evaluators.deterministic import AnswerCorrectness, ExactMatch, TaskSuccess
from agenteval.storage.database import SQLiteDatabase
from agenteval.storage.repositories import Repository


def _load_data_analyst_evaluator():
    eval_file = ROOT / "eval_data_analyst_agent.py"
    if not eval_file.exists():
        return None
    module = runpy.run_path(str(eval_file))
    cls = module.get("DataAnalystAgentEvaluator")
    if cls is None:
        return None
    return cls()


def _infer_expected_for_ticket(payload: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        str(part)
        for part in [
            payload.get("subject"),
            payload.get("resolution_summary"),
            payload.get("category"),
            payload.get("queue"),
            payload.get("priority"),
            payload.get("tool_result", {}).get("resolution_summary"),
        ]
        if part
    ).lower()

    if "password" in text or "forgot" in text or "reset" in text:
        return {
            "status": "completed",
            "action": "auto_resolve",
            "category": "Access Request",
            "recommended_action": "auto_resolve",
            "queue": "ServiceDesk-L1",
            "priority": "P3-Medium",
        }
    if "vpn" in text or "network" in text:
        return {
            "status": "completed",
            "action": "auto_route",
            "recommended_action": "auto_route",
            "category": "Network/VPN",
            "queue": "Network-Eng",
            "priority": "P2-High",
        }
    if "screen" in text or "flicker" in text or "device" in text or "hardware" in text:
        return {
            "status": "awaiting_approval",
            "action": "auto_route_spotcheck",
            "recommended_action": "auto_route",
            "guardrail_triggered": False,
        }
    if "terminated" in text or "phish" in text or "security" in text or "access after termination" in text:
        return {
            "status": "awaiting_approval",
            "action": "force_security_route",
            "category": "Security Incident",
            "recommended_action": "auto_route",
            "guardrail_triggered": True,
            "queue": "Security",
            "priority": "P1-Critical",
        }
    if "install" in text or "license" in text or "provision" in text:
        return {
            "status": "completed",
            "action": "auto_resolve",
            "recommended_action": "auto_resolve",
            "category": "Software Install",
        }
    return {
        "status": payload.get("status") or "completed",
        "action": payload.get("action") or "auto_route",
        "category": payload.get("category"),
    }


def _agent_for_resolved_ticket(payload: dict[str, Any]):
    def agent(case: dict[str, Any], tracer=None):
        decision = case.get("decision") or {}
        return {
            "status": "completed",
            "action": "auto_resolve",
            "decision": {
                "category": decision.get("category") or case.get("category") or "Other",
                "queue": decision.get("queue") or case.get("queue") or "ServiceDesk-L1",
                "priority": decision.get("priority") or case.get("priority") or "P4-Low",
                "recommended_action": "auto_resolve",
                "summary": decision.get("summary") or case.get("resolution_summary") or "Resolved via mirror evaluation",
            },
            "guardrail_triggered": bool(case.get("guardrail_triggered")),
            "requires_approval": False,
            "classifier_mode_used": "mirrored_eval",
        }

    return agent


def _default_suite() -> EvaluationSuite:
    evaluators = [
        TaskSuccess(),
        ExactMatch(),
        AnswerCorrectness(),
    ]
    custom = _load_data_analyst_evaluator()
    if custom is not None:
        evaluators.append(custom)
    return EvaluationSuite(evaluators)


def _path_from_env() -> Path:
    value = os.getenv("EVAL_MIRROR_FILE", "").strip()
    if value:
        return Path(value)
    return ROOT / "mirror_inbox.log"


def _db_path() -> str:
    value = os.getenv("AGENTEVAL_DB", "").strip()
    if value:
        return value
    return str(ROOT / "agenteval.sqlite3")


def _already_processed(path: Path, line: str) -> bool:
    try:
        payload = json.loads(line)
    except Exception:
        return True
    key = payload.get("ticket_id_source") or payload.get("id") or payload.get("kind")
    if not key:
        return True
    with sqlite3_connect(_db_path()) as conn:
        row = conn.execute(
            "SELECT 1 FROM inbox_entries WHERE payload_json = ? LIMIT 1",
            (json.dumps(payload, ensure_ascii=True),),
        ).fetchone()
    return row is not None


def sqlite3_connect(db_path: str):
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def main() -> int:
    mirror_path = _path_from_env()
    mirror_path.parent.mkdir(parents=True, exist_ok=True)
    if not mirror_path.exists():
        mirror_path.touch()
    repository = Repository(SQLiteDatabase(_db_path()))
    seen: set[str] = set()
    while True:
        try:
            if not mirror_path.exists():
                time.sleep(1)
                continue
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
                fingerprint = json.dumps(payload, sort_keys=True)
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                repository.save_mirror_inbox_entry(payload)
                expected = _infer_expected_for_ticket(payload)
                payload_for_eval = {**payload, "expected": expected}
                result = evaluate_inbox_entry_sync(
                    payload_for_eval,
                    agent=_agent_for_resolved_ticket(payload),
                    suite=_default_suite(),
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
