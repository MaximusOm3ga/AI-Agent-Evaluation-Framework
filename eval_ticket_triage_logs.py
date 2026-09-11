from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agenteval import Dataset, DatasetCase, EvaluationSuite, evaluate
from agenteval.core.evaluator import Evaluator
from agenteval.core.models import DatasetCase as CoreDatasetCase, Run, Trace
from agenteval.core.results import EvaluationResult

APP_ROOT = Path(__file__).resolve().parent.parent / "data-analyst-agent"


def _load_env(project_root: Path | None = None) -> None:
    candidates: list[Path] = []
    if project_root:
        candidates.append(project_root / ".env")
    candidates.extend(
        [
            ROOT / ".env",
            APP_ROOT / ".env",
            Path.cwd() / ".env",
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve(strict=False)
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            from dotenv import load_dotenv

            load_dotenv(resolved, override=False)
            return
        except Exception:
            for raw_line in resolved.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return


_ALLOWED_ACTIONS = {"auto_route", "auto_resolve", "auto_route_spotcheck", "force_security_route", "human_review"}
_VALID_STATUSES = {"resolved", "completed", "awaiting_approval", "failed"}


class TicketLogEvaluator(Evaluator):
    name = "TicketLogEvaluator"

    def evaluate(self, case: CoreDatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        payload = run.output if isinstance(run.output, dict) else {}
        request = payload.get("initial_request") or {}
        response = payload.get("agent_response") or {}
        decision = response.get("decision") or payload.get("decision") or {}
        tool_result = response.get("tool_result") or payload.get("tool_result") or {}

        checks = [
            ("request_present", bool(request)),
            ("response_present", bool(response)),
            ("status_valid", str(payload.get("status") or "unknown").lower() in _VALID_STATUSES),
            ("action_valid", str(payload.get("action") or response.get("action") or "").lower() in _ALLOWED_ACTIONS),
            ("category_present", bool(decision.get("category") or payload.get("category"))),
            ("queue_present", bool(decision.get("queue") or payload.get("queue"))),
            ("priority_present", bool(decision.get("priority") or payload.get("priority"))),
            ("summary_present", bool(response.get("summary") or payload.get("resolution_summary") or decision.get("summary"))),
            ("tool_output_present", bool(tool_result)),
        ]
        passed = sum(1 for _, ok in checks if ok)
        total = len(checks)
        score = (passed / total) if total else 1.0

        return EvaluationResult(
            evaluator=self.name,
            score=score,
            passed=score == 1.0,
            explanation=(
                f"Checked {passed}/{total} quality and policy checks on resolved ticket logs."
                if total
                else "No checks were configured for this ticket."
            ),
            metadata={
                "case_id": case.id,
                "request_present": bool(request),
                "response_present": bool(response),
                "actual": payload,
                "checks": {name: ok for name, ok in checks},
            },
            evidence=[event.id for event in trace.ordered_events()[-10:]],
        )


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    request = entry.get("initial_request") or {}
    response = entry.get("agent_response") or {}
    decision = response.get("decision") or {}
    tool_result = response.get("tool_result") or entry.get("tool_result") or {}

    status = (
        entry.get("status")
        or ("resolved" if entry.get("resolved") else None)
        or response.get("status")
        or "unknown"
    )
    action = entry.get("action") or response.get("action") or tool_result.get("action") or decision.get("recommended_action") or "unknown"

    normalized = {
        "ticket_id_source": entry.get("ticket_id_source") or request.get("ticket_id_source") or "unknown",
        "status": str(status),
        "action": str(action),
        "category": entry.get("category") or decision.get("category") or "unknown",
        "queue": entry.get("queue") or decision.get("queue") or "unknown",
        "priority": entry.get("priority") or decision.get("priority") or "unknown",
        "resolution_summary": entry.get("resolution_summary") or response.get("summary") or decision.get("summary") or "",
        "initial_request": request or {
            "ticket_id_source": entry.get("ticket_id_source"),
            "requester_identifier": entry.get("requester"),
            "subject": entry.get("subject"),
            "body_raw": entry.get("body_raw"),
        },
        "agent_response": response or {
            "status": status,
            "action": action,
            "decision": decision,
            "tool_result": tool_result,
            "summary": entry.get("resolution_summary") or decision.get("summary"),
        },
        "tool_result": tool_result,
        "decision": decision,
        "requester": entry.get("requester") or request.get("requester_identifier"),
        "subject": entry.get("subject") or request.get("subject"),
    }
    return normalized


def _load_latest_entries(log_path: Path, limit: int) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            if item.get("kind") == "resolved_ticket" or item.get("resolved") is True:
                entries.append(_normalize_entry(item))
            if limit > 0 and len(entries) >= limit:
                break
    return entries[-limit:] if limit > 0 else entries


def _build_dataset(entries: list[dict[str, Any]]) -> Dataset:
    cases = []
    for idx, entry in enumerate(entries):
        ticket_id = str(entry.get("ticket_id_source") or f"log-entry-{idx}")
        cases.append(
            DatasetCase(
                id=ticket_id,
                input=entry,
                expected={
                    "request_present": True,
                    "response_present": True,
                    "status_valid": True,
                    "action_valid": True,
                    "category_present": bool(entry.get("category") or entry.get("decision", {}).get("category")),
                    "queue_present": bool(entry.get("queue") or entry.get("decision", {}).get("queue")),
                    "priority_present": bool(entry.get("priority") or entry.get("decision", {}).get("priority")),
                    "summary_present": bool(entry.get("resolution_summary") or entry.get("agent_response", {}).get("summary")),
                },
            )
        )
    return Dataset(id="ticket-triage-logs", name="ticket-triage-logs", cases=cases)


def _run_one(agent_response: dict[str, Any], tracer=None):
    return agent_response


def main() -> int:
    _load_env(APP_ROOT)
    _load_env(ROOT)

    parser = argparse.ArgumentParser(description="Evaluate the latest ticket triage log entries against the agent evaluation framework.")
    parser.add_argument("--log-path", default=str(APP_ROOT / "resolved_tickets.log"), help="Path to the ticket log JSONL file.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of latest resolved tickets to evaluate.")
    args = parser.parse_args()

    log_path = Path(args.log_path).expanduser().resolve(strict=False)
    entries = _load_latest_entries(log_path, args.limit)
    if not entries:
        print(f"No resolved ticket entries found in {log_path}")
        return 0

    dataset = _build_dataset(entries)
    result = evaluate(
        agent=_run_one,
        dataset=dataset,
        suite=EvaluationSuite([TicketLogEvaluator()]),
    )

    print(f"Evaluated {len(result.cases)} resolved ticket log entries from {log_path}")
    scores = []
    for case in result.cases:
        if not case.results:
            continue
        score_value = case.results[0].score
        try:
            scores.append(float(score_value))
        except (TypeError, ValueError):
            scores.append(0.0)
    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    passed_cases = sum(1 for case in result.cases if case.results and all(r.passed for r in case.results))
    print(f"Average score: {avg_score:.2f}")
    print(f"Passed cases: {passed_cases}/{len(result.cases)}")
    print()

    for case in result.cases:
        eval_result = case.results[0] if case.results else None
        ticket_id = case.case.id if getattr(case, "case", None) is not None else "unknown"
        passed = bool(eval_result.passed) if eval_result is not None else False
        score_value = getattr(eval_result, "score", None)
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0
        label = "PASS" if passed else "FAIL"
        print(f"[{label}] {ticket_id} score={score:.2f}")
        if eval_result is None:
            print("  No evaluation result produced.")
            continue

        print(f"  Reason: {getattr(eval_result, 'explanation', 'No explanation available.')}")
        status = getattr(eval_result, 'status', None)
        if status == 'error':
            error_text = getattr(eval_result, 'error', 'Unknown evaluator error.')
            print(f"  Error: {error_text}")

        metadata = getattr(eval_result, 'metadata', {}) or {}
        checks = metadata.get('checks') if isinstance(metadata, dict) else None
        if isinstance(checks, dict):
            failed_checks = [name for name, ok in checks.items() if not ok]
            if failed_checks:
                print("  Failed checks:")
                for name in failed_checks:
                    print(f"    - {name}")
            else:
                print("  All checks passed.")

        actual = metadata.get('actual') if isinstance(metadata, dict) else None
        if isinstance(actual, dict):
            print("  Ticket summary:")
            print(f"    ticket_id_source={actual.get('ticket_id_source') or 'missing'}")
            print(f"    status={actual.get('status') or 'missing'}")
            print(f"    action={actual.get('action') or 'missing'}")
            print(f"    category={actual.get('category') or (actual.get('decision') or {}).get('category') or 'missing'}")
            print(f"    queue={actual.get('queue') or (actual.get('decision') or {}).get('queue') or 'missing'}")
            print(f"    priority={actual.get('priority') or (actual.get('decision') or {}).get('priority') or 'missing'}")
            print(f"    resolution_summary={actual.get('resolution_summary') or (actual.get('agent_response') or {}).get('summary') or 'missing'}")

        request = actual.get('initial_request') if isinstance(actual, dict) else None
        response = actual.get('agent_response') if isinstance(actual, dict) else None
        if isinstance(request, dict):
            print("  Initial request fields:")
            for key in ['ticket_id_source', 'requester_identifier', 'subject', 'body_raw', 'body_cleaned']:
                value = request.get(key)
                print(f"    - {key}: {value if value not in (None, '') else 'MISSING'}")
        if isinstance(response, dict):
            print("  Agent response fields:")
            for key in ['status', 'action', 'summary']:
                value = response.get(key)
                print(f"    - {key}: {value if value not in (None, '') else 'MISSING'}")

        if not passed and getattr(eval_result, 'metadata', None):
            print("  Metadata:")
            print(f"    {json.dumps(metadata, ensure_ascii=True, default=str, indent=2)[:1500]}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
