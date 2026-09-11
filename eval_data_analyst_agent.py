from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from textwrap import indent

import requests

from agenteval import Dataset, DatasetCase, EvaluationSuite, evaluate
from agenteval.core.evaluator import Evaluator
from agenteval.core.models import DatasetCase as CoreDatasetCase, Run, Trace
from agenteval.core.results import EvaluationResult


def _load_project_env(project_root: str | None = None) -> None:
    env_candidates: list[Path] = []
    if project_root:
        env_candidates.append(Path(project_root).expanduser().resolve() / ".env")
    env_candidates.extend(
        [
            Path(os.getenv("TARGET_PROJECT_ENV", "")).expanduser().resolve() / ".env",
            Path.cwd() / ".env",
            Path(__file__).resolve().parent.parent / "data-analyst-agent" / ".env",
        ]
    )
    seen: set[Path] = set()
    for candidate in env_candidates:
        try:
            resolved = candidate.resolve(strict=False)
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
            for line in resolved.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return


def _ascii_text(value: object) -> str:
    return str(value).encode("ascii", "replace").decode("ascii")


def _format_value(value: object, width: int = 100) -> str:
    text = json.dumps(value, ensure_ascii=True, indent=2, default=str)
    if len(text) <= width:
        return text
    return text


def _status_icon(passed: bool | None) -> str:
    if passed is True:
        return "PASS"
    if passed is False:
        return "FAIL"
    return "N/A"


class DataAnalystAgentEvaluator(Evaluator):
    name = "DataAnalystAgentEvaluator"

    def evaluate(self, case: CoreDatasetCase, run: Run, trace: Trace) -> EvaluationResult:
        payload = run.output if isinstance(run.output, dict) else {}
        checks: list[tuple[str, bool]] = []
        expected = case.expected or {}

        if expected.get("status") is not None:
            checks.append(("status", payload.get("status") == expected["status"]))
        if expected.get("action") is not None:
            checks.append(("action", payload.get("action") == expected["action"]))
        if expected.get("category") is not None:
            checks.append(("category", payload.get("decision", {}).get("category") == expected["category"]))
        if expected.get("recommended_action") is not None:
            checks.append(
                ("recommended_action", payload.get("decision", {}).get("recommended_action") == expected["recommended_action"])
            )
        if expected.get("guardrail_triggered") is not None:
            checks.append(("guardrail_triggered", payload.get("guardrail_triggered") == expected["guardrail_triggered"]))
        if expected.get("priority") is not None:
            checks.append(("priority", payload.get("decision", {}).get("priority") == expected["priority"]))
        if expected.get("queue") is not None:
            checks.append(("queue", payload.get("decision", {}).get("queue") == expected["queue"]))

        passed = sum(1 for _, ok in checks if ok)
        total = len(checks)
        score = (passed / total) if total else 1.0

        return EvaluationResult(
            evaluator=self.name,
            score=score,
            passed=score == 1.0,
            explanation=(
                f"Checked {passed}/{total} expected properties against the real data-analyst-agent response."
                if total
                else "No checks were configured for this case."
            ),
            metadata={
                "case_id": case.id,
                "expected": expected,
                "actual": payload,
                "checks": {name: ok for name, ok in checks},
            },
            evidence=[event.id for event in trace.ordered_events()[-10:]],
        )


def triage_agent(case: dict, tracer=None):
    url = os.getenv("DATA_ANALYST_AGENT_URL", "http://127.0.0.1:8000/ingest/web_form")
    payload = {
        "ticket_id_source": case["ticket_id_source"],
        "source_channel": case.get("source_channel", "web_form"),
        "requester_identifier": case.get("requester_identifier", "user@example.com"),
        "subject": case.get("subject"),
        "body_raw": case["body_raw"],
        "body_cleaned": case.get("body_cleaned"),
        "attachments": case.get("attachments", []),
        "timestamp_received": datetime.now(timezone.utc).isoformat(),
        "channel_metadata": case.get("channel_metadata", {}),
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    body = response.json()
    if tracer is not None:
        with tracer.span("HTTP_CALL", name="triage_agent_post", input=payload) as event:
            event.finish(output=body, status="succeeded")
    return body


def build_dataset() -> Dataset:
    return Dataset(
        id="data-analyst-agent-eval",
        name="data-analyst-agent-eval",
        cases=[
            DatasetCase(
                id="password-reset",
                input={
                    "ticket_id_source": "eval-web-1",
                    "source_channel": "web_form",
                    "requester_identifier": "user@example.com",
                    "subject": "Need password reset",
                    "body_raw": "I forgot my password and need a reset",
                },
                expected={
                    "status": "completed",
                    "action": "auto_route",
                    "category": "Account/Password",
                    "recommended_action": "auto_route",
                    "queue": "ServiceDesk-L1",
                    "priority": "P3-Medium",
                },
            ),
            DatasetCase(
                id="phishing-security",
                input={
                    "ticket_id_source": "eval-web-2",
                    "source_channel": "web_form",
                    "requester_identifier": "user2@example.com",
                    "subject": "Urgent suspicious email",
                    "body_raw": "I think this is phishing and maybe a data breach.",
                },
                expected={
                    "status": "awaiting_approval",
                    "action": "force_security_route",
                    "category": "Security Incident",
                    "recommended_action": "auto_route",
                    "guardrail_triggered": True,
                    "priority": "P1-Critical",
                    "queue": "Security",
                },
            ),
            DatasetCase(
                id="unknown-device-issue",
                input={
                    "ticket_id_source": "eval-web-3",
                    "source_channel": "web_form",
                    "requester_identifier": "user3@example.com",
                    "subject": "My screen flickers",
                    "body_raw": "Sometimes the display flickers when I open the browser",
                },
                expected={
                    "status": "completed",
                    "action": "auto_route",
                    "category": "Hardware",
                    "recommended_action": "auto_route",
                    "guardrail_triggered": False,
                    "queue": "ServiceDesk-L1",
                    "priority": "P3-Medium",
                },
            ),
            DatasetCase(
                id="vpn-connectivity",
                input={
                    "ticket_id_source": "eval-web-4",
                    "source_channel": "web_form",
                    "requester_identifier": "user4@example.com",
                    "subject": "VPN does not connect",
                    "body_raw": "VPN connection fails when I try to log in from home.",
                },
                expected={
                    "status": "awaiting_approval",
                    "action": "auto_route",
                    "recommended_action": "auto_route",
                    "category": "Network/VPN",
                    "queue": "Network-Eng",
                    "priority": "P2-High",
                    "guardrail_triggered": False,
                },
            ),
            DatasetCase(
                id="software-install",
                input={
                    "ticket_id_source": "eval-web-5",
                    "source_channel": "web_form",
                    "requester_identifier": "user5@example.com",
                    "subject": "Need license for project software",
                    "body_raw": "Please install and provision the design software license.",
                },
                expected={
                    "status": "completed",
                    "action": "auto_route",
                    "recommended_action": "auto_route",
                    "category": "Software Install",
                    "queue": "Software-Provisioning",
                    "priority": "P3-Medium",
                },
            ),
            DatasetCase(
                id="terminated-user-security",
                input={
                    "ticket_id_source": "eval-web-6",
                    "source_channel": "web_form",
                    "requester_identifier": "user6@example.com",
                    "subject": "Access request after termination",
                    "body_raw": "The terminated employee still has access to internal tools.",
                    "channel_metadata": {"employment_status": "terminated"},
                },
                expected={
                    "status": "awaiting_approval",
                    "action": "auto_route",
                    "category": "Security Incident",
                    "recommended_action": "auto_route",
                    "guardrail_triggered": False,
                    "queue": "Security",
                    "priority": "P2-High",
                },
            ),
        ],
    )


def _format_case(case_result) -> str:
    case = case_result.case
    run = case_result.run
    lines = [
        f"[{case.id}] {run.status.upper()} | {run.duration:.3f}s",
        f"  request : {case.input.get('subject') or case.input.get('body_raw')}",
        f"  action  : {run.output.get('action') if isinstance(run.output, dict) else run.output}",
        f"  status  : {run.output.get('status') if isinstance(run.output, dict) else run.status}",
    ]
    if case.expected:
        lines.append(f"  expect  : {case.expected}")
    for result in case_result.results:
        lines.append(
            f"  eval    : {result.evaluator} | {_status_icon(result.passed)} | score={result.score if result.score is not None else 'n/a'}"
        )
        if result.explanation:
            lines.append(f"            {result.explanation}")
        if result.metadata:
            lines.append(f"            metadata={_format_value(result.metadata)}")
    return "\n".join(lines)


def main() -> int:
    _load_project_env(os.getenv("TARGET_PROJECT_ENV"))
    result = evaluate(
        agent=triage_agent,
        dataset=build_dataset(),
        suite=EvaluationSuite([DataAnalystAgentEvaluator()]),
    )

    print(_ascii_text(f"Dataset: {result.dataset.name}"))
    print(_ascii_text(f"Cases: {len(result.cases)}"))
    print("")
    for case_result in result.cases:
        print(_ascii_text(_format_case(case_result)))
        print("")

    print("Aggregate:")
    for line in result.summary().detailed_summary().splitlines():
        print(_ascii_text(f"  {line}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
