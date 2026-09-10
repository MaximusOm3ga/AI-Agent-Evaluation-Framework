from __future__ import annotations

import os
from pathlib import Path

from agenteval import (
    Dataset,
    DatasetCase,
    EvaluationSuite,
    EventType,
    JudgeConfig,
    LLMJudge,
    TaskSuccess,
    ToolSelection,
    evaluate,
)


def _load_env_file(project_root: str | None = None) -> Path | None:
    candidates: list[Path] = []

    if project_root:
        project_root_path = Path(project_root).expanduser().resolve()
        candidates.append(project_root_path / ".env")
        candidates.append(project_root_path)

    for env_path in (
        os.getenv("TARGET_PROJECT_ENV"),
        os.getenv("AGENT_PROJECT_ENV"),
    ):
        if not env_path:
            continue
        env_file = Path(env_path).expanduser().resolve()
        candidates.append(env_file if env_file.name == ".env" else env_file / ".env")

    repo_root = Path(__file__).resolve().parent
    candidates.extend(
        [
            repo_root.parent / "data-analyst-agent" / ".env",
            repo_root / ".env",
            Path.cwd() / ".env",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve(strict=False)
        if normalized in seen:
            continue
        seen.add(normalized)

        if not normalized.exists():
            continue

        try:
            from dotenv import load_dotenv

            load_dotenv(normalized, override=False)
            return normalized
        except Exception:
            for line in normalized.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return normalized

    return None


_load_env_file(os.getenv("TARGET_PROJECT_ENV"))


def my_agent(task: dict, tracer=None):
    if tracer is not None:
        with tracer.span(EventType.LLM_CALL, name="plan", input=task) as llm_event:
            llm_event.finish(output={"strategy": "lookup revenue then compute growth"}, status="succeeded")

        with tracer.span(EventType.TOOL_CALL, name="finance_lookup", input={"metric": "revenue"}) as tool_event:
            tool_event.finish(
                output={"2024_revenue": 390.0, "2023_revenue": 360.0},
                status="succeeded",
            )

    revenue_2024 = 390.0
    revenue_2023 = 360.0
    growth = ((revenue_2024 - revenue_2023) / revenue_2023) * 100.0
    answer = f"Revenue grew by approximately {growth:.1f}% from 2023 to 2024."
    return answer


dataset = Dataset(
    id="finance_demo",
    name="finance_demo",
    cases=[
        DatasetCase(
            id="case-1",
            input={"task": "Find revenue growth from 2023 to 2024."},
            expected={
                "answer": "Revenue grew by approximately 8.3% from 2023 to 2024.",
                "tools": ["finance_lookup"],
            },
        )
    ],
)

judge = LLMJudge(
    config=JudgeConfig(
        provider="groq",
        model="openai/gpt-oss-20b",
        evaluation_prompt="Judge whether the agent answer matches the task and uses the correct evidence.",
        criteria=["correctness", "completeness", "relevance"],
    )
)

suite = EvaluationSuite(
    [
        TaskSuccess(),
        ToolSelection(),
        judge,
    ]
)

result = evaluate(
    agent=my_agent,
    dataset=dataset,
    suite=suite,
)

print(result.detailed_summary())
print()
print("Aggregate score summary:")
print(result.summary().summary())
