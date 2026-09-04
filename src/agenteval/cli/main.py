from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import json
import sys

from .. import (
    AnswerCorrectness,
    Cost,
    Dataset,
    EvaluationSuite,
    ExactMatch,
    Latency,
    LoopDetection,
    TaskSuccess,
    TokenUsage,
    ToolArgumentCorrectness,
    ToolErrorHandling,
    ToolSelection,
    evaluate,
)
from ..storage.database import SQLiteDatabase
from ..storage.repositories import Repository


def _load_callable(ref: str):
    module_name, _, attr = ref.partition(":")
    module = import_module(module_name)
    return getattr(module, attr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agenteval")
    parser.add_argument("--db", default="agenteval.sqlite3")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run")
    run_cmd.add_argument("--dataset", required=True)
    run_cmd.add_argument("--agent", required=True)
    run_cmd.add_argument("--experiment-id")

    inspect_cmd = sub.add_parser("inspect")
    inspect_cmd.add_argument("run_id")

    compare_cmd = sub.add_parser("compare")
    compare_cmd.add_argument("left")
    compare_cmd.add_argument("right")

    failures_cmd = sub.add_parser("failures")
    failures_cmd.add_argument("experiment_id")

    report_cmd = sub.add_parser("report")
    report_cmd.add_argument("experiment_id")

    return parser


def _default_suite() -> EvaluationSuite:
    return EvaluationSuite(
        [
            TaskSuccess(),
            ExactMatch(),
            AnswerCorrectness(),
            ToolSelection(),
            ToolArgumentCorrectness(),
            ToolErrorHandling(),
            LoopDetection(),
            Latency(),
            TokenUsage(),
            Cost(),
        ]
    )


def cmd_run(args: argparse.Namespace) -> int:
    dataset = Dataset.load(args.dataset)
    agent = _load_callable(args.agent)
    result = evaluate(agent=agent, dataset=dataset, suite=_default_suite(), experiment_id=args.experiment_id)
    repository = Repository(SQLiteDatabase(args.db))
    for case in result.cases:
        repository.save_run(case.run, case.trace)
    for case in result.cases:
        repository.save_evaluation_results(case.run.id, case.results)
    print(result.summary().summary())
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    repository = Repository(SQLiteDatabase(args.db))
    run = repository.load_run(args.run_id)
    trace = repository.load_trace(args.run_id)
    results = repository.list_evaluation_results(args.run_id)
    print(f"Run: {run.id}")
    print(f"Status: {run.status}")
    print(f"Case: {run.dataset_case_id}")
    print(f"Duration: {run.duration:.3f}s" if run.duration is not None else "Duration: n/a")
    print()
    print("Trace:")
    for index, event in enumerate(trace.ordered_events(), start=1):
        print(f"[{index:02d}] {event.type}")
        if event.name:
            print(f"    {event.name}")
        if event.output is not None:
            print(f"    {event.output}")
    if results:
        print()
        print("Evaluations:")
        for result in results:
            print(f"- {result.evaluator}: {result.score}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    repository = Repository(SQLiteDatabase(args.db))
    left = repository.list_runs(args.left)
    right = repository.list_runs(args.right)
    left_summary = _summarize_runs(repository, left)
    right_summary = _summarize_runs(repository, right)
    print(f"Experiment {args.left}")
    print(left_summary)
    print()
    print(f"Experiment {args.right}")
    print(right_summary)
    return 0


def _summarize_runs(repository: Repository, runs):
    from statistics import mean

    scores: dict[str, list[float]] = {}
    for run in runs:
        for result in repository.list_evaluation_results(run.id):
            if result.score is not None:
                scores.setdefault(result.evaluator, []).append(result.score)
    lines = []
    for evaluator, values in sorted(scores.items()):
        value = mean(values)
        lines.append(f"{evaluator}: {value:.3f}")
    return "\n".join(lines) if lines else "No results"


def cmd_failures(args: argparse.Namespace) -> int:
    repository = Repository(SQLiteDatabase(args.db))
    runs = repository.list_runs(args.experiment_id)
    for run in runs:
        if run.status == "failed":
            print(run.id)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    repository = Repository(SQLiteDatabase(args.db))
    runs = repository.list_runs(args.experiment_id)
    print(f"Experiment: {args.experiment_id}")
    print(f"Runs: {len(runs)}")
    print(_summarize_runs(repository, runs))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "failures":
        return cmd_failures(args)
    if args.command == "report":
        return cmd_report(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
