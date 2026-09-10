from __future__ import annotations

import os
from pathlib import Path


def _candidate_paths() -> list[Path]:
    cwd = Path.cwd()
    paths: list[Path] = []
    seen: set[Path] = set()
    for base in [cwd, *cwd.parents]:
        env_path = base / ".env"
        if env_path not in seen:
            paths.append(env_path)
            seen.add(env_path)
    return paths


def load_env(path: str | os.PathLike[str] | None = None) -> None:
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend(_candidate_paths())

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
            return
        except Exception:
            for line in normalized.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
            return
