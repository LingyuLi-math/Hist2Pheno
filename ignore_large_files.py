#!/usr/bin/env python3
"""Ignore and untrack repository files larger than GitHub's 100 MiB limit."""

from __future__ import annotations

import subprocess
from pathlib import Path

LIMIT = 100 * 1024 * 1024
BEGIN = "# BEGIN AUTO-IGNORED FILES > 100 MiB (managed by ignore_large_files.py)"
END = "# END AUTO-IGNORED FILES"
ROOT = Path(__file__).resolve().parent
GITIGNORE = ROOT / ".gitignore"


def git_paths(*args: str) -> set[str]:
    result = subprocess.run(
        ["git", *args, "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {
        path.decode("utf-8", errors="surrogateescape")
        for path in result.stdout.split(b"\0")
        if path
    }


def generated_paths(lines: list[str]) -> set[str]:
    try:
        start = lines.index(BEGIN) + 1
        end = lines.index(END, start)
    except ValueError as error:
        raise SystemExit("Generated large-file section is missing from .gitignore") from error
    return {line for line in lines[start:end] if line and not line.startswith("#")}


def main() -> None:
    lines = GITIGNORE.read_text(encoding="utf-8").splitlines()
    candidates = git_paths("ls-files", "--cached", "--others", "--exclude-standard")
    candidates.update(generated_paths(lines))

    oversized = sorted(
        path
        for path in candidates
        if (file_path := ROOT / path).is_file() and file_path.stat().st_size > LIMIT
    )

    start = lines.index(BEGIN) + 1
    end = lines.index(END, start)
    updated = [*lines[:start], *oversized, *lines[end:]]
    GITIGNORE.write_text("\n".join(updated) + "\n", encoding="utf-8")

    tracked = git_paths("ls-files")
    to_untrack = sorted(set(oversized) & tracked)
    if to_untrack:
        subprocess.run(
            ["git", "rm", "--cached", "--ignore-unmatch", "--", *to_untrack],
            cwd=ROOT,
            check=True,
        )

    for path in oversized:
        size_mib = (ROOT / path).stat().st_size / (1024 * 1024)
        print(f"Ignored {size_mib:.2f} MiB: {path}")
    print(f"Found {len(oversized)} file(s) over 100 MiB.")


if __name__ == "__main__":
    main()
