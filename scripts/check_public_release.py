"""Small release allowlist and accidental-data/secret check; no network calls.

This is a heuristic publication check, not a comprehensive security audit.
When Git is initialized, inspect tracked paths (including the staging index).
Otherwise inspect the candidate source tree, excluding local build metadata.
Never print matched secret contents.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {"README.md", "README.zh-CN.md", "LICENSE", "NOTICE.md", ".gitignore", "pyproject.toml"}
TOP_DIRS = {"src", "tests", "examples", "scripts", "docs", ".github"}
EXTENSIONS = {".py", ".md", ".toml", ".yml", ".yaml", ".dot", ".svg", ".png"}
FORBIDDEN_PARTS = {"data", "datasets", "private", "models", "weights", "checkpoints",
                   "artifacts", "runs", "outputs", ".ssh", "node_modules", ".venv"}
TOKEN_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "model-service token": re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
    "cloud access key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "personal home path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "private LAN address": re.compile(r"\b(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3})\b"),
}


def candidate_paths():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True)
    if result.returncode == 0 and result.stdout:
        return [Path(p.decode()) for p in result.stdout.split(b"\0") if p]
    return [p.relative_to(ROOT) for p in ROOT.rglob("*") if p.is_file()
            and not any(part in {".git", ".venv", "__pycache__", "build", "dist"}
                        or part.endswith(".egg-info") for part in p.relative_to(ROOT).parts)]


def main():
    errors, total = [], 0
    paths = candidate_paths()
    for relative in sorted(paths):
        path = ROOT / relative
        if path.is_symlink():
            errors.append((str(relative), "symlink not allowed"))
            continue
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            errors.append((str(relative), "excluded artifact directory"))
        if str(relative) not in TOP_FILES and (
            relative.parts[0] not in TOP_DIRS or relative.suffix not in EXTENSIONS
        ):
            errors.append((str(relative), "outside publication allowlist"))
        size = path.stat().st_size
        total += size
        if size > 2_000_000:
            errors.append((str(relative), "unexpectedly large public artifact"))
        if relative.suffix == ".png":
            if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                errors.append((str(relative), "invalid PNG"))
            continue
        text = path.read_text(encoding="utf-8")
        for label, pattern in TOKEN_PATTERNS.items():
            if pattern.search(text):
                errors.append((str(relative), label))
    if errors:
        for name, reason in errors:
            print(f"FAIL {name}: {reason}", file=sys.stderr)
        return 1
    print(f"PASS: {len(paths)} public files, {total:,} bytes; no excluded artifact or flagged pattern.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
