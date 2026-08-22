#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

required = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".gitignore",
    ".github/workflows/ci.yml",
    "requirements.txt",
    "START_HERE_V11_RADIA_v9.command",
]
missing = [p for p in required if not (ROOT / p).exists()]
if missing:
    print("Missing required repository files:")
    for p in missing:
        print(" -", p)
    raise SystemExit(1)

blocked_names = {
    ".env", ".streamlit/secrets.toml", "id_rsa", "id_rsa.pub"
}
blocked_suffixes = {".pem", ".key"}
secret_patterns = [
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]

problems = []
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(ROOT).as_posix()
    if ".git/" in rel or "__pycache__/" in rel:
        continue
    if rel in blocked_names or path.suffix.lower() in blocked_suffixes:
        problems.append(f"sensitive filename: {rel}")
    if path.stat().st_size > 90 * 1024 * 1024:
        problems.append(f"large file >90 MiB: {rel}")
    if path.stat().st_size <= 5 * 1024 * 1024:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pat in secret_patterns:
            if pat.search(text):
                problems.append(f"possible secret pattern: {rel}")
                break

if problems:
    print("Preflight FAILED:")
    for p in problems:
        print(" -", p)
    raise SystemExit(1)

print("GitHub preflight: PASS")
print("No obvious secret files/patterns detected.")
print("No files exceed 90 MiB.")
print("Required repository metadata present.")
