#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "tests"
tests = sorted(TEST_DIR.glob("test_*.py"))

if not tests:
    raise SystemExit("No test scripts found.")

failed = []
for test in tests:
    print(f"\n=== {test.name} ===", flush=True)
    result = subprocess.run(
        [sys.executable, str(test)],
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": f"{ROOT}:{TEST_DIR}"},
    )
    if result.returncode:
        failed.append((test.name, result.returncode))

if failed:
    print("\nFAILED:")
    for name, code in failed:
        print(f"  {name}: exit {code}")
    raise SystemExit(1)

print(f"\nALL {len(tests)} REGRESSION SCRIPTS PASSED")
