#!/bin/zsh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "1/3 GitHub preflight"
python3 scripts/preflight_github.py

echo
echo "2/3 Python compile"
python3 -m compileall -q .

echo
echo "3/3 Regression suite"
python3 scripts/run_tests.py

echo
echo "READY FOR GITHUB"
echo "Next: open docs/GITHUB_UPLOAD.md"
