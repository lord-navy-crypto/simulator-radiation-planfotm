#!/bin/zsh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
exec "$DIR/START_HERE_V11_RADIA_v9.command"
