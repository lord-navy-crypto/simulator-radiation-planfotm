#!/bin/zsh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

export RADIA_PYTHONPATH="${RADIA_PYTHONPATH:-$HOME/Desktop/Radia-master/cpp/gcc}"
export PYTHONPATH="$DIR:$RADIA_PYTHONPATH:$PYTHONPATH"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" - <<'PY2'
import importlib.util, subprocess, sys
mods={"numpy":"numpy","scipy":"scipy","pandas":"pandas","matplotlib":"matplotlib","streamlit":"streamlit","plotly":"plotly"}
missing=[pkg for mod,pkg in mods.items() if importlib.util.find_spec(mod) is None]
if missing:
    print("Installing missing packages:",", ".join(missing))
    subprocess.check_call([sys.executable,"-m","pip","install","--user",*missing])
PY2

echo "Starting V11 RADIA Radiation Studio v9..."
echo "RADIA path: $RADIA_PYTHONPATH"
exec "$PYTHON_BIN" -m streamlit run undulator_v11_radia_gui_v9.py
