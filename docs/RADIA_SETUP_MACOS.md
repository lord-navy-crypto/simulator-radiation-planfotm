# RADIA setup on macOS

This repository does not redistribute the official RADIA source or binary.

Official upstream:

- https://github.com/ochubar/Radia
- https://www.esrf.fr/home/Accelerators/instrumentation--equipment/Software/Radia/Documentation.html

The launcher uses:

```bash
export RADIA_PYTHONPATH="${RADIA_PYTHONPATH:-$HOME/Desktop/Radia-master/cpp/gcc}"
```

Before launching the GUI, verify:

```bash
PYTHONPATH="$HOME/Desktop/Radia-master/cpp/gcc" python3 -c \
'import radia; print(radia.__file__)'
```

If your compiled module is elsewhere:

```bash
export RADIA_PYTHONPATH="/your/path/to/radia/module"
./START_HERE_V11_RADIA_v9.command
```

Do not commit a compiled local RADIA binary, personal build directory, or
machine-specific credentials into this repository unless you have separately
reviewed the applicable distribution/license requirements.
