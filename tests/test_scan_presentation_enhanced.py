from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting.scan_overview import (
    representative_indices,
    build_scan_figure,
    build_observable_figure,
    axis_title,
    METRIC_BY_KEY,
)


df = pd.DataFrame({
    "scan_x": np.geomspace(1.25, 6.0e4, 31),
    "f0": np.geomspace(1e9, 1e15, 31),
    "f_expected": np.geomspace(1.01e9, 1.01e15, 31),
    "P_larmor": np.geomspace(1e-18, 1e-6, 31),
    "P_schwinger": np.geomspace(1.1e-18, 1.1e-6, 31),
    "P_circ": np.linspace(-1.0, -0.85, 31),
    "P_lin": np.linspace(0.0, 0.2, 31),
})
idx = representative_indices(df, n=4, strategy="Coverage")
assert len(idx) == 4, idx
assert idx[0] == 0, idx
assert idx[-1] == len(df) - 1, idx
assert np.all(np.diff(idx) > 0), idx

idx_feature = representative_indices(df, n=4, strategy="Feature-aware")
assert len(idx_feature) == 4
assert idx_feature[0] == 0 and idx_feature[-1] == len(df)-1

# Core architecture invariant: every scan-level trace uses scan_x as x.
fig = build_scan_figure(
    df,
    "gamma",
    [("f0", "Simulation"), ("f_expected", "Theory")],
    "Fundamental frequency",
    "Frequency (Hz)",
    idx,
    log_y=True,
)
assert fig is not None
scan_values = df["scan_x"].to_numpy(float)
for trace in fig.data:
    tx = np.asarray(trace.x, dtype=float)
    assert all(np.any(np.isclose(v, scan_values, rtol=1e-13, atol=0.0)) for v in tx), tx
assert fig.layout.xaxis.title.text == axis_title("gamma")
assert fig.layout.xaxis.type == "log"

# A primary chart is ONE dependent observable; only same-observable theory overlay
# plus representative markers are allowed.
ffig = build_observable_figure(df, "gamma", METRIC_BY_KEY["f0"], idx)
assert ffig is not None
names = [str(t.name) for t in ffig.data]
assert "Simulation" in names
assert "Theory" in names
assert "Representative points" in names
assert "P_larmor" not in names
for trace in ffig.data:
    tx = np.asarray(trace.x, dtype=float)
    assert all(np.any(np.isclose(v, scan_values, rtol=1e-13, atol=0.0)) for v in tx)

src = (ROOT / "undulator_v11_radia_gui_v9.py").read_text(encoding="utf-8")
assert "render_scan_overview" in src
assert "render_representative_comparison" in src
assert "Run selected scan-point deep analysis" in src
assert "Run all representative full cases" in src
assert "Deep-analysis scan point" in src
assert '"gamma"' in src and '"K"' in src and '"N_periods"' in src
assert '"observer_distance"' in src and '"angle"' in src
assert '"Scalar scan points across selected range", 7, 81, 50, 1' in src
assert '60000.0, 60000.0, key="gmax"' in src
assert '["gamma", "velocity", "K", "N_periods", "observer_distance", "angle"]' in src
assert 'rr["scan_x"] = float(beta)' in src
assert 'gamma_for_solver = float(v11.gamma_from_beta(float(beta)))' in src
assert "Whichever quantity is selected here becomes the actual x-axis" in src
assert "Run scan across {axis_title(scan_preset)}" in src
assert "Show optional single-point z-axis result suite" in src
assert "Show single-device z-axis field preview" in src
assert "FIXED OPERATING POINT" in src
assert "Error-strength response scan" in src
assert "Maximum strength relative to nominal" in src
assert "run_error_strength_scan" in src

overview = (ROOT / "reporting" / "scan_overview.py").read_text(encoding="utf-8")
assert "FULL SCAN TREND" in overview
assert "FIXED SCAN POINT" in overview
assert "scan_point_context_text" in overview

full = (ROOT / "reporting" / "full_results.py").read_text(encoding="utf-8")
assert "Comprehensive single-point result output" in full
assert "def _secondary_plot" in full
# Cross-variable loci are allowed only in the single-point report and must be
# explicitly secondary diagnostics.
for title in [
    "Transverse magnetic-field locus",
    "Transverse orbit locus",
    "Horizontal phase-space projection",
    "Vertical phase-space projection",
    "Time-domain polarization locus",
    "Normalized Q–U polarization plane",
    "Gaunt factor versus χe",
]:
    assert f'_secondary_plot(st, fig, "{title}"' in full, title

# Physics backends must remain external to this presentation-only change.
for name in [
    "undulator_v11_radia_integrated_v9.py",
    "v11_radia_backend_v8.py",
    "v11_field_quality_v6.py",
    "solver/pipeline.py",
    "devices/factory.py",
    "errors/model.py",
    "calibration/target_b0.py",
]:
    assert (ROOT / name).exists(), name

print("SCAN-CENTRIC PRESENTATION ENHANCEMENT TEST PASSED")
