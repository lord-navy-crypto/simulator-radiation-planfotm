from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting.scan_overview import (
    METRICS, build_observable_figure, representative_indices,
    validate_scan_figure_axis, scan_point_context,
)

x = np.geomspace(1.2, 2e4, 17)
data = {"scan_x": x}
for j, spec in enumerate(METRICS):
    # Keep log quantities positive and linear quantities finite.
    if spec.y_scale == "log":
        data[spec.key] = np.geomspace(1e-9*(j+1), 1e2*(j+1), len(x))
    else:
        data[spec.key] = np.linspace(-0.5, 0.5, len(x)) + j
    if spec.reference_key:
        data[spec.reference_key] = np.asarray(data[spec.key]) * 1.01

df = pd.DataFrame(data)
rep = representative_indices(df, n=4, strategy="Feature-aware")

for spec in METRICS:
    fig = build_observable_figure(df, "gamma", spec, rep)
    assert fig is not None, spec.key
    for tr in fig.data:
        tx = np.asarray(tr.x, dtype=float)
        assert len(tx) > 0
        for v in tx:
            assert np.any(np.isclose(v, x, rtol=1e-12, atol=0.0)), (spec.key, v)
    assert fig.layout.xaxis.title.text == "Lorentz factor γ"

print("V9.3 SCAN AXIS INVARIANTS TEST PASSED")

beta_points = np.linspace(0.60, 0.95, 50)
velocity_df = pd.DataFrame({
    "scan_x": beta_points,
    "gamma_avg": 1.0 / np.sqrt(1.0 - beta_points**2),
})
velocity_fig = build_observable_figure(
    velocity_df, "velocity", next(s for s in METRICS if s.key == "gamma_avg")
)
assert velocity_fig.layout.xaxis.title.text == "Electron speed β = v/c"
assert velocity_fig.layout.xaxis.type != "log"
assert np.allclose(np.asarray(velocity_fig.data[0].x, dtype=float), beta_points)
assert not np.allclose(np.asarray(velocity_fig.data[0].x, dtype=float), velocity_df["gamma_avg"].to_numpy(float))
print("V9.4 VELOCITY AXIS VALUES TEST PASSED")

# The invariant is generic, not velocity-specific: whichever quantity is
# scanned must be the exact x-array for every primary scan-level trend.
scan_cases = {
    "gamma": np.geomspace(1.25, 6.0e4, 13),
    "velocity": np.linspace(0.60, 0.95, 13),
    "K": np.linspace(0.3, 1.2, 13),
    "N_periods": np.arange(5, 18, dtype=float),
    "observer_distance": np.geomspace(20.0, 200.0, 13),
    "angle": np.linspace(-2.0, 2.0, 13),
}
expected_titles = {
    "gamma": "Lorentz factor γ",
    "velocity": "Electron speed β = v/c",
    "K": "Undulator parameter K",
    "N_periods": "Number of periods N",
    "observer_distance": "Observer distance R (m)",
    "angle": "Observer θx (mrad)",
}
for selected_scan, selected_points in scan_cases.items():
    generic_df = pd.DataFrame({
        "scan_x": selected_points,
        "gamma_avg": np.linspace(2.0, 3.0, len(selected_points)),
    })
    generic_fig = build_observable_figure(
        generic_df, selected_scan, next(s for s in METRICS if s.key == "gamma_avg")
    )
    plotted_x = np.asarray(generic_fig.data[0].x, dtype=float)
    assert np.allclose(plotted_x, selected_points), selected_scan
    assert generic_fig.layout.xaxis.title.text == expected_titles[selected_scan]
print("V9.4 DYNAMIC SELECTED-SCAN AXIS TEST PASSED")

# Runtime protection must reject both a fake title-only repair and a z-array
# passed into a scan-level chart.
bad = build_observable_figure(
    velocity_df, "velocity", next(s for s in METRICS if s.key == "gamma_avg")
)
bad.data[0].x = np.linspace(0.0, 1.0, len(beta_points))
try:
    validate_scan_figure_axis(bad, velocity_df, "velocity")
except AssertionError:
    pass
else:
    raise AssertionError("Runtime scan-axis invariant accepted a non-scan x-array")
print("V10.2 RUNTIME SCAN-AXIS GUARD TEST PASSED")

speed_context = scan_point_context("velocity", 0.80)
assert np.isclose(speed_context["speed_m_per_s"], 0.80 * 299792458.0)
assert np.isclose(speed_context["gamma"], 1.0 / np.sqrt(1.0-0.80**2))
gamma_context = scan_point_context("gamma", 2.0)
assert np.isclose(gamma_context["beta_v_over_c"], np.sqrt(3.0)/2.0)
print("V10.3 FIXED-POINT SPEED CONTEXT TEST PASSED")
