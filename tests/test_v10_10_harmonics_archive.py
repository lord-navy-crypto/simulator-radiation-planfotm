from pathlib import Path
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-radia-v1010")

backend_source = (ROOT / "undulator_v11_radia_integrated_v9.py").read_text(encoding="utf-8")
gui_source = (ROOT / "undulator_v11_radia_gui_v9.py").read_text(encoding="utf-8")
studio_source = (ROOT / "magnet_studio" / "app" / "studio.py").read_text(encoding="utf-8")
helical_source = (ROOT / "magnet_studio" / "devices" / "helical.py").read_text(encoding="utf-8")

assert "analytic_h3=0.0" in backend_source
assert "analytic_h5=0.0" in backend_source
assert "harmonic_velocity_match_factor" in backend_source
assert "Third-harmonic field coefficient H3/H1" in gui_source
assert "Fifth-harmonic field coefficient H5/H1" in gui_source
assert "Saved Stage-1 model used by Stage 2" in gui_source
assert "magnet_model_history" in studio_source
assert "Download selected model transfer ZIP" in studio_source
assert "saved_models" in studio_source
assert 'size_xy = [height,width] if row in ("right","left") else [width,height]' in helical_source
assert "handedness*2*math.pi" in helical_source
assert "must be <= magnetic gap" in helical_source

import sys
sys.path.insert(0, str(ROOT))
from scipy.integrate import solve_ivp
import undulator_v11_radia_integrated_v9 as v11

device = v11.make_default_undulator(
    realistic=False,
    preset="helical",
    field_model="analytic",
    n_periods=20,
    analytic_h3=0.03,
    analytic_h5=0.005,
)
gamma = 100.0
state0, metadata = v11.make_initial_state_device(gamma, device)
span = v11.simulation_span_for_device(gamma, device, n_periods=20)
solution = solve_ivp(
    v11.rhs_lorentz,
    span,
    state0,
    args=(device, v11.me, -v11.qe),
    rtol=1e-10,
    atol=1e-12,
)
assert solution.success
position = solution.y[:3].T
return_error = np.linalg.norm(position[-1, :2] - position[0, :2])
assert return_error < 1e-9, return_error
assert abs(metadata["harmonic_velocity_match_factor"] - 1.011) < 1e-12

print("V10.10 HARMONIC MATCHING AND MODEL ARCHIVE TEST PASSED")
