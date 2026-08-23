from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integration.bridge import load_transfer_package
from magnet_studio.analysis.metrics import analyze
from magnet_studio.export.exporters import research_package_bytes


def run():
    entry = (ROOT / "app" / "unified_entry.py").read_text(encoding="utf-8")
    stage1 = (ROOT / "magnet_studio" / "app" / "studio.py").read_text(encoding="utf-8")
    stage2 = (ROOT / "undulator_v11_radia_gui_v9.py").read_text(encoding="utf-8")
    assert "st.navigation" in entry and "1 · Magnet design & field" in entry
    assert "2 · Trajectory & radiation scan" in entry
    assert 'st.session_state["magnet_scan_bridge"]' in stage1
    assert 'st.session_state["magnet_completed_visuals"]' in stage1
    assert "def render_completed_entity_gallery" in stage1
    assert 'if not run and st.session_state.get("magnet_completed_visuals") is not None' in stage1
    assert "Restored completed magnet entity & field gallery" in stage1
    for restored_view in [
        "restored_stage1_geometry", "restored_stage1_axis_field",
        "restored_stage1_2d_field", "restored_stage1_3d_field",
        "restored_stage1_trajectory", "restored_stage1_electron_phase",
    ]:
        assert restored_view in stage1
    assert "Use imported field in Stage 2" in stage1
    assert "Magnet Studio Stage-1 realized field" in stage2
    assert 'field_model in {"radia_csv", "studio_bridge"}' in stage2

    x = np.array([-1.0, 1.0])
    y = np.array([-1.0, 1.0])
    z = np.linspace(-25.0, 25.0, 5)
    field3 = np.zeros((len(z), len(y), len(x), 3))
    for iz, zz in enumerate(z):
        field3[iz, :, :, 1] = 0.15 * np.sin(2 * np.pi * zz / 50.0)
    axis_z = np.linspace(-100.0, 100.0, 401)
    axis_B = np.column_stack((np.zeros_like(axis_z), 0.15*np.sin(2*np.pi*axis_z/50.0), np.zeros_like(axis_z)))
    metrics = analyze(axis_z, axis_B, 50.0, 3.0)
    package = research_package_bytes(
        {"device": "Planar", "period_mm": 50.0, "periods": 20},
        metrics, axis_z, axis_B, grid3=(x, y, z), field3=field3, blocks=[],
    )
    bridge = load_transfer_package(package)
    assert bridge["parameters"]["device"] == "Planar"
    assert bridge["parameters"]["period_mm"] == 50.0
    assert bridge["field_map_csv"].startswith(b"x_m,y_m,z_m,Bx_T,By_T,Bz_T")
    print("UNIFIED TWO-STAGE MAGNET-TO-SCAN WORKFLOW TEST PASSED")


if __name__ == "__main__":
    run()
