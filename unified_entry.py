from __future__ import annotations

from pathlib import Path
import runpy

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MAGNET_WORKSPACE = PROJECT_ROOT / "magnet_studio" / "app" / "studio.py"
RADIATION_WORKSPACE = PROJECT_ROOT / "undulator_v11_radia_gui_v9.py"

for workspace_path in (MAGNET_WORKSPACE, RADIATION_WORKSPACE):
    if not workspace_path.is_file():
        raise FileNotFoundError(f"Unified RADIA workspace is missing: {workspace_path}")

st.set_page_config(
    page_title="RADIA Unified Magnet → Radiation Studio",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("## Unified Engineering Workspace")
st.sidebar.caption("Setup → Field model → Trajectory → Radiation → Analysis → Verification → Export")

bridge = st.session_state.get("magnet_scan_bridge")
if isinstance(bridge, dict) and bridge.get("field_map_csv"):
    st.sidebar.success("Realized Stage-1 field is connected to the radiation workflow.")
else:
    st.sidebar.info("Build or import a magnetic field first; analytic/RADIA alternatives remain available in the radiation workspace.")

workspace = st.sidebar.radio(
    "Workspace",
    ["1 · Magnet & Field", "2 · Trajectory & Radiation"],
    key="unified_radia_workspace",
)

st.sidebar.divider()
st.sidebar.markdown("### Combined capability chain")
st.sidebar.markdown(
    """
- Magnet geometry & presets
- RADIA field solve / analytic fallback
- 1D / 2D / 3D field inspection
- Manufacturing-error model
- Field integrals, harmonics & K
- Electron trajectory & phase
- Dynamic parameter scans
- Radiation observer calculation
- Deep operating-point analysis
- Error-strength studies
- Physics compliance / verification
- Reproducibility exports
"""
)

if workspace.startswith("1"):
    runpy.run_path(str(MAGNET_WORKSPACE), run_name="__main__")
else:
    runpy.run_path(str(RADIATION_WORKSPACE), run_name="__main__")
