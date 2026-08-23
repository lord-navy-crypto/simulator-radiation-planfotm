from __future__ import annotations

from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MAGNET_PAGE = PROJECT_ROOT / "magnet_studio" / "app" / "studio.py"
SCAN_PAGE = PROJECT_ROOT / "undulator_v11_radia_gui_v9.py"

for page_path in (MAGNET_PAGE, SCAN_PAGE):
    if not page_path.is_file():
        raise FileNotFoundError(f"Unified RADIA page is missing: {page_path}")

st.set_page_config(
    page_title="RADIA Unified Magnet → Radiation Studio",
    page_icon="⚛️",
    layout="wide",
)

st.sidebar.markdown("## Unified workflow")
bridge = st.session_state.get("magnet_scan_bridge")
if isinstance(bridge, dict) and bridge.get("field_map_csv"):
    st.sidebar.success("Stage 1 field ready → Stage 2 unlocked")
else:
    st.sidebar.info("Begin with Stage 1, or import a completed transfer package there.")

pages = [
    st.Page(
        MAGNET_PAGE,
        title="1 · Magnet design & field",
        icon="🧲",
        default=True,
    ),
    st.Page(
        SCAN_PAGE,
        title="2 · Trajectory & radiation scan",
        icon="📈",
    ),
]
st.navigation(pages, position="sidebar").run()
