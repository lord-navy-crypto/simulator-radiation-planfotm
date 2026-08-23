from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = (ROOT / "reporting" / "full_results.py").read_text(encoding="utf-8")
gui = (ROOT / "undulator_v11_radia_gui_v9.py").read_text(encoding="utf-8")

assert "PLOT_HEIGHT = 650" in report
assert "PLOT_HEIGHT_3D = 760" in report
assert "TABLE_HEIGHT = 620" in report
assert 'format="%.10e"' in report
assert "scrollZoom" in report
assert "Exact numerical summary" in report
assert "Horizontal phase-space projection" in report
assert "Vertical phase-space projection" in report
assert "Retarded source time mapped to observer time" in report
assert "Spectrum on photon-energy axis" in report
assert "Spatial magnetic-field spectrum" in report
assert "Gaunt factor versus χe" in report
assert "3-D electron trajectory — physical coordinates" in report
assert "3-D electron orbit — transverse motion magnified" in report
assert "Visualization scaling notice" in report
assert "render_fixed_point_orbit_dashboard" in report
assert "Orbit-centred transverse radius" in report
assert "Distance from design axis" in report
assert "entrance" in report and "exit" in report
scan_report = (ROOT / "reporting" / "scan_overview.py").read_text(encoding="utf-8")
assert "3-D electron trajectories — fixed representative scan points" in scan_report
assert "rep_traj_radius_" in scan_report
assert "Orbit-centred transverse radius" in scan_report
assert "show_single_point_results = st.toggle" not in gui
assert 'if "full_result" in st.session_state:' in gui
assert report.count("_plot(st,") >= 30
assert "max-width: 1920px" in gui
assert "st.columns(2, gap=\"large\")" in report
assert "st.columns(4)" not in report

print("FULL VISUALIZATION LAYOUT TEST PASSED")
