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
assert report.count("_plot(st,") >= 30
assert "max-width: 1920px" in gui
assert "st.columns(2, gap=\"large\")" in report
assert "st.columns(4)" not in report

print("FULL VISUALIZATION LAYOUT TEST PASSED")
