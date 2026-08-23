from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

expected = [
    ROOT / "unified_entry.py",
    ROOT / "magnet_studio" / "app" / "studio.py",
    ROOT / "undulator_v11_radia_gui_v9.py",
]
for path in expected:
    assert path.is_file(), path

root_entry = (ROOT / "unified_entry.py").read_text(encoding="utf-8")
app_entry = (ROOT / "app" / "unified_entry.py").read_text(encoding="utf-8")
launcher = (ROOT / "START_HERE_V11_RADIA_v9.command").read_text(encoding="utf-8")

for source in (root_entry, app_entry):
    assert '"magnet_studio" / "app" / "studio.py"' in source
    assert '"undulator_v11_radia_gui_v9.py"' in source
    assert "page_path.is_file()" in source
    assert "st.Page(" in source

assert "streamlit run unified_entry.py" in launcher
assert "streamlit run app/unified_entry.py" not in launcher

print("UNIFIED ENTRYPOINT PAGE PATH TEST PASSED")
