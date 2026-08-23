from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting.live_progress import LiveProgressTable


class Slot:
    def __init__(self):
        self.last = None
    def dataframe(self, frame, **kwargs):
        self.last = frame.copy()


class Bar:
    def __init__(self):
        self.value = 0.0
        self.text = ""
    def progress(self, value, text=""):
        self.value = float(value)
        self.text = str(text)


class FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.slot = Slot()
        self.bar = Bar()
    def markdown(self, *args, **kwargs):
        pass
    def progress(self, value, text=""):
        self.bar.progress(value, text)
        return self.bar
    def empty(self):
        return self.slot


st = FakeStreamlit()
tracker = LiveProgressTable(
    st,
    [("point 1", "K=0.3"), ("point 2", "K=0.4"), ("point 3", "K=0.5")],
    title="Test scan",
    session_key="saved",
)
tracker.start(0, "solving")
assert tracker.rows[0]["Status"] == "Running"
tracker.complete(0, "done")
tracker.start(1, "solving")
tracker.fail(1, "intentional test failure")
tracker.skip(2, "not needed")
tracker.finish("finished")

assert [row["Status"] for row in tracker.rows] == ["Complete", "Failed", "Skipped"]
assert st.bar.value == 1.0
assert len(st.session_state["saved"]) == 3
assert st.slot.last is not None
assert list(st.slot.last["Input / stage"]) == ["K=0.3", "K=0.4", "K=0.5"]

gui = (ROOT / "undulator_v11_radia_gui_v9.py").read_text(encoding="utf-8")
magnet = (ROOT / "magnet_studio" / "app" / "studio.py").read_text(encoding="utf-8")
assert "Live calculation progress —" in gui
assert "last_primary_scan_progress" in gui
assert "last_representative_progress" in gui
assert "last_single_point_progress" in gui
assert "last_magnet_build_progress" in magnet

print("LIVE CALCULATION PROGRESS TABLE TEST PASSED")
