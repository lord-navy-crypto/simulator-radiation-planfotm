from __future__ import annotations

import time
from typing import Iterable

import pandas as pd


class LiveProgressTable:
    """A Streamlit progress bar plus a live, persistent task table."""

    def __init__(self, st, tasks: Iterable[tuple[str, str]], title: str, session_key: str | None = None):
        self.st = st
        self.title = str(title)
        self.session_key = session_key
        self.started_at = time.perf_counter()
        self.current_index: int | None = None
        self.rows = [
            {
                "#": i + 1,
                "Task": str(task),
                "Input / stage": str(value),
                "Status": "Pending",
                "Overall": "0.0%",
                "Elapsed (s)": 0.0,
                "Detail": "Waiting",
            }
            for i, (task, value) in enumerate(tasks)
        ]
        self.st.markdown(f"### {self.title}")
        self.bar = st.progress(0.0, text="Queued…")
        self.table = st.empty()
        self._render()

    def _overall(self) -> float:
        if not self.rows:
            return 1.0
        done = sum(row["Status"] in {"Complete", "Skipped", "Failed"} for row in self.rows)
        return done / len(self.rows)

    def _render(self) -> None:
        overall = self._overall()
        for row in self.rows:
            row["Overall"] = f"{100.0 * overall:.1f}%"
        frame = pd.DataFrame(self.rows)
        self.table.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            height=min(620, 78 + 35 * max(1, len(frame))),
        )
        if self.session_key:
            self.st.session_state[self.session_key] = [dict(row) for row in self.rows]

    def start(self, index: int, detail: str = "Running") -> None:
        self.current_index = int(index)
        row = self.rows[self.current_index]
        row["Status"] = "Running"
        row["Elapsed (s)"] = round(time.perf_counter() - self.started_at, 3)
        row["Detail"] = str(detail)
        self.bar.progress(self._overall(), text=f"Running {row['Task']} — {row['Input / stage']}")
        self._render()

    def complete(self, index: int, detail: str = "Complete") -> None:
        row = self.rows[int(index)]
        row["Status"] = "Complete"
        row["Elapsed (s)"] = round(time.perf_counter() - self.started_at, 3)
        row["Detail"] = str(detail)
        overall = self._overall()
        self.bar.progress(overall, text=f"{sum(r['Status']=='Complete' for r in self.rows)}/{len(self.rows)} complete")
        self._render()

    def skip(self, index: int, detail: str = "Not requested") -> None:
        row = self.rows[int(index)]
        row["Status"] = "Skipped"
        row["Elapsed (s)"] = round(time.perf_counter() - self.started_at, 3)
        row["Detail"] = str(detail)
        self.bar.progress(self._overall(), text=detail)
        self._render()

    def fail(self, index: int | None = None, detail: str = "Failed") -> None:
        idx = self.current_index if index is None else int(index)
        if idx is None:
            return
        row = self.rows[idx]
        row["Status"] = "Failed"
        row["Elapsed (s)"] = round(time.perf_counter() - self.started_at, 3)
        row["Detail"] = str(detail)
        self.bar.progress(self._overall(), text=f"Failed: {row['Task']}")
        self._render()

    def finish(self, text: str = "Calculation complete") -> None:
        self.bar.progress(1.0, text=text)
        self._render()


def render_saved_progress(st, session_key: str, title: str = "Last calculation progress") -> None:
    rows = st.session_state.get(session_key)
    if rows:
        with st.expander(title, expanded=False):
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
