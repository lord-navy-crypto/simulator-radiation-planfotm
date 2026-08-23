# Changelog

## v10.10 — configurable harmonics and multi-model Stage-1 archive

- Added explicit analytic-field H3/H1 and H5/H1 controls with zero defaults;
  RADIA/imported-map harmonics remain derived from the realized magnetic field.
- Corrected the matched analytic helical-orbit initial velocity and position to
  include each configured harmonic's field-integral contribution.
- Synchronized the fused Stage-1 helical geometry: side-bank dimensions,
  handedness, and rectangular-bank overlap validation are now consistent.
- Every successful Stage-1 build is saved as an individual timestamped transfer
  ZIP under `saved_models/` and retained as a selectable model archive entry.
- Stage 2 now selects the exact saved Stage-1 model used for trajectory,
  radiation, and scanning; saved files are rediscovered after app restart.
- The magnified 3-D orbit subtracts a clearly labelled best-fit centreline while
  the physical-coordinate view retains real offset and drift.

## v10.9 — visible fixed-point 3-D particle orbit and radius

- Removed the default-off main-result switch that hid the full fixed-point
  result suite even after a successful calculation.
- Moved the interactive physical-scale and transverse-magnified 3-D particle
  trajectories to the top of every full fixed-point report.
- Added radius-versus-z diagnostics at the main point, any selected scan point,
  and every representative full-case point.
- Separately reports orbit-centred radius and distance from the design axis so
  beam/orbit offset is not mistaken for oscillation amplitude.
- Retains the scan rule: scan-level trend charts use the selected scan quantity
  as x; z is used only inside a clearly labelled fixed-point trajectory report.

## v10.8 — persistent Stage-1 completed-entity gallery

- Fixed the completed Magnet Studio figures disappearing after navigating to
  Stage 2 and returning to Stage 1.
- Persists the raw solved geometry, on-axis field, optional 2D/3D maps,
  trajectory and electron-phase data in Streamlit session state.
- Rebuilds a restored gallery with 3D magnet entities first, followed by field,
  trajectory and phase views; labels it as the last successful solved result so
  unsolved sidebar edits cannot be mistaken for recomputed geometry.

## v10.7 — enhanced 3-D electron-orbit visualization

- Upgraded the full-result 3-D trajectory to consistent physical-coordinate
  units with entrance/exit markers, path-progress colouring and time-aware hover.
- Added a second explicitly labelled transverse-magnified 3-D orbit so the
  micrometre-scale oscillation remains visible without pretending it is true scale.
- Added fixed-point 3-D electron trajectories to the representative-case suite;
  the full-scan independent-axis rule remains unchanged.

## v10.6 — physics compliance and spectrum correction

- Added a per-result PASS / WARNING / FAIL physics-compliance gate covering
  relativistic state, resonance, energy accounting, Stokes realizability,
  sampling, radiation-zone applicability, aperture loss, quantum regime and
  available RADIA calibration metadata.
- Corrected orthogonal-polarization spectral combination to use
  `|S1|² + |S2|²`, eliminating an artificial cross term.
- Declared the observer solver accurately as the retarded `1/R` acceleration
  field and exposed the omitted `1/R²` near/velocity field as a model boundary.
- Declared the Stokes basis and circular-polarization sign convention.
- Updated non-exact SI constants to 2022 CODATA values.
- Added the v10.6 physics audit and new regression tests.

## v10.5 — live calculation progress tables

- Added a reusable live progress-table component with task/input, status,
  overall percentage, elapsed time, and detail columns.
- Stage 1 now reports RADIA initialization, optional B0 calibration, geometry
  generation, magnetic solve, on-axis sampling, 2D/3D maps, and Stage-2 publish.
- Primary scans now report every requested scan point separately, including the
  exact selected scan value and whether that point completed or failed.
- Single-point V11 runs and representative full-case batches now expose their
  own live stage/point tables.
- The last completed progress table remains available in a collapsed section
  after Streamlit reruns.

## v10.4 — unified Streamlit entrypoint path repair

- Moved the normal Streamlit entrypoint to the project root so page paths are
  naturally anchored beside `magnet_studio/` and the V11 scan page.
- Changed both entrypoints to pass verified absolute `Path` objects to
  `st.Page`.
- Updated the macOS launcher to run `unified_entry.py` from the root.
- Added a regression test that resolves both navigation targets and rejects the
  former broken `app/magnet_studio/...` path layout.

## v10.3 — full-scan versus fixed-point visualization scope

- Generalized the axis rule beyond x/z: every full-scan observable uses the
  selected scan quantity, while natural local axes are restricted to clearly
  labelled fixed-point analyses.
- Added `FULL SCAN TREND` titles to scan-level figures.
- Added `FIXED SCAN POINT` context banners and titles to selected and
  representative deep analyses.
- Velocity/gamma fixed points now show β, m/s, and γ together.
- Borrowed the useful grouping idea from the older visualization while keeping
  the current V11/RADIA algorithms and scan-aware plotting backend.

## v10.2 — selected scan quantity is visibly and numerically the scan x-axis

- Made the parameter scan the primary research output and labelled its run
  button with the selected independent quantity.
- Hid optional single-device and single-operating-point z diagnostics by
  default so they cannot be mistaken for cross-scan results.
- Added an exact scan-axis point table; velocity scans also show m/s and the
  solver-derived Lorentz factor beside β=v/c.
- Added a runtime invariant that checks both the visible axis title and every
  plotted numeric x value against `scan_x`.

## 10.0.0 — Unified Magnet-to-Radiation Workflow

- Fused Magnet Studio 3.1 and the dynamic-axis V11 scanner into one Streamlit application.
- Added Stage 1 magnet selection, preset import, RADIA solve, inspection and 3-D field generation.
- Added an in-session bridge that transfers the realized field and device metadata directly to Stage 2.
- Added validated Magnet Studio transfer-ZIP import for previously calculated fields.
- Namespaced Magnet Studio modules so its device/solver code cannot shadow the V11 scan backend.
- Added an end-to-end transfer-package-to-scan integration regression.

## 9.4.0 — Dynamic Selected-Scan Axis

- The selected scan quantity is always the independent x-axis of scan-level trends.
- Gamma, speed beta=v/c, K, period count, observer distance and angle each retain their actual requested scan-point arrays on x.
- Velocity scans store requested speed in scan_x and derive gamma only at the solver boundary.
- Changing the independent-variable selector clears stale plots from the previous axis.
- Regression checks actual x-arrays for every supported scan variable, not only labels.

## 9.3.0 — Scan-Centric Research Visualization

- Rebuilt primary visualization around the invariant `scan variable → one dependent observable`.
- Added scan-aware focused/comprehensive metric groups with exact units and meaning.
- Added 3–6 representative-row selection with Coverage and Feature-aware strategies; all representatives are existing computed scan rows.
- Added representative full-case suites with frequency-based spectra plus z-based trajectory and magnetic-field plots.
- Added exact selected-point metric interpretation tables.
- Moved RADIA manufacturing-error strengths into the error-analysis controls.
- Added one-error-at-a-time physical-strength sweeps and all-selected error-strength multiplier sweeps.
- Added regression tests that verify every primary scan trace uses the scan variable as x.
- Physics/numerical backend files remain byte-for-byte unchanged from v9.2/v9.

## 9.2.0 — Scan-First Visualization

- Enforced scan-variable-as-x-axis architecture for all scan-level trend figures.
- Added full deep analysis for any selected existing scan row across gamma, K, period-count, observer-distance, and angle scans.
- Reframed main-run and selected-point outputs as single-case diagnostics with natural independent axes (z, observer time, frequency, photon energy).
- Collapsed cross-variable loci/phase-space plots into secondary diagnostics so they cannot be mistaken for scan trends.
- Removed mixed-unit `Quantity vs Value` bar charts in favor of exact tables.
- Physics/numerical backend files remain byte-for-byte unchanged from v9.1/v9.


## v9.1.0 — Presentation Enhanced

- Added full-scan research overview and representative-case highlighting.
- Gamma scan defaults to the full 1.25–60000 research range and 31 points.
- Added optional four-case deep comparison using the current V11/RADIA full solver.
- Added complete high-precision scan table and CSV exports.
- Physics/numerical backend files intentionally unchanged.

## 9.0.1 — Full Results UI

- Added comprehensive scrollable result visualization with more than a dozen plots.
- Added complete numerical tables for trajectory, observer field, spectrum, quantum arrays and scalar diagnostics.
- Added one-click ZIP export of the complete numerical result payload.
- Physics/numerical backend remains byte-for-byte unchanged from V9.


## 9.0.0 — 2026-08-22

- Finalized radiative-fluence angular weighting.
- Added explicit invalid-pixel masks/failure metadata for 2-D angular maps.
- Corrected legacy public scan helpers to the current `P_larmor` result schema.
- Retained V8 off-axis timing/resonance and non-overlapping geometry fixes.
- Retained V7 causal retarded-time, CSV-period, B0-convergence, and caching
  corrections.
- Added GitHub repository metadata, CI, contribution templates, publication
  documentation, and third-party notices.
