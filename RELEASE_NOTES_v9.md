# V11 RADIA Radiation Studio v9.0.0

V9 is the current research baseline.

Highlights:

- RADIA-generated 3-D fields with manufacturing-error support.
- Ideal-geometry B0 calibration with explicit convergence gate.
- Full fringe-field trajectory tracking.
- Generalized device resonance and theoretical-power handling.
- Corrected causal retarded-time Lienard-Wiechert solver.
- On-axis/off-axis spectral analysis.
- 1-D and 2-D angular radiation analysis.
- Radiative-fluence weighting in J/m².
- Explicit NaN/failure masks for failed angular-map samples.
- Correct scan-helper result schema using `P_larmor`.
- RADIA CSV import with arbitrary magnetic period.
- NumPy-safe JSON export.
- Regression tests covering physics, geometry, runtime, and API behavior.

Scientific scope: single-electron research simulation. Facility-specific device
certification and collective-beam effects are outside this release.
