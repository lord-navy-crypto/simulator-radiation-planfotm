# V11 RADIA Radiation Studio v9.3 — Scan-Centric Research Visualization

v9.3 is a presentation/experiment-organization release. The validated V9/RADIA physics and numerical backend files are unchanged.

## Core visualization rule

At scan level, the selected scan parameter is always the independent horizontal axis. Each primary chart contains one dependent physical observable. A theory curve can share a chart only when it predicts that same observable.

Examples:

- gamma -> frequency
- gamma -> wavelength
- gamma -> radiated power
- gamma -> photon energy
- K -> polarization
- number of periods -> linewidth
- observer distance -> observer-field peak
- angle -> redshift / polarization
- error strength -> frequency residual / power / linewidth / trajectory quality

Output-vs-output loci are retained only as secondary diagnostics in a fixed single-case analysis.

## Representative full cases

The complete scalar scan is always retained. Because full trajectories, observer waveforms and detailed spectra are more expensive, the GUI can propose 3–6 existing scan rows for full V11 analysis. Coverage and Feature-aware selection both choose actual calculated rows; no displayed representative result is interpolated.

Representative full-case output includes case-by-case spectra versus frequency, transverse trajectory versus z, and magnetic field along the trajectory versus z. Any scalar scan row can still be selected manually for the complete Full Visualization report.

## Error strength

All RADIA engineering-error sources expose physical strength inputs in the Error isolation section. A dedicated error-strength response scan supports:

- field amplitude sigma (%)
- longitudinal position sigma (um)
- transverse position sigma (um)
- magnetization angle sigma (mrad)
- gap asymmetry (um)
- bank strength imbalance (%)
- all currently selected errors scaled together by a nominal multiplier

One-error-at-a-time sweeps disable the other error sources and keep the random seed fixed through the sweep.

## Validation

- Python compileall PASS
- scan-centric presentation regression PASS
- v9.3 scan-axis invariant regression PASS
- full visualization layout PASS
- V6/V7/V8/V9 regression tests PASS
- GitHub preflight PASS
