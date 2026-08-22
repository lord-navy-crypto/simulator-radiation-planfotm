# Validation status

## Automated regression coverage

The repository currently contains regression checks covering:

- RADIA bridge construction using a fake RADIA implementation.
- Full real-field-map tracking range.
- NumPy-safe JSON serialization.
- Generalized resonance and theoretical-power regressions.
- Central-period magnetic harmonics versus full-range field integrals.
- Causal retarded-time solver behavior.
- RADIA CSV import and arbitrary magnetic period.
- B0 target convergence gate.
- Off-axis spectrum and angular-analysis behavior.
- Non-overlapping prototype geometry checks.
- Tracking-resolution control behavior.
- Radiative-fluence angular weighting.
- Explicit invalid-pixel masks for failed 2-D angular-map samples.
- Current scan-helper result schema.

Run:

```bash
python3 scripts/run_tests.py
```

## What passing tests mean

A passing test suite demonstrates consistency with the implemented analytic and
mock regression cases. It does not by itself certify:

- agreement with a specific facility's magnetic measurement pipeline;
- mechanical fidelity to a specific installed APPLE-II or helical undulator;
- collective bunch, emittance, wakefield, or coherent-radiation effects;
- accuracy outside the numerical/physical assumptions of the implemented
  single-electron model.

## RADIA runtime

The GitHub CI workflow intentionally does not bundle a compiled RADIA binary.
Real RADIA numerical runs therefore remain a local platform validation step.

For published research, record:

- RADIA revision/build;
- Python version;
- dependency versions;
- device geometry and material assumptions;
- field-map resolution;
- tracking resolution;
- observer geometry;
- random seed for manufacturing errors.
