# RADIA Magnet Preset v1

`radia-magnet-preset` version `1.0.0` is the portable, settings-only contract
between RADIA Magnet Studio and downstream applications. It does not contain a
field map or calculated result arrays; those remain in the research package.

The required sections are `schema`, `device`, `material`,
`manufacturing_errors`, `calibration`, `solver`, `sampling`, and `conventions`.
The v1 units are fixed to millimetres, tesla, and GeV, with z longitudinal and
x/y transverse in a right-handed coordinate system.

Consumers must reject an incompatible major schema version, invalid required
values, unsupported units, or a mismatched `fingerprint_sha256`. They may ignore
unknown top-level fields with a warning. Program-specific settings belong under
`extensions`; this lets another simulator add controls without changing the
stable magnetic-device contract.

The UI exports a requested preset before solving and a realized preset after a
successful solve. A realized preset records the calibrated Br and calibration
history. `examples/read_preset.py` is the reference consumer.

