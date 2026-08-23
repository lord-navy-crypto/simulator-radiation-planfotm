# RADIA Magnet Studio Transfer Schema 1.0

The downstream export is a ZIP container with a stable, versioned contract.

## Required files

- `manifest.json`: schema version, producer version, file sizes and SHA-256 checksums.
- `device_config.json`: device, material, error and sampling parameters plus unit and coordinate conventions.
- `analysis_summary.json`: scalar analysis results; non-finite diagnostics are represented as JSON `null`.
- `on_axis_field.csv`: `z_mm,Bx_T,By_T,Bz_T`.

## Optional files

- `device_geometry.json`: the realized block geometry, including deterministic manufacturing errors.
- `field_map_3d.csv`: `x_m,y_m,z_m,Bx_T,By_T,Bz_T` for direct downstream field interpolation.

## Reader requirements

Readers must reject unsupported major schema versions, verify each checksum, use the declared units and coordinate system, and report missing optional files without failing the required on-axis workflow.

The first implementation is intentionally one-way: Magnet Studio exports and the downstream radiation platform imports. This keeps the interface testable before bidirectional editing is introduced.
