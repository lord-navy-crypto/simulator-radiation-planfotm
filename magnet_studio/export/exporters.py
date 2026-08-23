from __future__ import annotations
import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

SCHEMA_NAME = "radia-magnet-studio-transfer"
SCHEMA_VERSION = "1.0.0"
PRODUCER_VERSION = "3.1.0"


def _field_arrays(z_mm, B):
    z = np.asarray(z_mm, dtype=float)
    field = np.asarray(B, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z_mm must be a one-dimensional array with at least two samples.")
    if field.shape != (z.size, 3):
        raise ValueError(f"B must have shape ({z.size}, 3); got {field.shape}.")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(field)):
        raise ValueError("Field data must contain only finite numerical values.")
    if not np.all(np.diff(z) > 0):
        raise ValueError("z_mm must be strictly increasing.")
    return z, field


def csv_bytes(z_mm, B):
    z_mm, B = _field_arrays(z_mm, B)
    out=io.StringIO()
    out.write("z_mm,Bx_T,By_T,Bz_T\n")
    for z,b in zip(z_mm,B):
        out.write(f"{float(z)},{float(b[0])},{float(b[1])},{float(b[2])}\n")
    return out.getvalue().encode("utf-8")

def _json_safe(value):
    """
    Recursively convert NumPy/Python scientific objects to standard JSON types.
    This keeps the numerical values unchanged while making arrays serializable.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value type: {type(value).__name__}")

def json_bytes(params, metrics):
    payload = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "producer": {"name": "RADIA Magnet Studio", "version": PRODUCER_VERSION},
        "units": {"length": "mm", "magnetic_field": "T", "energy": "GeV"},
        "coordinate_system": {
            "handedness": "right-handed",
            "x": "horizontal",
            "y": "vertical",
            "z": "longitudinal beam direction",
        },
        "parameters": _json_safe(params),
        "metrics": _json_safe(metrics),
    }
    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _write_hdf5_value(group, key, value, skipped):
    """Write nested scientific data without silently losing array results."""
    safe_key = str(key).replace("/", "_")
    if isinstance(value, dict):
        child = group.create_group(safe_key)
        for nested_key, nested_value in value.items():
            _write_hdf5_value(child, nested_key, nested_value, skipped)
        return
    if value is None:
        group.attrs[safe_key] = "None"
        return
    if isinstance(value, (str, bytes, bool, int, float, np.number)):
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            group.attrs[safe_key] = "non-finite"
        else:
            group.attrs[safe_key] = value
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.kind in "biufc":
            group.create_dataset(safe_key, data=array)
            return
        if array.dtype.kind in "SU":
            dtype = __import__("h5py").string_dtype(encoding="utf-8")
            group.create_dataset(safe_key, data=array.astype(object), dtype=dtype)
            return
    skipped.append(f"{group.name}/{safe_key}:{type(value).__name__}")

def hdf5_bytes(z_mm, B, params, metrics):
    import h5py
    z_mm, B = _field_arrays(z_mm, B)
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    skipped = []
    try:
        with h5py.File(path, "w") as f:
            f.create_dataset("z_mm", data=np.asarray(z_mm, float))
            f.create_dataset("B_T", data=np.asarray(B, float))

            f.attrs["schema_name"] = SCHEMA_NAME
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["producer_version"] = PRODUCER_VERSION
            f["z_mm"].attrs["unit"] = "mm"
            f["B_T"].attrs["unit"] = "T"
            p = f.create_group("parameters")
            for k, v in params.items():
                _write_hdf5_value(p, k, v, skipped)
            m = f.create_group("metrics")
            for k, v in metrics.items():
                _write_hdf5_value(m, k, v, skipped)

            if skipped:
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset("export_skipped_items", data=np.asarray(skipped, dtype=object), dtype=dt)

        return open(path, "rb").read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            # Failure to remove a temporary file does not corrupt the exported
            # HDF5 bytes; deliberately ignore only this cleanup condition.
            pass

def pdf_bytes(z_mm,B,params,metrics):
    z_mm, B = _field_arrays(z_mm, B)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    bio=io.BytesIO()
    with PdfPages(bio) as pdf:
        fig=plt.figure(figsize=(8.5,11))
        fig.text(0.08,0.95,"RADIA Magnet Studio Research Report",fontsize=16)
        y=0.90
        for k,v in params.items():
            fig.text(0.08,y,f"{k}: {v}",fontsize=9); y-=0.025
        y-=0.02
        for k,v in metrics.items():
            if k in ("trajectory","electron_phase"): continue
            fig.text(0.08,y,f"{k}: {v}",fontsize=9); y-=0.025
            if y<0.08: break
        plt.axis("off"); pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(10,6))
        plt.plot(z_mm,B[:,0],label="Bx")
        plt.plot(z_mm,B[:,1],label="By")
        plt.plot(z_mm,B[:,2],label="Bz")
        plt.xlabel("z (mm)"); plt.ylabel("B (T)"); plt.legend(); plt.title("On-axis magnetic field")
        pdf.savefig(fig); plt.close(fig)

        tr=metrics.get("trajectory")
        if tr is not None:
            fig=plt.figure(figsize=(10,6))
            plt.plot(z_mm,tr["x_mm"],label="x(z)")
            plt.plot(z_mm,tr["y_mm"],label="y(z)")
            plt.xlabel("z (mm)"); plt.ylabel("displacement (mm)"); plt.legend(); plt.title("Electron trajectory")
            pdf.savefig(fig); plt.close(fig)
    return bio.getvalue()


def fieldmap3d_csv_bytes(x_mm,y_mm,z_mm,B3):
    x_mm = np.asarray(x_mm, dtype=float)
    y_mm = np.asarray(y_mm, dtype=float)
    z_mm = np.asarray(z_mm, dtype=float)
    B3 = np.asarray(B3, dtype=float)
    expected = (len(z_mm), len(y_mm), len(x_mm), 3)
    if B3.shape != expected:
        raise ValueError(f"B3 must have shape {expected}; got {B3.shape}.")
    if not all(np.all(np.isfinite(a)) for a in (x_mm, y_mm, z_mm, B3)):
        raise ValueError("3D field-map coordinates and values must be finite.")
    out=io.StringIO()
    out.write("x_m,y_m,z_m,Bx_T,By_T,Bz_T\n")
    for iz,z in enumerate(z_mm):
        for iy,y in enumerate(y_mm):
            for ix,x in enumerate(x_mm):
                b=B3[iz,iy,ix]
                out.write(
                    f"{float(x)*1e-3},{float(y)*1e-3},{float(z)*1e-3},"
                    f"{float(b[0])},{float(b[1])},{float(b[2])}\n"
                )
    return out.getvalue().encode("utf-8")


def research_package_bytes(
    params, metrics, z_mm, B, *, grid3=None, field3=None, blocks=None,
    comparison=None, run_metadata=None,
):
    """Build a versioned, checksum-protected package for a downstream simulator."""
    z_mm, B = _field_arrays(z_mm, B)
    files = {
        "device_config.json": json.dumps(
            {
                "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
                "units": {"length": "mm", "magnetic_field": "T", "energy": "GeV"},
                "coordinate_system": {
                    "handedness": "right-handed", "x": "horizontal",
                    "y": "vertical", "z": "longitudinal beam direction",
                },
                "parameters": _json_safe(params),
            }, indent=2, ensure_ascii=False, allow_nan=False,
        ).encode("utf-8"),
        "analysis_summary.json": json.dumps(
            {"metrics": _json_safe({k: v for k, v in metrics.items()
                                     if k not in ("trajectory", "electron_phase")})},
            indent=2, ensure_ascii=False, allow_nan=False,
        ).encode("utf-8"),
        "on_axis_field.csv": csv_bytes(z_mm, B),
    }
    if blocks is not None:
        files["device_geometry.json"] = json.dumps(
            {"blocks": _json_safe(blocks)}, indent=2, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    if grid3 is not None or field3 is not None:
        if grid3 is None or field3 is None:
            raise ValueError("grid3 and field3 must be supplied together.")
        files["field_map_3d.csv"] = fieldmap3d_csv_bytes(*grid3, field3)
    if comparison is not None:
        files["ideal_error_comparison.json"] = json.dumps(
            {"comparison": _json_safe(comparison)},
            indent=2, ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    if run_metadata is not None:
        files["run_metadata.json"] = json.dumps(
            {"run_metadata": _json_safe(run_metadata)},
            indent=2, ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")

    manifest = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "producer": {"name": "RADIA Magnet Studio", "version": PRODUCER_VERSION},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "compatibility": {"minimum_reader_schema": "1.0.0"},
        "files": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(files.items())
        ],
    }
    files["manifest.json"] = json.dumps(
        manifest, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    return output.getvalue()


def validate_research_package_bytes(package):
    """Validate paths, schema, required files, sizes and SHA-256 checksums."""
    required = {
        "manifest.json", "device_config.json", "analysis_summary.json",
        "on_axis_field.csv",
    }
    try:
        archive = zipfile.ZipFile(io.BytesIO(package))
    except (zipfile.BadZipFile, TypeError) as exc:
        raise ValueError("Invalid RADIA transfer ZIP.") from exc
    with archive:
        names = set(archive.namelist())
        unsafe = [name for name in names if name.startswith(("/", "\\")) or ".." in Path(name).parts]
        if unsafe:
            raise ValueError(f"Unsafe path(s) in transfer package: {unsafe}")
        missing = sorted(required - names)
        if missing:
            raise ValueError("Transfer package is missing: " + ", ".join(missing))
        manifest = json.loads(archive.read("manifest.json"))
        schema = manifest.get("schema", {})
        if schema.get("name") != SCHEMA_NAME:
            raise ValueError(f"Unexpected schema name: {schema.get('name')!r}.")
        version = str(schema.get("version", ""))
        if version.split(".", 1)[0] != SCHEMA_VERSION.split(".", 1)[0]:
            raise ValueError(f"Unsupported schema version: {version!r}.")
        checked = 0
        for item in manifest.get("files", []):
            name = item.get("path")
            if name not in names:
                raise ValueError(f"Manifest references missing file: {name!r}.")
            data = archive.read(name)
            if len(data) != int(item.get("bytes", -1)):
                raise ValueError(f"Size mismatch for {name}.")
            if hashlib.sha256(data).hexdigest() != item.get("sha256"):
                raise ValueError(f"Checksum mismatch for {name}.")
            checked += 1
        if checked != len(names) - 1:
            raise ValueError("Manifest does not describe every payload file.")
        config = json.loads(archive.read("device_config.json"))
        if config.get("schema") != schema:
            raise ValueError("Manifest and device-config schema versions disagree.")
        header = archive.read("on_axis_field.csv").splitlines()[0]
        if header != b"z_mm,Bx_T,By_T,Bz_T":
            raise ValueError("Unexpected on-axis field CSV header.")
        return {
            "schema_name": schema["name"],
            "schema_version": version,
            "payload_files_checked": checked,
            "has_3d_field_map": "field_map_3d.csv" in names,
            "has_realized_geometry": "device_geometry.json" in names,
        }
