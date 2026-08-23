from __future__ import annotations

import io
import hashlib
import json
import zipfile
from pathlib import Path

from magnet_studio.export.exporters import validate_research_package_bytes


def saved_model_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "saved_models"


def load_transfer_package(package_bytes: bytes) -> dict:
    """Validate a Magnet Studio transfer package and expose it to Stage 2."""
    raw = bytes(package_bytes)
    status = validate_research_package_bytes(raw)
    if not status.get("has_3d_field_map"):
        raise ValueError(
            "This transfer contains only an on-axis field. Stage 2 requires field_map_3d.csv "
            "for off-axis trajectory and radiation tracking."
        )
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        config = json.loads(archive.read("device_config.json"))
        field_map = archive.read("field_map_3d.csv")
        analysis = json.loads(archive.read("analysis_summary.json"))
    parameters = config.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("device_config.json does not contain a parameters object.")
    return {
        "source": "validated_transfer_package",
        "field_map_csv": field_map,
        "parameters": parameters,
        "metrics": analysis.get("metrics", {}),
        "schema_version": status["schema_version"],
        "has_3d_field_map": True,
    }


def discover_saved_transfer_records() -> list[dict]:
    """Restore separately saved Stage-1 transfer files after an app restart."""
    folder = saved_model_directory()
    if not folder.exists():
        return []
    records = []
    for path in sorted(folder.glob("*.zip"), key=lambda p: p.stat().st_mtime):
        try:
            package = path.read_bytes()
            bridge = load_transfer_package(package)
        except Exception:
            continue
        params = dict(bridge.get("parameters", {}))
        created = str(params.get("model_record_created_utc", "saved file"))
        name = str(params.get("model_record_name", path.stem))
        digest = hashlib.sha256(package).hexdigest()[:16]
        records.append({
            "id": f"saved_{digest}", "name": name, "created_utc": created,
            "file_name": path.name, "saved_path": str(path), "package": package,
            "bridge": bridge, "visuals": None, "parameters": params,
            "metrics": dict(bridge.get("metrics", {})),
        })
    return records
