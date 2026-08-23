"""Versioned, portable configuration presets."""

from .schema import (
    BUILTIN_PRESETS,
    PRESET_SCHEMA,
    PRESET_VERSION,
    build_preset,
    parse_preset,
    preset_json_bytes,
    runtime_to_widget_state,
)

