from __future__ import annotations

def z_edges_from_blocks(blocks):
    if not blocks:
        raise ValueError("Cannot determine field range from an empty geometry.")
    lo = float("inf")
    hi = float("-inf")
    for b in blocks:
        c = b["center"]
        s = b["size"]
        z0 = float(c[2])
        hz = 0.5 * float(s[2])
        lo = min(lo, z0 - hz)
        hi = max(hi, z0 + hz)
    return lo, hi

def geometry_field_range(blocks, period_mm, margin_periods=1.0):
    lo, hi = z_edges_from_blocks(blocks)
    margin = max(0.0, float(margin_periods)) * float(period_mm)
    return lo - margin, hi + margin

def union_field_range(models, period_mm, margin_periods=1.0):
    ranges = [z_edges_from_blocks(m["blocks"]) for m in models if m is not None]
    if not ranges:
        raise ValueError("No models supplied for field-range calculation.")
    lo = min(r[0] for r in ranges)
    hi = max(r[1] for r in ranges)
    margin = max(0.0, float(margin_periods)) * float(period_mm)
    return lo - margin, hi + margin
