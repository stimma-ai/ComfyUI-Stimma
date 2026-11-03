"""Opaque media companions and media-time mapping. No model or tensor dependencies."""
from fractions import Fraction

SIDECAR_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["format", "format_version"],
        "properties": {
            "asset_id": {"type": "string"},
            "format": {"type": "string"},
            "format_version": {"type": "integer"},
            "mime_type": {"type": "string"},
            "sha256": {"type": "string"},
            "byte_size": {"type": "integer"},
            "display_label": {"type": "string"},
            "descriptor": {"type": "object"},
            "media_mapping": {"type": "object"},
            "status": {"enum": ["available", "unavailable"]},
            "unavailable_reason": {"type": "string"},
        },
    },
}


def map_interval(mapping, start, length, streams=()):
    """Map seconds on the current media to original-media seconds, or decline.

    Only a single preserved rate-1 span is supported. Native sample slicing is
    deliberately left to the consuming adapter.
    """
    if not isinstance(mapping, dict) or mapping.get("version") != 1:
        return None
    try:
        current = mapping["media_time_base"]
        original = mapping["source_time_base"]
        ct = Fraction(current["num"], current["den"])
        ot = Fraction(original["num"], original["den"])
        if ct <= 0 or ot <= 0:
            return None
        start = Fraction(str(start))
        length = Fraction(str(length))
        if start < 0 or length <= 0:
            return None
        lo, hi = mapping["source_extent"]
        for span in mapping["spans"]:
            if not set(streams).issubset(span["preserved_streams"]):
                continue
            a, b = span["media"]
            x, y = span["source"]
            if not (0 <= a < b and 0 <= lo <= x < y <= hi):
                continue
            if (b - a) * ct != (y - x) * ot:
                continue
            # Decimal seconds can be a rounded display of an exact frame time.
            # Range selection tolerates <1 microsecond, never a different frame.
            eps = Fraction(1, 1000000)
            end = start + length
            if start >= a * ct - eps and end <= b * ct + eps:
                offset = max(Fraction(0), start - a * ct)
                return x * ot + offset, min(length, y * ot - x * ot - offset)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return None


def identity_mapping(frame_count, fps, streams):
    return {
        "version": 1,
        "media_time_base": {"num": 1, "den": fps},
        "source_time_base": {"num": 1, "den": fps},
        "source_extent": [0, frame_count],
        "spans": [{
            "media": [0, frame_count], "source": [0, frame_count],
            "preserved_streams": list(streams),
        }],
    }
