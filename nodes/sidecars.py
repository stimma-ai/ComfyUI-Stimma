"""Prepared-file companion publication, independent of payload format."""
import hashlib
import json
import os
import shutil
from pathlib import Path

SIDECARS_TYPE = "STIMMA_SIDECARS"


def publish_sidecars(bundle, output_path, fps, frame_count, width, height, preserved_streams=None):
    """Bind prepared companion files to a render without opening their format."""
    output_path = Path(output_path)
    entries = []
    for original in (bundle or {}).get("sidecars", []):
        entry = {k: v for k, v in original.items() if k != "file"}
        if entry.get("status") == "unavailable":
            entries.append(entry)
            continue
        if preserved_streams is not None and entry.get("media_mapping"):
            import copy
            entry["media_mapping"] = copy.deepcopy(entry["media_mapping"])
            for span in entry["media_mapping"].get("spans", []):
                span["preserved_streams"] = [s for s in span.get("preserved_streams", []) if s in preserved_streams]
        try:
            facts = original.get("render_facts", {})
            expected = {"fps": fps, "frame_count": frame_count, "width": width, "height": height}
            if facts and facts != expected:
                raise ValueError("render_mismatch")
            source = Path(original["file"])
            dest = output_path.parent / (output_path.name + f".state_{len(entries)}" + source.suffix)
            if source != dest:
                shutil.copyfile(source, dest)
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            if entry.get("sha256") and digest != entry["sha256"]:
                dest.unlink(missing_ok=True)
                raise ValueError("integrity_failed")
            entry.update(file=dest.name, sha256=digest, byte_size=dest.stat().st_size)
        except (OSError, ValueError) as exc:
            entry.update(status="unavailable", unavailable_reason=str(exc) if isinstance(exc, ValueError) else "capture_failed")
        entries.append(entry)
    manifest = {"version": 1, "assets": [{
        "file": output_path.name, "type": "video", "role": "primary", "sidecars": entries,
    }], "input_usage": (bundle or {}).get("input_usage", [])}
    path = Path(str(output_path) + ".manifest.json")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest), encoding="utf-8")
    os.replace(tmp, path)
    return manifest
