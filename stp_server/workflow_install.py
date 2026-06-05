"""Install bundled Stimma workflows into ComfyUI's user workflow directory.

Copies workflows from this plugin's ``workflows/`` directory into
``ComfyUI/user/default/workflows/Stimma/`` so they appear in ComfyUI's
workflow browser.  Tracks file hashes in a small manifest to avoid
clobbering user edits and to respect user deletions.
"""

import hashlib
import json
import logging
import os
import shutil

logger = logging.getLogger(__name__)

_MANIFEST_NAME = ".stimma-manifest.json"
_MANIFEST_VERSION = 1


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(manifest_path: str) -> dict:
    if not os.path.exists(manifest_path):
        return {"version": _MANIFEST_VERSION, "files": {}}
    try:
        with open(manifest_path, "r") as f:
            data = json.load(f)
        if data.get("version") != _MANIFEST_VERSION:
            return {"version": _MANIFEST_VERSION, "files": {}}
        return data
    except (json.JSONDecodeError, KeyError, TypeError):
        return {"version": _MANIFEST_VERSION, "files": {}}


def _save_manifest(manifest_path: str, manifest: dict):
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)


def sync_bundled_workflows():
    """Copy bundled workflows to ComfyUI's user workflow directory.

    Rules:
    - New source file, no dest → copy it
    - Source updated, dest unchanged from our last write → update it
    - Source updated, dest modified by user → leave it alone
    - Dest deleted by user → mark deleted, never re-copy
    - Source removed from plugin → leave dest alone
    """
    try:
        import folder_paths
    except ImportError:
        logger.warning("folder_paths not available, skipping workflow install")
        return

    plugin_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_dir = os.path.join(plugin_dir, "workflows")
    if not os.path.isdir(source_dir):
        return

    user_dir = folder_paths.get_user_directory()
    dest_dir = os.path.join(user_dir, "default", "workflows", "Stimma")
    os.makedirs(dest_dir, exist_ok=True)

    manifest_path = os.path.join(dest_dir, _MANIFEST_NAME)
    manifest = _load_manifest(manifest_path)
    files = manifest["files"]
    changed = False

    for name in sorted(os.listdir(source_dir)):
        if not name.endswith(".json"):
            continue

        src_path = os.path.join(source_dir, name)
        dst_path = os.path.join(dest_dir, name)
        src_hash = _sha256(src_path)

        entry = files.get(name)

        if entry is None:
            # Never tracked — only copy if dest doesn't already exist
            if os.path.exists(dst_path):
                logger.debug(f"Workflow {name} already exists (not ours), skipping")
            else:
                shutil.copy2(src_path, dst_path)
                files[name] = {"hash": src_hash}
                changed = True
                logger.info(f"Installed workflow: {name}")

        elif entry.get("deleted"):
            # User deleted it — respect that
            pass

        elif not os.path.exists(dst_path):
            # We wrote it before but user deleted it
            entry["deleted"] = True
            changed = True
            logger.info(f"Workflow {name} deleted by user, will not re-copy")

        elif src_hash == entry["hash"]:
            # Source unchanged — nothing to do
            pass

        else:
            # Source changed — only update if user hasn't touched the dest
            dst_hash = _sha256(dst_path)
            if dst_hash == entry["hash"]:
                shutil.copy2(src_path, dst_path)
                entry["hash"] = src_hash
                changed = True
                logger.info(f"Updated workflow: {name}")
            else:
                logger.info(f"Workflow {name} modified by user, skipping update")

    if changed:
        _save_manifest(manifest_path, manifest)
