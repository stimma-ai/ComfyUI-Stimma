"""Resolve a missing model file to a download source + destination.

Order: the workflow's own embedded hint (ComfyUI's nodes[].properties.models)
→ our bundled manifest (models.json) → nothing (user pastes a URL).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = _PLUGIN_DIR / "models.json"
USER_SOURCES_PATH = _PLUGIN_DIR / ".stimma-manage" / "sources.json"

_manifest_cache: Dict[str, Any] = {"mtime": None, "data": {}}


def load_manifest() -> Dict[str, Any]:
    try:
        mtime = MANIFEST_PATH.stat().st_mtime
    except OSError:
        return {}
    if _manifest_cache["mtime"] != mtime:
        try:
            _manifest_cache["data"] = json.loads(MANIFEST_PATH.read_text()).get("models", {}) or {}
        except Exception:
            logger.warning("models.json unreadable", exc_info=True)
            _manifest_cache["data"] = {}
        _manifest_cache["mtime"] = mtime
    return _manifest_cache["data"]


def _load_user_sources() -> Dict[str, Any]:
    try:
        if USER_SOURCES_PATH.exists():
            return json.loads(USER_SOURCES_PATH.read_text())
    except Exception:
        pass
    return {}


def remember_user_source(filename: str, url: str, folder: Optional[str]) -> None:
    data = _load_user_sources()
    data[filename] = {"url": url, "folder": folder}
    USER_SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_SOURCES_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


def _norm(name: str) -> str:
    return name.replace("\\", "/")


def resolve_source(filename: str, hints: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[dict]:
    """Return {url, size?, sha256?, gated, license_url?, repo?, folder?, via} or None."""
    fname = _norm(filename)
    base = os.path.basename(fname)

    # 1. embedded workflow hint
    if hints:
        h = hints.get(base) or hints.get(fname)
        if h and h.get("url"):
            sha = None
            if h.get("hash") and (h.get("hash_type") or "").lower() in ("", "sha256") and len(str(h["hash"])) == 64:
                sha = str(h["hash"]).lower()
            # Enrich with the manifest's size/hash/gating when it knows this file.
            m = load_manifest().get(fname) or load_manifest().get(base) or {}
            return {
                "url": h["url"], "size": m.get("size"), "sha256": sha or (m.get("sha256") or None),
                "gated": bool(m.get("gated")),
                "license_url": m.get("license_url"), "repo": _repo_from_url(h["url"]),
                "folder": h.get("directory"), "via": "workflow",
            }

    # 2. bundled manifest (exact key, then basename)
    manifest = load_manifest()
    entry = manifest.get(fname) or manifest.get(base)
    if entry:
        src = entry.get("source") or {}
        if src.get("type") == "huggingface" and src.get("repo") and src.get("path"):
            url = hf_url(src["repo"], src["path"])
            repo = src["repo"]
        else:
            url = src.get("url")
            repo = _repo_from_url(url or "")
        if url:
            return {
                "url": url, "size": entry.get("size"), "sha256": (entry.get("sha256") or None),
                "gated": bool(entry.get("gated")), "license_url": entry.get("license_url") or (f"https://huggingface.co/{repo}" if repo and entry.get("gated") else None),
                "repo": repo, "folder": entry.get("directory"), "via": "manifest",
            }

    # 3. remembered user-supplied URL
    user = _load_user_sources()
    u = user.get(fname) or user.get(base)
    if u and u.get("url"):
        return {"url": u["url"], "size": None, "sha256": None, "gated": False, "license_url": None,
                "repo": _repo_from_url(u["url"]), "folder": u.get("folder"), "via": "user"}
    return None


def _repo_from_url(url: str) -> Optional[str]:
    if "huggingface.co/" not in (url or ""):
        return None
    parts = url.split("huggingface.co/", 1)[1].split("/")
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else None


def dest_path_for(filename: str, folder: Optional[str]) -> Optional[str]:
    """Absolute destination for a model file inside ComfyUI's model dirs."""
    try:
        import folder_paths
    except ImportError:
        return None
    fname = _norm(filename)
    folder = folder or "diffusion_models"
    # ComfyUI aliases: unet → diffusion_models, clip → text_encoders
    alias = {"unet": "diffusion_models", "clip": "text_encoders"}
    folder = alias.get(folder, folder)
    try:
        dirs = folder_paths.get_folder_paths(folder)
    except Exception:
        dirs = []
    if not dirs:
        try:
            dirs = [os.path.join(folder_paths.models_dir, folder)]
        except Exception:
            return None
    base_dir = dirs[0]
    return os.path.join(base_dir, *fname.split("/"))


def model_dirs() -> list:
    """Distinct model root directories ComfyUI knows about (for disk stats)."""
    try:
        import folder_paths
        roots = [folder_paths.models_dir]
        for name in ("checkpoints", "diffusion_models", "loras", "text_encoders", "vae"):
            try:
                roots.extend(folder_paths.get_folder_paths(name))
            except Exception:
                pass
        seen, out = set(), []
        for r in roots:
            rp = os.path.realpath(r)
            if rp not in seen and os.path.isdir(rp):
                seen.add(rp)
                out.append(rp)
        return out
    except Exception:
        return []
