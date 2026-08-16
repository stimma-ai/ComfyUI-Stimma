"""Custom node packs: class_type → pack lookup, install via ComfyUI-Manager."""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _PLUGIN_DIR / ".stimma-manage"
_NODE_MAP_CACHE = _CACHE_DIR / "extension-node-map.json"
_NODE_MAP_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"
_NODE_MAP_TTL = 24 * 3600

_node_map: Dict[str, Any] = {"loaded_at": 0, "by_class": {}}


def comfy_base_path() -> Optional[str]:
    try:
        import folder_paths
        return folder_paths.base_path
    except Exception:
        return None


def manager_dir() -> Optional[Path]:
    base = comfy_base_path()
    if not base:
        return None
    for name in ("ComfyUI-Manager", "comfyui-manager"):
        p = Path(base) / "custom_nodes" / name
        if (p / "cm-cli.py").exists():
            return p
    return None


def has_manager() -> bool:
    if manager_dir() is not None:
        return True
    try:
        import comfyui_manager  # noqa: F401  (pip-installed manager, newer ComfyUI)
        return True
    except Exception:
        return False


def _index_map(raw: dict) -> Dict[str, dict]:
    """extension-node-map.json: {repo_url: [[class,...], {title_aux, ...}]} → class → {url, title}."""
    by_class = {}
    for url, val in raw.items():
        try:
            classes, meta = val[0], (val[1] if len(val) > 1 else {})
        except Exception:
            continue
        for c in classes:
            by_class.setdefault(c, {"url": url, "title": meta.get("title_aux") or url.rstrip("/").split("/")[-1]})
    return by_class


async def _ensure_node_map() -> Dict[str, dict]:
    now = time.time()
    if _node_map["by_class"] and now - _node_map["loaded_at"] < _NODE_MAP_TTL:
        return _node_map["by_class"]
    # Local Manager copy first (ships in the repo), then our own cached fetch.
    md = manager_dir()
    candidates = []
    if md:
        candidates.append(md / "extension-node-map.json")
    candidates.append(_NODE_MAP_CACHE)
    raw = None
    for c in candidates:
        try:
            if not c.exists():
                continue
            if c == _NODE_MAP_CACHE and now - c.stat().st_mtime > _NODE_MAP_TTL:
                continue
            raw = json.loads(c.read_text())
            break
        except Exception:
            continue
    if raw is None:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as s:
                async with s.get(_NODE_MAP_URL) as r:
                    if r.status == 200:
                        text = await r.text()
                        raw = json.loads(text)
                        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        _NODE_MAP_CACHE.write_text(text)
        except Exception:
            logger.debug("node map fetch failed", exc_info=True)
    if raw:
        _node_map["by_class"] = _index_map(raw)
        _node_map["loaded_at"] = now
    return _node_map["by_class"]


async def lookup_pack(class_type: str) -> Optional[dict]:
    m = await _ensure_node_map()
    return m.get(class_type)


def installed_pack_names() -> List[str]:
    base = comfy_base_path()
    if not base:
        return []
    cn = Path(base) / "custom_nodes"
    try:
        return sorted(p.name for p in cn.iterdir() if p.is_dir() and not p.name.startswith((".", "__")))
    except OSError:
        return []


def pack_installed(url: str) -> bool:
    name = url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    names = {n.lower() for n in installed_pack_names()}
    return name.lower() in names


def python_exe() -> str:
    return sys.executable


async def install_pack(url: str, log) -> None:
    """Install a node pack with cm-cli (Manager's CLI, same venv). Raises on failure."""
    md = manager_dir()
    if md is None:
        raise RuntimeError("ComfyUI-Manager is not installed")
    base = comfy_base_path() or str(md.parent.parent)
    cmd = [python_exe(), str(md / "cm-cli.py"), "install", url]
    env = dict(os.environ)
    env.setdefault("COMFYUI_PATH", base)
    log(f"$ {' '.join(cmd)}")
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=base, env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    assert proc.stdout
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        log(line.decode(errors="replace").rstrip())
    rc = await proc.wait()
    if rc != 0:
        raise RuntimeError(f"cm-cli exited with code {rc}")


def manual_instructions(url: str) -> str:
    base = comfy_base_path() or "<ComfyUI>"
    return f"cd {os.path.join(base, 'custom_nodes')} && git clone {url}"
