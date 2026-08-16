"""Plugin version / update via git + GitHub releases.

Two modes:
  release — HEAD is exactly at a tag `vX.Y.Z`: compare against the newest
            remote tag; update = fetch + checkout newest tag.
  dev     — HEAD is on a branch: report commits behind origin/<branch>;
            update = fast-forward pull. Never switches a dev checkout to a tag.
"""

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Optional

from ..version import PRODUCT_VERSION

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).resolve().parent.parent.parent
_TAG_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")

_state = {"checked_at": 0.0, "result": None, "checking": False}


async def _git(*args, timeout: float = 30) -> tuple:
    proc = await asyncio.create_subprocess_exec(
        "git", *args, cwd=str(_PLUGIN_DIR),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "", "timeout"
    return proc.returncode, out.decode(errors="replace").strip(), err.decode(errors="replace").strip()


def _ver_key(tag: str):
    m = _TAG_RE.match(tag)
    return tuple(int(x) for x in m.groups()) if m else None


def is_git_checkout() -> bool:
    return (_PLUGIN_DIR / ".git").exists()


async def status(force: bool = False) -> dict:
    """Cached (6h) update status."""
    now = time.time()
    if not force and _state["result"] and now - _state["checked_at"] < 6 * 3600:
        return _state["result"]
    if _state["checking"] and _state["result"]:
        return _state["result"]
    _state["checking"] = True
    try:
        res = await _compute()
    finally:
        _state["checking"] = False
    _state["result"] = res
    _state["checked_at"] = now
    return res


async def _compute() -> dict:
    base = {"version": PRODUCT_VERSION, "git": is_git_checkout(), "mode": None,
            "update_available": False, "latest": None, "behind": 0, "branch": None,
            "head": None, "error": None, "checked_at": time.time()}
    if not is_git_checkout():
        base["mode"] = "static"
        return base
    rc, head, _ = await _git("rev-parse", "--short", "HEAD")
    base["head"] = head if rc == 0 else None
    rc, _, err = await _git("fetch", "--tags", "--quiet", timeout=60)
    if rc != 0:
        base["error"] = f"git fetch failed: {err.splitlines()[-1] if err else rc}"
    rc, tag_here, _ = await _git("describe", "--tags", "--exact-match", "HEAD")
    if rc == 0 and _ver_key(tag_here):
        base["mode"] = "release"
        base["version"] = tag_here.lstrip("v")
        rc, tags, _ = await _git("tag", "--list")
        vers = [t for t in tags.splitlines() if _ver_key(t)]
        if vers:
            latest = max(vers, key=_ver_key)
            base["latest"] = latest.lstrip("v")
            base["update_available"] = _ver_key(latest) > _ver_key(tag_here)
        return base
    base["mode"] = "dev"
    rc, branch, _ = await _git("rev-parse", "--abbrev-ref", "HEAD")
    base["branch"] = branch if rc == 0 else None
    if branch and branch != "HEAD":
        rc, cnt, _ = await _git("rev-list", "--count", f"HEAD..origin/{branch}")
        if rc == 0 and cnt.isdigit():
            base["behind"] = int(cnt)
            base["update_available"] = int(cnt) > 0
    return base


async def apply_update(log) -> None:
    """Perform the update for the current mode. Raises on failure."""
    st = await status(force=True)
    if st["mode"] == "release":
        if not st.get("latest") or not st["update_available"]:
            log("Already on the newest release.")
            return
        rc, out, err = await _git("checkout", f"v{st['latest']}", timeout=60)
        log(out or err)
        if rc != 0:
            raise RuntimeError(f"git checkout failed: {err}")
    elif st["mode"] == "dev":
        rc, out, err = await _git("pull", "--ff-only", timeout=120)
        log(out or err)
        if rc != 0:
            raise RuntimeError(f"git pull failed: {err}")
    else:
        raise RuntimeError("Not a git checkout — update ComfyUI-Stimma the way you installed it.")
    _state["checked_at"] = 0
