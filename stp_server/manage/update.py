"""Plugin update: main-or-newer.

main is the release channel (like most ComfyUI plugins). A checkout strictly
behind origin/main can update (fast-forward); a checkout with local commits is
"ahead" and never nagged. Identity is the git hash. Tagged releases can layer
on later without changing this surface.
"""

import asyncio
import logging
import time
from pathlib import Path

from ..version import PRODUCT_VERSION

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).resolve().parent.parent.parent
_BRANCH = "main"

_state = {"checked_at": 0.0, "result": None, "checking": False}


def _checkout_head() -> str | None:
    """Read the checked-out commit without invoking git."""
    git_dir = _PLUGIN_DIR / ".git"
    if git_dir.is_file():
        try:
            marker = git_dir.read_text().strip()
            if marker.startswith("gitdir:"):
                git_dir = (_PLUGIN_DIR / marker.split(":", 1)[1].strip()).resolve()
        except OSError:
            return None
    try:
        head = (git_dir / "HEAD").read_text().strip()
    except OSError:
        return None
    if not head.startswith("ref:"):
        return head[:7] if head else None
    ref = head.split(":", 1)[1].strip()
    try:
        value = (git_dir / ref).read_text().strip()
        return value[:7] if value else None
    except OSError:
        try:
            for line in (git_dir / "packed-refs").read_text().splitlines():
                if line and not line.startswith(("#", "^")):
                    value, name = line.split(" ", 1)
                    if name == ref:
                        return value[:7]
        except (OSError, ValueError):
            pass
    return None


_RUNNING_HEAD = _checkout_head()


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


def is_git_checkout() -> bool:
    return (_PLUGIN_DIR / ".git").exists()


def cached_status() -> dict | None:
    """Return the last completed update check without doing git I/O."""
    return _state["result"]


async def status(force: bool = False) -> dict:
    """Cached (6h) update status."""
    now = time.time()
    checkout_head = _checkout_head()
    checkout_changed = bool(
        _state["result"]
        and checkout_head
        and _state["result"].get("head") != checkout_head
    )
    if (
        not force
        and not checkout_changed
        and _state["result"]
        and now - _state["checked_at"] < 6 * 3600
    ):
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
    base = {"version": PRODUCT_VERSION, "git": is_git_checkout(), "head": None,
            "running_head": _RUNNING_HEAD, "restart_required": False,
            "target": None,
            "behind": 0, "ahead": 0, "update_available": False, "error": None,
            "checked_at": time.time()}
    if not base["git"]:
        return base
    rc, head, _ = await _git("rev-parse", "--short", "HEAD")
    base["head"] = head if rc == 0 else None
    base["restart_required"] = bool(
        base["running_head"] and base["head"] and base["running_head"] != base["head"]
    )
    rc, _, err = await _git("fetch", "--quiet", "origin", _BRANCH, timeout=60)
    if rc != 0:
        base["error"] = f"git fetch failed: {err.splitlines()[-1] if err else rc}"
        return base
    rc, target, _ = await _git("rev-parse", "--short", f"origin/{_BRANCH}")
    base["target"] = target if rc == 0 else None
    rc, behind, _ = await _git("rev-list", "--count", f"HEAD..origin/{_BRANCH}")
    if rc == 0 and behind.isdigit():
        base["behind"] = int(behind)
    rc, ahead, _ = await _git("rev-list", "--count", f"origin/{_BRANCH}..HEAD")
    if rc == 0 and ahead.isdigit():
        base["ahead"] = int(ahead)
    # Local commits mean a dev checkout: identify it, never nag it.
    base["update_available"] = base["behind"] > 0 and base["ahead"] == 0
    return base


async def apply_update(log) -> None:
    """Fast-forward to origin/main. Raises on failure."""
    st = await status(force=True)
    if not st["git"]:
        raise RuntimeError("Not a git checkout")
    if st.get("error"):
        raise RuntimeError(st["error"])
    if not st["update_available"]:
        log("Already current")
        return
    rc, branch, _ = await _git("rev-parse", "--abbrev-ref", "HEAD")
    if rc == 0 and branch != _BRANCH:
        rc, out, err = await _git("checkout", _BRANCH, timeout=60)
        log(out or err)
        if rc != 0:
            raise RuntimeError(f"git checkout {_BRANCH} failed: {err}")
    rc, out, err = await _git("merge", "--ff-only", f"origin/{_BRANCH}", timeout=120)
    log(out or err)
    if rc != 0:
        raise RuntimeError(f"git merge failed: {err}")
    _state["checked_at"] = 0
    _state["result"] = None
