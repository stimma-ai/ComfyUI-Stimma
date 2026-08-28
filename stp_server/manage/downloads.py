"""Model download engine.

Downloads run on this machine (the ComfyUI host), straight into ComfyUI's
model directories, with resume (HTTP Range onto a .part file), sha256
verification when the hash is known, an HF bearer token when we have one,
and clear classification of the usual failure modes (gated repo, bad token,
missing file, disk full).
"""

import asyncio
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Any

import aiohttp

from . import credentials
from .ops import Operation, OperationRegistry, STATE_QUEUED, STATE_RUNNING, STATE_DONE, STATE_FAILED, STATE_CANCELLED, STATE_PAUSED

logger = logging.getLogger(__name__)

_CHUNK = 1024 * 1024
_MAX_PARALLEL = 2


def _fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB", "MB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return ""
    s = int(seconds)
    if s < 60:
        return f"{s}s left"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}:{s:02d} left"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m left"


class DownloadError(Exception):
    def __init__(self, message: str, kind: str = "other", fix: Optional[dict] = None):
        super().__init__(message)
        self.kind = kind
        self.fix = fix


class DownloadManager:
    def __init__(self, ops: OperationRegistry, on_complete: Optional[Callable[[Operation, bool], Any]] = None):
        self._ops = ops
        self._on_complete = on_complete
        self._sem = asyncio.Semaphore(_MAX_PARALLEL)
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cancel: Dict[str, str] = {}  # op_id -> "cancel" | "pause"
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=120))
        return self._session

    async def stop(self):
        for t in list(self._tasks.values()):
            t.cancel()
        if self._session and not self._session.closed:
            await self._session.close()

    # -- public --
    def enqueue(self, *, filename: str, url: str, dest_path: str, size: Optional[int] = None,
                sha256: Optional[str] = None, gated: bool = False, license_url: Optional[str] = None,
                repo: Optional[str] = None, group: Optional[str] = None, workflows: Optional[list] = None,
                title: Optional[str] = None) -> Operation:
        """Queue a download; dedupes on dest_path while an op is active."""
        existing = self._ops.find("download", dest_path=dest_path)
        if existing:
            if workflows:
                ws = set(existing.meta.get("workflows") or [])
                ws.update(workflows)
                existing.meta["workflows"] = sorted(ws)
            return existing
        op = self._ops.create(
            "download", title or filename,
            meta={
                "filename": filename, "url": url, "dest_path": dest_path, "size": size,
                "sha256": sha256, "gated": gated, "license_url": license_url, "repo": repo,
                "bytes": 0, "workflows": sorted(set(workflows or [])),
            },
            group=group,
        )
        self._start(op)
        return op

    def resume_all_queued(self):
        for op in self._ops.all():
            if op.kind == "download" and op.state == STATE_QUEUED and op.id not in self._tasks:
                self._start(op)

    def retry(self, op_id: str) -> Optional[Operation]:
        op = self._ops.get(op_id)
        if not op or op.kind != "download":
            return None
        if op.id in self._tasks and not self._tasks[op.id].done():
            return op
        self._ops.update(op, state=STATE_QUEUED, error=None, error_kind=None, fix=None, detail=None, finished_at=None)
        self._start(op)
        return op

    def pause(self, op_id: str) -> bool:
        op = self._ops.get(op_id)
        if not op or op.state not in (STATE_RUNNING, STATE_QUEUED):
            return False
        self._cancel[op_id] = "pause"
        t = self._tasks.get(op_id)
        if t and not t.done():
            t.cancel()
        else:
            self._ops.update(op, state=STATE_PAUSED, detail="Paused")
        return True

    def cancel(self, op_id: str) -> bool:
        op = self._ops.get(op_id)
        if not op:
            return False
        self._cancel[op_id] = "cancel"
        t = self._tasks.get(op_id)
        if t and not t.done():
            t.cancel()
        else:
            self._finish_cancel(op)
        return True

    # -- internals --
    def _start(self, op: Operation):
        self._cancel.pop(op.id, None)
        self._tasks[op.id] = asyncio.create_task(self._run(op))

    def _finish_cancel(self, op: Operation):
        part = op.meta.get("dest_path", "") + ".part"
        try:
            if os.path.exists(part):
                os.remove(part)
        except OSError:
            pass
        self._ops.update(op, state=STATE_CANCELLED, detail="Cancelled", progress=None)
        self._ops.save()

    async def _run(self, op: Operation):
        async with self._sem:
            if self._cancel.get(op.id):
                return
            try:
                self._ops.update(op, state=STATE_RUNNING, detail=None, error=None, error_kind=None, fix=None)
                await self._download(op)
                self._ops.update(op, state=STATE_DONE, progress=1.0,
                                 detail=("Verified" if op.meta.get("sha256") else "Downloaded") + f" · {_fmt_bytes(op.meta.get('size'))}")
                self._ops.save()
                if self._on_complete:
                    try:
                        r = self._on_complete(op, True)
                        if asyncio.iscoroutine(r):
                            await r
                    except Exception:
                        logger.debug("download on_complete failed", exc_info=True)
            except asyncio.CancelledError:
                mode = self._cancel.pop(op.id, None)
                if mode == "pause":
                    self._ops.update(op, state=STATE_PAUSED, detail=f"Paused · {_fmt_bytes(op.meta.get('bytes'))} of {_fmt_bytes(op.meta.get('size'))}")
                    self._ops.save()
                else:
                    self._finish_cancel(op)
            except DownloadError as e:
                self._ops.update(op, state=STATE_FAILED, error=str(e), error_kind=e.kind, fix=e.fix or {"action": "retry"}, detail=None)
                self._ops.save()
                if self._on_complete:
                    try:
                        r = self._on_complete(op, False)
                        if asyncio.iscoroutine(r):
                            await r
                    except Exception:
                        pass
            except Exception as e:
                logger.warning("download failed: %s", e, exc_info=True)
                self._ops.update(op, state=STATE_FAILED, error=str(e) or type(e).__name__, error_kind="other", fix={"action": "retry"}, detail=None)
                self._ops.save()
                if self._on_complete:
                    try:
                        r = self._on_complete(op, False)
                        if asyncio.iscoroutine(r):
                            await r
                    except Exception:
                        pass
            finally:
                self._tasks.pop(op.id, None)

    def _headers(self, url: str) -> dict:
        h = {"User-Agent": "ComfyUI-Stimma"}
        if "huggingface.co" in url:
            tok = credentials.hf_token()
            if tok:
                h["Authorization"] = f"Bearer {tok}"
        elif "civitai.com" in url:
            key = credentials.civitai_key()
            if key:
                h["Authorization"] = f"Bearer {key}"
        return h

    async def _download(self, op: Operation):
        meta = op.meta
        url = meta["url"]
        dest = Path(meta["dest_path"])
        part = Path(str(dest) + ".part")
        want_sha = (meta.get("sha256") or "").lower() or None
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Already there and (if we can) verified? Done.
        if dest.exists() and dest.stat().st_size > 0:
            if want_sha:
                self._ops.update(op, detail="Verifying")
                if await _sha256_file(dest) == want_sha:
                    meta["bytes"] = dest.stat().st_size
                    return
                # wrong bytes: replace it
                dest.unlink()
            else:
                meta["bytes"] = dest.stat().st_size
                return

        # Resume?
        have = part.stat().st_size if part.exists() else 0
        hasher = hashlib.sha256() if want_sha else None
        if have and hasher:
            self._ops.update(op, detail=f"Resuming · {_fmt_bytes(have)}")
            with open(part, "rb") as f:
                while True:
                    b = f.read(8 * _CHUNK)
                    if not b:
                        break
                    hasher.update(b)
                    await asyncio.sleep(0)

        session = await self._get_session()
        headers = self._headers(url)
        if have:
            headers["Range"] = f"bytes={have}-"

        async with session.get(url, headers=headers, allow_redirects=True) as resp:
            if resp.status in (401, 403):
                raise self._auth_error(resp.status, meta)
            if resp.status == 404:
                raise DownloadError("Not found at the download URL (404).", "not_found", {"action": "add_url"})
            if resp.status == 416:  # range not satisfiable → part is complete or bogus
                part.unlink(missing_ok=True)
                have = 0
                return await self._download(op)
            if resp.status not in (200, 206):
                raise DownloadError(f"HTTP {resp.status}.", "network", {"action": "retry"})
            if have and resp.status == 200:
                # Server ignored Range: start over
                have = 0
                hasher = hashlib.sha256() if want_sha else None
                part.unlink(missing_ok=True)

            total = None
            cl = resp.headers.get("Content-Length")
            if cl and cl.isdigit():
                total = int(cl) + have
            if total and not meta.get("size"):
                meta["size"] = total
            size = meta.get("size") or total

            # Disk preflight
            try:
                st = os.statvfs(str(dest.parent))
                free = st.f_bavail * st.f_frsize
                need = (size or 0) - have
                if need > 0 and free < need + 512 * 1024 * 1024:
                    raise DownloadError(
                        f"Not enough disk space: {_fmt_bytes(free)} free, {_fmt_bytes(need)} needed.", "disk", {"action": "retry"}
                    )
            except OSError:
                pass

            done = have
            t0 = time.time()
            last_ui = 0.0
            b0 = have
            mode = "ab" if have else "wb"
            with open(part, mode) as f:
                async for chunk in resp.content.iter_chunked(_CHUNK):
                    f.write(chunk)
                    if hasher:
                        hasher.update(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last_ui >= 0.5:
                        last_ui = now
                        elapsed = max(now - t0, 1e-6)
                        speed = (done - b0) / elapsed
                        eta = ((size - done) / speed) if (size and speed > 0) else None
                        meta["bytes"] = done
                        self._ops.update(
                            op,
                            progress=(done / size) if size else None,
                            detail=f"{_fmt_bytes(done)} of {_fmt_bytes(size)} · {_fmt_bytes(speed)}/s · {_fmt_eta(eta)}".rstrip(" ·"),
                        )
            meta["bytes"] = done

        if want_sha and hasher and hasher.hexdigest() != want_sha:
            part.unlink(missing_ok=True)
            raise DownloadError(
                "Checksum mismatch — downloaded file was discarded. If this repeats, update ComfyUI-Stimma before retrying.",
                "verify",
                {"action": "checksum"},
            )

        os.replace(part, dest)

    def _auth_error(self, status: int, meta: dict) -> DownloadError:
        url = meta.get("url", "")
        repo = meta.get("repo") or _repo_from_hf_url(url)
        license_url = meta.get("license_url") or (f"https://huggingface.co/{repo}" if repo else None)
        has_token = bool(credentials.hf_token()) if "huggingface.co" in url else bool(credentials.civitai_key())
        if "huggingface.co" in url:
            if not has_token:
                return DownloadError(
                    "Hugging Face token required.", "auth",
                    {"action": "hf_token", "license_url": license_url, "repo": repo},
                )
            if status == 401:
                return DownloadError(
                    "Hugging Face rejected the token (401).", "auth",
                    {"action": "hf_token", "license_url": license_url, "repo": repo},
                )
            return DownloadError(
                "Hugging Face 403 — model license not accepted for this token.",
                "gated", {"action": "hf_license", "license_url": license_url, "repo": repo},
            )
        return DownloadError(f"HTTP {status}.", "auth", {"action": "retry"})


def _repo_from_hf_url(url: str) -> Optional[str]:
    # https://huggingface.co/{owner}/{name}/resolve/{rev}/{path}
    try:
        if "huggingface.co/" not in url:
            return None
        tail = url.split("huggingface.co/", 1)[1]
        parts = tail.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    except Exception:
        pass
    return None


async def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 * _CHUNK)
            if not b:
                break
            h.update(b)
            await asyncio.sleep(0)
    return h.hexdigest()


async def probe_hf(url: str) -> dict:
    """HEAD a HF resolve URL with our token: {ok, status, gated, needs_token, size}."""
    headers = {"User-Agent": "ComfyUI-Stimma"}
    tok = credentials.hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.head(url, headers=headers, allow_redirects=True) as r:
                size = None
                cl = r.headers.get("Content-Length") or r.headers.get("X-Linked-Size")
                if cl and cl.isdigit():
                    size = int(cl)
                return {"ok": r.status == 200, "status": r.status, "gated": r.status == 403,
                        "needs_token": r.status == 401 or (r.status == 403 and not tok), "size": size}
    except Exception as e:
        return {"ok": False, "status": None, "error": str(e), "gated": False, "needs_token": False, "size": None}
