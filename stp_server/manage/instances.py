"""ComfyUI instance liveness, queue view, and host GPU/disk stats."""

import asyncio
import json
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


def _local_host_names() -> set:
    names = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}
    try:
        hn = socket.gethostname()
        names.add(hn)
        names.add(hn.split(".")[0])
        try:
            for info in socket.getaddrinfo(hn, None):
                names.add(info[4][0])
        except socket.gaierror:
            pass
    except Exception:
        pass
    return names


def _addr_host(addr: str) -> str:
    return addr.rsplit(":", 1)[0] if ":" in addr else addr


def is_local_addr(addr: str, _cache: dict = {}) -> bool:
    if "names" not in _cache:
        _cache["names"] = _local_host_names()
    return _addr_host(addr) in _cache["names"]


@dataclass
class InstanceStatus:
    addr: str
    local: bool
    healthy: bool = False
    error: Optional[str] = None
    last_seen: Optional[float] = None
    devices: List[Dict[str, Any]] = field(default_factory=list)  # from /system_stats
    running: List[Dict[str, Any]] = field(default_factory=list)  # [{prompt_id, ours, title}]
    pending: int = 0
    comfy_version: Optional[str] = None
    stimma_manage: bool = False  # peer plugin's manage API reachable

    def to_dict(self) -> dict:
        return {
            "addr": self.addr,
            "host": _addr_host(self.addr),
            "local": self.local,
            "healthy": self.healthy,
            "error": self.error,
            "last_seen": self.last_seen,
            "devices": self.devices,
            "running": self.running,
            "pending": self.pending,
            "comfy_version": self.comfy_version,
        }


def gpu_stats() -> List[Dict[str, Any]]:
    """Local GPUs via nvidia-smi (empty list when unavailable)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4,
        )
        if out.returncode != 0:
            return []
    except Exception:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "util": float(parts[3]) if parts[3] not in ("[N/A]", "") else None,
                # GB10 reports utilization and temperature through nvidia-smi,
                # but unified memory is [N/A]. Keep the GPU and let ComfyUI's
                # /system_stats fill the memory fields below.
                "mem_used": int(float(parts[4])) * 1024 * 1024 if parts[4] not in ("[N/A]", "") else None,
                "mem_total": int(float(parts[5])) * 1024 * 1024 if parts[5] not in ("[N/A]", "") else None,
                "temp": float(parts[6]) if parts[6] not in ("[N/A]", "") else None,
            })
        except ValueError:
            continue
    return gpus


def merge_comfy_gpu_memory(gpus: List[Dict[str, Any]], devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill missing GPU memory from ComfyUI's /system_stats device records.

    NVIDIA unified-memory systems such as GB10 intentionally return [N/A] for
    nvidia-smi's framebuffer-memory query, while ComfyUI reports the shared
    allocation as vram_total/vram_free. Dedicated-memory GPUs keep the more
    precise nvidia-smi values.
    """
    result = [dict(g) for g in gpus]
    by_index = {g.get("index"): g for g in result}
    for device in devices:
        if device.get("type") != "cuda":
            continue
        index = device.get("index")
        gpu = by_index.get(index)
        if gpu is None:
            gpu = {
                "index": index,
                "uuid": None,
                "name": str(device.get("name") or f"CUDA {index}"),
                "util": None,
                "mem_used": None,
                "mem_total": None,
                "temp": None,
            }
            result.append(gpu)
            by_index[index] = gpu
        total = device.get("vram_total")
        free = device.get("vram_free")
        if not gpu.get("mem_total") and isinstance(total, (int, float)) and total > 0:
            gpu["mem_total"] = int(total)
            if isinstance(free, (int, float)):
                gpu["mem_used"] = max(0, int(total - free))
            gpu["unified_memory"] = True
    return result


def disk_stats(paths: List[str]) -> List[Dict[str, Any]]:
    """Free/total for each distinct filesystem behind the given paths."""
    seen = set()
    result = []
    for p in paths:
        try:
            st = os.statvfs(p)
        except OSError:
            continue
        key = (st.f_fsid if hasattr(st, "f_fsid") else p)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "path": p,
            "free": st.f_bavail * st.f_frsize,
            "total": st.f_blocks * st.f_frsize,
        })
    return result


class InstanceMonitor:
    """Polls every configured ComfyUI instance for liveness + queue."""

    def __init__(self, comfy_client, our_prompt_ids_fn=None):
        self._client = comfy_client
        self._statuses: Dict[str, InstanceStatus] = {
            inst.addr: InstanceStatus(addr=inst.addr, local=is_local_addr(inst.addr))
            for inst in comfy_client.instances
        }
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._interval_idle = 5.0
        self._interval_active = 2.0
        self._last_interest = 0.0
        self._listeners = []
        self._our_prompt_ids_fn = our_prompt_ids_fn or (lambda: {})
        self._gpus: List[Dict[str, Any]] = []
        self._gpus_at = 0.0

    # -- lifecycle --
    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._session and not self._session.closed:
            await self._session.close()

    def on_change(self, cb):
        self._listeners.append(cb)

    def touch(self):
        """A UI is watching — poll faster for a while."""
        self._last_interest = time.time()

    @property
    def statuses(self) -> List[InstanceStatus]:
        return list(self._statuses.values())

    def summary(self) -> dict:
        st = self.statuses
        up = [s for s in st if s.healthy]
        return {"total": len(st), "healthy": len(up), "down": [s.addr for s in st if not s.healthy]}

    def gpus(self) -> List[Dict[str, Any]]:
        if time.time() - self._gpus_at > 2.0:
            self._gpus = gpu_stats()
            self._gpus_at = time.time()
        return self._gpus

    # -- polling --
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4))
        return self._session

    async def poll_once(self) -> bool:
        """Poll all instances. Returns True when any health flag flipped."""
        changed = False
        results = await asyncio.gather(*[self._poll(addr) for addr in self._statuses], return_exceptions=True)
        for addr, res in zip(list(self._statuses), results):
            if res is True:
                changed = True
        # Reflect liveness into the load balancer
        by_addr = {i.addr: i for i in self._client.instances}
        for addr, st in self._statuses.items():
            inst = by_addr.get(addr)
            if inst is not None:
                inst.healthy = st.healthy
        if changed:
            for cb in list(self._listeners):
                try:
                    r = cb()
                    if asyncio.iscoroutine(r):
                        await r
                except Exception:
                    logger.debug("instance monitor listener failed", exc_info=True)
        return changed

    async def _poll(self, addr: str) -> bool:
        st = self._statuses[addr]
        was = st.healthy
        session = await self._get_session()
        try:
            async with session.get(f"http://{addr}/system_stats") as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                stats = json.loads(await r.read())
            st.devices = stats.get("devices") or []
            st.comfy_version = (stats.get("system") or {}).get("comfyui_version")
            async with session.get(f"http://{addr}/queue") as r:
                q = json.loads(await r.read()) if r.status == 200 else {}
            ours = self._our_prompt_ids_fn()
            running = []
            for item in q.get("queue_running") or []:
                try:
                    pid = item[1]
                    prompt = item[2] if len(item) > 2 else {}
                except Exception:
                    continue
                info = ours.get(pid)
                running.append({
                    "prompt_id": pid,
                    "ours": info is not None,
                    "title": (info or {}).get("title") or _title_from_prompt(prompt),
                    "request_id": (info or {}).get("request_id"),
                    "started_at": (info or {}).get("started_at"),
                    "progress": (info or {}).get("progress"),
                })
            st.running = running
            st.pending = len(q.get("queue_pending") or [])
            st.healthy = True
            st.error = None
            st.last_seen = time.time()
        except Exception as e:
            st.healthy = False
            st.error = _short_err(e)
            st.running = []
            st.pending = 0
        return was != st.healthy

    async def _loop(self):
        # First poll immediately so state is right at startup
        try:
            await self.poll_once()
        except Exception:
            logger.debug("initial instance poll failed", exc_info=True)
        while True:
            active = (time.time() - self._last_interest) < 30
            await asyncio.sleep(self._interval_active if active else self._interval_idle)
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("instance poll failed", exc_info=True)


def _short_err(e: Exception) -> str:
    if isinstance(e, (aiohttp.ClientConnectorError, ConnectionRefusedError)):
        return "connection refused"
    if isinstance(e, asyncio.TimeoutError):
        return "timed out"
    return str(e) or type(e).__name__


def _title_from_prompt(prompt: Any) -> str:
    """Best-effort label for a job that isn't ours (started from Comfy's UI)."""
    if not isinstance(prompt, dict):
        return "ComfyUI job"
    for node in prompt.values():
        if isinstance(node, dict) and node.get("class_type") == "StimmaToolInfo":
            name = (node.get("inputs") or {}).get("display_name")
            if name:
                return str(name)
    return "ComfyUI job"
