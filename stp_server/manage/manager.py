"""Manager — the brain behind /stp-v1/manage.

Owns instance monitoring, the operations registry (Activity), downloads,
node installs, restart/update, and the workflow readiness view. Feeds
provider-level state + milestone notifications back over STP.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from . import credentials, jobs, nodes as nodepacks, resolve, update as updater
from .downloads import DownloadManager, probe_hf
from .instances import InstanceMonitor, disk_stats, is_local_addr
from .ops import OperationRegistry, Operation, STATE_QUEUED, STATE_RUNNING, STATE_DONE, STATE_FAILED, TERMINAL

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).resolve().parent.parent.parent
_STATE_DIR = _PLUGIN_DIR / ".stimma-manage"


def _fmt_gb(n: Optional[float]) -> str:
    if not n:
        return "?"
    gb = n / (1024 ** 3)
    return f"{gb:.0f} GB" if gb >= 10 else f"{gb:.1f} GB"


class Manager:
    def __init__(self, provider, config):
        self.provider = provider
        self.config = config
        self.ops = OperationRegistry(_STATE_DIR / "operations.json")
        self.downloads = DownloadManager(self.ops, on_complete=self._on_download_complete)
        self.instances = InstanceMonitor(provider.comfy_client, our_prompt_ids_fn=jobs.snapshot)
        self._workflows: List[Any] = []       # DiscoveredWorkflow list from last rebuild
        self._tools_by_slug: Dict[str, Any] = {}
        self._restart_needed: List[str] = []   # reasons
        self._dismissed_failures: set = set()
        self._started = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._log_lines: List[str] = []
        self.ops.on_change(self._on_op_change)
        self.instances.on_change(self._on_instances_change)

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        if self._started:
            return
        self._started = True
        self.instances.start()
        self.downloads.resume_all_queued()
        asyncio.create_task(self._background_update_check())

    async def stop(self):
        await self.instances.stop()
        await self.downloads.stop()
        if self._session and not self._session.closed:
            await self._session.close()

    async def _background_update_check(self):
        await asyncio.sleep(20)
        while True:
            try:
                await updater.status()
                await self.provider.push_state()
            except Exception:
                pass
            await asyncio.sleep(6 * 3600)

    def log(self, line: str):
        self._log_lines.append(f"{time.strftime('%H:%M:%S')} {line}")
        del self._log_lines[:-400]

    # ------------------------------------------------------------------ hooks
    def on_tools_rebuilt(self, workflows, tools, changed: bool):
        self._workflows = list(workflows)
        self._tools_by_slug = {t.slug: t for t in tools}
        # Any setup group whose ops are all done and whose workflow is now ready → notify.
        for w in self._workflows:
            slug = w.tool_info.get("slug")
            if not slug or w.warnings:
                continue
            group = f"setup:{slug}"
            group_ops = [o for o in self.ops.all() if o.group == group]
            if not group_ops:
                continue
            pending = [o for o in group_ops if not o.meta.get("notified")]
            if pending and all(o.state == STATE_DONE for o in group_ops):
                for o in group_ops:
                    o.meta["notified"] = True
                self.ops.save()
                asyncio.ensure_future(self.provider.notify(
                    id=f"workflow-ready:{slug}", level="info",
                    title=f"{w.tool_info.get('display_name') or slug} is ready",
                    body="Downloads finished and verified.", action="manage", anchor="workflows",
                ))
        asyncio.ensure_future(self.provider.push_state())

    async def _on_instances_change(self):
        s = self.instances.summary()
        self.log(f"instances: {s['healthy']}/{s['total']} healthy" + (f", down: {', '.join(s['down'])}" if s['down'] else ""))
        await self.provider.push_state()

    def _on_op_change(self, op: Operation):
        if op.state == STATE_FAILED and op.kind == "download":
            asyncio.ensure_future(self.provider.notify(
                id=f"download-failed:{op.id}", level="error",
                title="A download failed", body=op.error or op.title, action="manage", anchor="activity",
            ))
        if op.state in TERMINAL:
            asyncio.ensure_future(self.provider.push_state())

    async def _on_download_complete(self, op: Operation, ok: bool):
        if ok:
            self.log(f"downloaded {op.meta.get('filename')}")
            # Poke discovery so the tool flips to ready quickly (the file
            # watcher's fingerprint would catch it within its interval anyway).
            try:
                await self.provider.discover_and_register_tools()
                await self.provider.notify_tools_changed()
            except Exception:
                logger.debug("post-download rescan failed", exc_info=True)

    # ------------------------------------------------------------------ provider state
    def provider_state(self):
        s = self.instances.summary()
        if s["total"] and s["healthy"] == 0:
            return "error", "No ComfyUI instance is reachable"
        if s["total"] > 1 and s["healthy"] < s["total"]:
            n = s["total"] - s["healthy"]
            return "warning", f"{n} of {s['total']} instances unreachable"
        if self._restart_needed:
            return "warning", "Restart ComfyUI to finish setup"
        failed = [o for o in self.ops.all() if o.state == STATE_FAILED and o.id not in self._dismissed_failures]
        if failed:
            return "warning", "A download failed" if failed[0].kind == "download" else "An operation failed"
        return "ready", None

    # ------------------------------------------------------------------ overview
    async def overview(self) -> dict:
        self.instances.touch()
        st = self.instances.statuses
        # Group instances by host; local host gets nvidia-smi GPUs.
        hosts: Dict[str, dict] = {}
        for s in st:
            key = "local" if s.local else s.addr.rsplit(":", 1)[0]
            h = hosts.setdefault(key, {"host": key, "local": s.local, "instances": [], "gpus": [], "reachable": False})
            h["instances"].append(s.to_dict())
            if s.healthy:
                h["reachable"] = True
        if "local" in hosts:
            hosts["local"]["gpus"] = self.instances.gpus()
        # Remote hosts: derive GPU cards from /system_stats devices (no util) or peer manage API
        for key, h in hosts.items():
            if key == "local":
                continue
            peer = await self._peer_host_stats(h["instances"])
            if peer:
                h["gpus"] = peer.get("gpus", [])
            if not h["gpus"]:
                for inst in h["instances"]:
                    for d in inst.get("devices") or []:
                        if d.get("type") == "cuda" or "cuda" in str(d.get("name", "")).lower():
                            h["gpus"].append({"index": d.get("index"), "name": d.get("name"), "util": None,
                                              "mem_used": (d.get("vram_total") or 0) - (d.get("vram_free") or 0),
                                              "mem_total": d.get("vram_total"), "temp": None})
        running, pending = [], 0
        for s in st:
            for r in s.running:
                running.append({**r, "addr": s.addr})
            pending += s.pending
        # Provider-side queue (jobs accepted but not yet on a Comfy instance)
        try:
            qs = {"queued": len(self.provider._queued_jobs), "running": len(self.provider._running_jobs)}
        except Exception:
            qs = {"queued": 0, "running": 0}
        upd = await updater.status()
        state, summary = self.provider_state()
        return {
            "state": state, "summary": summary,
            "hosts": list(hosts.values()),
            "running": running, "pending": pending, "stp_queue": qs,
            "disk": disk_stats(resolve.model_dirs()),
            "plugin": upd,
            "restart_needed": list(self._restart_needed),
            "tools_ready": sum(1 for w in self._workflows if not w.warnings and w.tool_info.get("slug")),
            "instances_total": len(st), "instances_healthy": sum(1 for s in st if s.healthy),
        }

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6))
        return self._session

    async def _peer_host_stats(self, insts: list) -> Optional[dict]:
        for inst in insts:
            if not inst.get("healthy"):
                continue
            try:
                s = await self._get_session()
                async with s.get(f"http://{inst['addr']}/stp-v1/manage/api/host") as r:
                    if r.status == 200:
                        return await r.json()
            except Exception:
                continue
        return None

    def host_stats(self) -> dict:
        return {"gpus": self.instances.gpus(), "disk": disk_stats(resolve.model_dirs()),
                "model_dirs": resolve.model_dirs(), "hostname": os.uname().nodename if hasattr(os, "uname") else ""}

    # ------------------------------------------------------------------ jobs
    async def cancel_job(self, prompt_id: str, addr: Optional[str]) -> bool:
        rid = jobs.request_id_for(prompt_id)
        if rid:
            job = self.provider._running_jobs.get(rid)
            if job:
                job.cancelled = True
                if job.task and not job.task.done():
                    job.task.cancel()
                return True
        # Not ours (or unknown): interrupt / dequeue on the instance directly
        for inst in self.provider.comfy_client.instances:
            if addr and inst.addr != addr:
                continue
            try:
                s = await self._get_session()
                async with s.post(f"http://{inst.addr}/queue", json={"delete": [prompt_id]}) as r:
                    pass
                async with s.post(f"http://{inst.addr}/interrupt") as r:
                    pass
                return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------ workflows
    def _scan(self):
        from .. import discovery
        return discovery.LAST_SCAN

    def workflows_view(self) -> dict:
        scan = self._scan()
        items = []
        active_groups = {o.group for o in self.ops.active() if o.group}
        for w in self._workflows:
            slug = w.tool_info.get("slug") or ""
            if not slug:
                continue
            state = "ready" if not w.warnings else "needs_setup"
            summary = self._setup_summary(w)
            items.append({
                "slug": slug,
                "name": w.tool_info.get("display_name") or slug,
                "task_types": w.tool_info.get("task_types") or [],
                "file": os.path.basename(w.file_path),
                "path": w.file_path,
                "bundled": os.sep + "Stimma" + os.sep in w.file_path,
                "state": state,
                "issues": w.issues or [{"kind": "other", "name": x} for x in w.warnings],
                "summary": summary,
                "in_progress": f"setup:{slug}" in active_groups,
            })
        others = []
        if scan:
            for o in scan.others:
                others.append({
                    "file": os.path.basename(o.file_path), "path": o.file_path,
                    "state": "error" if o.error else ("incomplete" if o.has_stimma_nodes else "other"),
                    "detail": o.error or ("Has Stimma nodes but no StimmaToolInfo" if o.has_stimma_nodes else "Not a Stimma workflow"),
                })
            for slug, kept, skipped in scan.duplicates:
                others.append({"file": os.path.basename(skipped), "path": skipped, "state": "duplicate",
                               "detail": f"Duplicate slug '{slug}' — {os.path.basename(kept)} is used"})
        return {"tools": sorted(items, key=lambda i: (i["state"] != "ready", i["name"].lower())),
                "others": sorted(others, key=lambda i: i["file"].lower()),
                "directories": scan.directories if scan else []}

    def _setup_summary(self, w) -> Optional[str]:
        if not w.warnings:
            return None
        parts = []
        total = 0
        unresolved = 0
        gated = False
        node_count = 0
        for i in (w.issues or []):
            if i["kind"] == "missing_node":
                node_count += 1
            elif i["kind"] == "missing_model":
                src = resolve.resolve_source(i["name"], w.model_hints)
                if not src:
                    unresolved += 1
                else:
                    total += src.get("size") or 0
                    gated = gated or bool(src.get("gated"))
            elif i["kind"] == "missing_checkpoint":
                unresolved += 1
        if total:
            parts.append(f"{_fmt_gb(total)} to download")
        elif any(i["kind"] == "missing_model" for i in (w.issues or [])) and not unresolved:
            parts.append("downloads")
        if gated:
            parts.append("Hugging Face license")
        if node_count:
            parts.append(f"{node_count} node pack{'s' if node_count != 1 else ''}")
        if unresolved:
            parts.append("no download source known" if unresolved == 1 else f"{unresolved} files without a source")
        return " · ".join(parts) or "needs setup"

    def _workflow(self, slug: str):
        for w in self._workflows:
            if w.tool_info.get("slug") == slug:
                return w
        return None

    async def plan_setup(self, slug: str) -> dict:
        w = self._workflow(slug)
        if w is None:
            raise KeyError(slug)
        downloads, packs, blockers = [], [], []
        seen_files = set()
        for i in (w.issues or []):
            if i["kind"] == "missing_model":
                fname = i["name"]
                if fname in seen_files:
                    continue
                seen_files.add(fname)
                src = resolve.resolve_source(fname, w.model_hints)
                # Live COMBO→folder match is ground truth; manifest/hint is the fallback.
                folder = i.get("folder") or (src or {}).get("folder")
                dest = resolve.dest_path_for(fname, folder)
                entry = {"filename": fname, "folder": folder, "dest_path": dest, "resolved": bool(src),
                         **({k: src.get(k) for k in ("url", "size", "sha256", "gated", "license_url", "repo", "via")} if src else {})}
                if dest and os.path.exists(dest):
                    entry["already_present"] = True
                downloads.append(entry)
                if not src:
                    blockers.append({"kind": "no_source", "filename": fname, "folder": folder})
            elif i["kind"] == "missing_checkpoint":
                blockers.append({"kind": "no_source", "filename": f"a checkpoint matching {i['name']}", "folder": "checkpoints"})
            elif i["kind"] == "missing_node":
                pack = await nodepacks.lookup_pack(i["name"])
                if pack:
                    if not any(p["url"] == pack["url"] for p in packs):
                        packs.append({"class_type": i["name"], "url": pack["url"], "title": pack["title"],
                                      "installed": nodepacks.pack_installed(pack["url"]),
                                      "installable": nodepacks.has_manager(),
                                      "manual": nodepacks.manual_instructions(pack["url"])})
                else:
                    blockers.append({"kind": "unknown_node", "class_type": i["name"]})
        # Probe HF for gated / token needs on the resolved HF downloads
        hf_needs_token, hf_license = False, []
        probes = [d for d in downloads if d.get("resolved") and "huggingface.co" in (d.get("url") or "") and not d.get("already_present")]
        if probes:
            results = await asyncio.gather(*[probe_hf(d["url"]) for d in probes], return_exceptions=True)
            for d, r in zip(probes, results):
                if isinstance(r, Exception) or not isinstance(r, dict):
                    continue
                if r.get("size") and not d.get("size"):
                    d["size"] = r["size"]
                d["probe"] = r
                if r.get("needs_token"):
                    hf_needs_token = True
                elif r.get("gated"):
                    hf_license.append({"repo": d.get("repo"), "license_url": d.get("license_url") or (f"https://huggingface.co/{d['repo']}" if d.get("repo") else None)})
        if hf_needs_token:
            blockers.append({"kind": "hf_token", "repos": sorted({d.get("repo") for d in probes if d.get("probe", {}).get("needs_token") and d.get("repo")}),
                             "license_url": next((d.get("license_url") or f"https://huggingface.co/{d['repo']}" for d in probes if d.get("probe", {}).get("needs_token") and d.get("repo")), None)})
        for lic in hf_license:
            if not any(b.get("kind") == "hf_license" and b.get("repo") == lic["repo"] for b in blockers):
                blockers.append({"kind": "hf_license", **lic})
        if any(p for p in packs if not p["installed"] and not p["installable"]):
            blockers.append({"kind": "no_manager", "packs": [p for p in packs if not p["installed"] and not p["installable"]]})
        total = sum((d.get("size") or 0) for d in downloads if not d.get("already_present"))
        free = None
        for d in downloads:
            if d.get("dest_path"):
                try:
                    st = os.statvfs(os.path.dirname(d["dest_path"]) if os.path.isdir(os.path.dirname(d["dest_path"])) else resolve.model_dirs()[0])
                    free = st.f_bavail * st.f_frsize
                    break
                except Exception:
                    pass
        return {
            "slug": slug, "name": w.tool_info.get("display_name") or slug,
            "downloads": downloads, "packs": packs, "blockers": blockers,
            "total_size": total, "free_space": free,
            "targets": self._download_targets(),
            "hf_token_set": bool(credentials.hf_token()),
            "in_progress": any(o.group == f"setup:{slug}" for o in self.ops.active()),
        }

    def _download_targets(self) -> List[str]:
        """Distinct hosts that need their own copy (local + each remote host)."""
        hosts = ["local"]
        for s in self.instances.statuses:
            if not s.local:
                h = s.addr.rsplit(":", 1)[0]
                if h not in hosts:
                    hosts.append(h)
        return hosts

    async def start_setup(self, slug: str, hf_token: Optional[str] = None, extra_sources: Optional[dict] = None) -> dict:
        if hf_token:
            credentials.set_hf_token(hf_token)
        if extra_sources:
            for fname, url in extra_sources.items():
                if url:
                    resolve.remember_user_source(fname, url, None)
        plan = await self.plan_setup(slug)
        group = f"setup:{slug}"
        queued = []
        for d in plan["downloads"]:
            if d.get("already_present") or not d.get("resolved") or not d.get("dest_path"):
                continue
            op = self.downloads.enqueue(
                filename=d["filename"], url=d["url"], dest_path=d["dest_path"], size=d.get("size"),
                sha256=d.get("sha256"), gated=bool(d.get("gated")), license_url=d.get("license_url"),
                repo=d.get("repo"), group=group, workflows=[slug],
            )
            queued.append(op.id)
            await self._fanout_download(d, slug)
        for p in plan["packs"]:
            if p["installed"] or not p["installable"]:
                continue
            existing = self.ops.find("install_node", url=p["url"])
            if existing:
                queued.append(existing.id)
                continue
            op = self.ops.create("install_node", f"Install {p['title']}", meta={"url": p["url"], "workflows": [slug]}, group=group)
            queued.append(op.id)
            asyncio.create_task(self._run_install(op))
        self.log(f"setup started for {slug}: {len(queued)} operation(s)")
        return {"queued": queued, "plan": plan}

    async def _fanout_download(self, d: dict, slug: str):
        """Ask remote-host peers (other machines running the plugin) to fetch the same file."""
        for s in self.instances.statuses:
            if s.local or not s.healthy:
                continue
            host = s.addr.rsplit(":", 1)[0]
            try:
                sess = await self._get_session()
                async with sess.post(f"http://{s.addr}/stp-v1/manage/api/downloads", json={
                    "filename": d["filename"], "url": d["url"], "folder": d.get("folder"), "size": d.get("size"),
                    "sha256": d.get("sha256"), "gated": d.get("gated"), "license_url": d.get("license_url"),
                    "repo": d.get("repo"), "group": f"setup:{slug}", "workflows": [slug],
                }) as r:
                    if r.status != 200:
                        self.log(f"peer {host} declined download {d['filename']}: HTTP {r.status}")
            except Exception as e:
                self.log(f"peer {host} unreachable for download fan-out: {e}")

    async def _run_install(self, op: Operation):
        self.ops.update(op, state=STATE_RUNNING, detail="Installing…")
        lines: List[str] = []

        def _log(line: str):
            lines.append(line)
            del lines[:-50]
            self.log(f"[install {op.meta.get('url','')}] {line}")
            self.ops.update(op, detail=line[:120])

        try:
            await nodepacks.install_pack(op.meta["url"], _log)
            self._restart_needed.append(f"install:{op.meta['url']}")
            self.ops.update(op, state=STATE_DONE, detail="Installed · restart ComfyUI to load it")
            self.ops.save()
            await self.provider.notify(id="restart-needed", level="warning", title="Restart ComfyUI to finish",
                                       body="Node packs were installed.", action="manage", anchor="overview")
            await self.provider.push_state(force=True)
        except Exception as e:
            self.ops.update(op, state=STATE_FAILED, error=str(e), error_kind="other",
                            fix={"action": "manual", "command": nodepacks.manual_instructions(op.meta["url"])})
            self.ops.save()
            self.log(f"install failed: {e}")

    # ------------------------------------------------------------------ activity
    def activity(self) -> dict:
        return {"operations": [o.to_dict() for o in self.ops.all()],
                "restart_needed": list(self._restart_needed)}

    # ------------------------------------------------------------------ restart / update
    async def restart(self, scope: str = "all") -> dict:
        """Restart ComfyUI. Peers first (via their manage API), then ourselves via execv."""
        op = self.ops.create("restart", "Restart ComfyUI", meta={"scope": scope})
        self.ops.update(op, state=STATE_RUNNING, detail="Restarting…")
        peers = []
        for s in self.instances.statuses:
            # Skip ourselves: the instance whose port matches this process
            if self._is_self(s.addr):
                continue
            peers.append(s.addr)
        for addr in peers:
            try:
                sess = await self._get_session()
                async with sess.post(f"http://{addr}/stp-v1/manage/api/restart", json={"scope": "self"}) as r:
                    self.log(f"restart {addr}: HTTP {r.status}")
            except Exception as e:
                self.log(f"restart {addr}: {e}")
        self._restart_needed.clear()
        self.ops.update(op, state=STATE_DONE, detail="Restarted")
        self.ops.save()
        asyncio.get_event_loop().call_later(0.6, self._execv_self)
        return {"ok": True, "peers": peers}

    def restart_self_only(self) -> dict:
        self._restart_needed.clear()
        asyncio.get_event_loop().call_later(0.4, self._execv_self)
        return {"ok": True}

    def _is_self(self, addr: str) -> bool:
        try:
            from server import PromptServer
            port = int(PromptServer.instance.port)
        except Exception:
            return False
        return is_local_addr(addr) and addr.endswith(f":{port}")

    def _execv_self(self):
        self.log("exec restart")
        try:
            sys.stdout.flush(); sys.stderr.flush()
        except Exception:
            pass
        argv = [sys.executable] + sys.argv
        if sys.platform == "win32":
            # execv on Windows spawns detached; use the same trick ComfyUI-Manager does.
            import subprocess
            subprocess.Popen(argv, cwd=os.getcwd())
            os._exit(0)
        os.execv(sys.executable, argv)

    async def apply_update(self) -> dict:
        op = self.ops.create("update", "Update ComfyUI-Stimma")
        self.ops.update(op, state=STATE_RUNNING, detail="Updating…")
        try:
            await updater.apply_update(lambda l: self.ops.update(op, detail=str(l)[:120]))
            self._restart_needed.append("update")
            self.ops.update(op, state=STATE_DONE, detail="Updated · restart ComfyUI to load it")
            self.ops.save()
            await self.provider.push_state(force=True)
            return {"ok": True, "restart_needed": True}
        except Exception as e:
            self.ops.update(op, state=STATE_FAILED, error=str(e), error_kind="other", fix={"action": "retry"})
            self.ops.save()
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------ settings
    def settings_view(self) -> dict:
        return {
            "credentials": credentials.summary(),
            "instances": [s.to_dict() for s in self.instances.statuses],
            "manager_present": nodepacks.has_manager(),
            "config_path": str(credentials.CONFIG_PATH),
        }

    def dismiss_failure(self, op_id: str):
        self._dismissed_failures.add(op_id)

    def scan_report(self) -> dict:
        scan = self._scan()
        if not scan:
            return {"tools": [], "others": [], "directories": []}
        return {
            "directories": scan.directories,
            "tools": [{"slug": w.tool_info.get("slug"), "file": w.file_path, "warnings": w.warnings} for w in scan.tools],
            "others": [{"file": o.file_path, "has_stimma_nodes": o.has_stimma_nodes, "error": o.error} for o in scan.others],
            "duplicates": [list(d) for d in scan.duplicates],
        }

    def recent_log(self) -> List[str]:
        return list(self._log_lines)
