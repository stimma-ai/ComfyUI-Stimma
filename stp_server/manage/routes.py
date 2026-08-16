"""aiohttp routes for the management UI + API at /stp-v1/manage/."""

import json
import logging
import mimetypes
import os
from pathlib import Path

from aiohttp import web

from . import credentials, resolve, update as updater

logger = logging.getLogger(__name__)

_UI_DIR = Path(__file__).resolve().parent / "ui"
PREFIX = "/stp-v1/manage"


def _json(data, status=200):
    return web.json_response(data, status=status, dumps=lambda d: json.dumps(d, default=str))


def _err(msg, status=400, **extra):
    return _json({"error": msg, **extra}, status=status)


def _mutation_allowed(request: web.Request) -> bool:
    """CSRF guard for mutations (downloads, installs, restarts, credentials).

    ComfyUI itself has no auth: anyone who can reach the port can already run
    arbitrary workflows, so the manager is exactly as open as ComfyUI. What we
    do refuse is a *cross-origin browser* request — the /stp-v1/* CORS bypass
    (needed for asset PUTs) would otherwise let any web page CSRF a loopback
    restart or download. Same-origin sends Origin == our host (or no Origin);
    the Stimma app's proxy strips Origin and stamps X-Stimma-Manage.
    """
    origin = request.headers.get("Origin")
    if not origin:
        return True
    if request.headers.get("X-Stimma-Manage"):
        return True
    from urllib.parse import urlparse
    o = urlparse(origin)
    return (o.netloc or "").lower() == (request.host or "").lower()


def make_routes(manager) -> list:
    r = []

    # ---------------- static UI ----------------
    async def ui_index(request):
        return await _serve(_UI_DIR / "index.html")

    async def ui_static(request):
        rel = request.match_info.get("path", "")
        target = (_UI_DIR / rel).resolve()
        if not str(target).startswith(str(_UI_DIR.resolve())):
            raise web.HTTPNotFound()
        if target.is_dir():
            target = target / "index.html"
        return await _serve(target)

    async def _serve(path: Path):
        if not path.exists():
            # SPA fallback
            path = _UI_DIR / "index.html"
            if not path.exists():
                return web.Response(status=503, text="ComfyUI-Stimma manager UI is not built. Run `npm run build` in manage-ui/.")
        ctype, _ = mimetypes.guess_type(str(path))
        headers = {"Cache-Control": "no-cache"} if path.suffix in (".html",) else {"Cache-Control": "public, max-age=3600"}
        return web.FileResponse(path, headers={"Content-Type": ctype or "application/octet-stream", **headers})

    async def ui_redirect(request):
        raise web.HTTPFound(PREFIX + "/")

    r.append(web.get(PREFIX, ui_redirect))
    r.append(web.get(PREFIX + "/", ui_index))
    r.append(web.get(PREFIX + "/index.html", ui_index))

    # ---------------- API ----------------
    api = PREFIX + "/api"

    async def overview(request):
        return _json(await manager.overview())

    async def host(request):
        return _json(manager.host_stats())

    async def workflows(request):
        manager.instances.touch()
        return _json(manager.workflows_view())

    async def rescan(request):
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        try:
            changed = await manager.provider.discover_and_register_tools(force=True)
            if changed:
                await manager.provider.notify_tools_changed()
        except Exception as e:
            return _err(str(e), 500)
        return _json(manager.workflows_view())

    async def plan(request):
        slug = request.match_info["slug"]
        try:
            return _json(await manager.plan_setup(slug))
        except KeyError:
            return _err("unknown workflow", 404)

    async def setup(request):
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        slug = request.match_info["slug"]
        body = await _body(request)
        try:
            return _json(await manager.start_setup(slug, hf_token=body.get("hf_token"), extra_sources=body.get("sources")))
        except KeyError:
            return _err("unknown workflow", 404)

    async def activity(request):
        manager.instances.touch()
        return _json(manager.activity())

    async def op_action(request):
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        op_id = request.match_info["op_id"]
        action = request.match_info["action"]
        op = manager.ops.get(op_id)
        if not op:
            return _err("unknown operation", 404)
        if action == "retry":
            if op.kind == "download":
                manager.downloads.retry(op_id)
            elif op.kind == "install_node":
                import asyncio
                from .ops import STATE_QUEUED
                manager.ops.update(op, state=STATE_QUEUED, error=None, error_kind=None, fix=None)
                asyncio.create_task(manager._run_install(op))
            elif op.kind == "install_manager":
                import asyncio
                from .ops import STATE_QUEUED
                manager.ops.update(op, state=STATE_QUEUED, error=None, error_kind=None, fix=None)
                asyncio.create_task(manager._run_manager_install(op))
            else:
                return _err("not retryable")
        elif action == "pause":
            manager.downloads.pause(op_id)
        elif action == "cancel":
            manager.downloads.cancel(op_id)
        elif action == "dismiss":
            manager.dismiss_failure(op_id)
            manager.ops.remove(op_id)
            await manager.provider.push_state()
        else:
            return _err("unknown action")
        return _json({"ok": True, "operation": (manager.ops.get(op_id) or op).to_dict()})

    async def clear_done(request):
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        manager.ops.clear_done()
        await manager.provider.push_state()
        return _json({"ok": True})

    async def add_download(request):
        """Direct download (peer fan-out or a user-supplied URL)."""
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        b = await _body(request)
        fname, url = b.get("filename"), b.get("url")
        if not fname or not url:
            return _err("filename and url required")
        dest = b.get("dest_path") or resolve.dest_path_for(fname, b.get("folder"))
        if not dest:
            return _err("could not resolve destination")
        if b.get("remember"):
            resolve.remember_user_source(fname, url, b.get("folder"))
        op = manager.downloads.enqueue(
            filename=fname, url=url, dest_path=dest, size=b.get("size"), sha256=b.get("sha256"),
            gated=bool(b.get("gated")), license_url=b.get("license_url"), repo=b.get("repo"),
            group=b.get("group"), workflows=b.get("workflows") or [],
        )
        return _json({"ok": True, "operation": op.to_dict()})

    async def cancel_job(request):
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        b = await _body(request)
        ok = await manager.cancel_job(b.get("prompt_id", ""), b.get("addr"))
        return _json({"ok": ok})

    async def settings(request):
        return _json(manager.settings_view())

    async def install_manager(request):
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        return _json(await manager.start_manager_install())

    async def set_credentials(request):
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        b = await _body(request)
        if "huggingface_token" in b:
            credentials.set_hf_token(b.get("huggingface_token"))
        if "civitai_api_key" in b:
            credentials.set_civitai_key(b.get("civitai_api_key"))
        return _json(credentials.summary())

    async def restart(request):
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        b = await _body(request)
        if b.get("scope") == "self":
            return _json(manager.restart_self_only())
        return _json(await manager.restart())

    async def update_status(request):
        force = request.query.get("force") in ("1", "true")
        return _json(await updater.status(force=force))

    async def update_apply(request):
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        return _json(await manager.apply_update())

    async def restore_bundled(request):
        if not _mutation_allowed(request):
            return _err("forbidden", 403)
        from ..workflow_install import sync_bundled_workflows
        try:
            sync_bundled_workflows(restore_deleted=True)
        except TypeError:
            sync_bundled_workflows()
        await manager.provider.discover_and_register_tools(force=True)
        await manager.provider.notify_tools_changed()
        return _json({"ok": True})

    async def workflow_detail(request):
        slug = request.match_info["slug"]
        try:
            return _json(await manager.workflow_detail(slug))
        except KeyError:
            return _err("unknown workflow", 404)

    r += [
        web.get(api + "/overview", overview),
        web.get(api + "/host", host),
        web.get(api + "/workflows", workflows),
        web.post(api + "/workflows/rescan", rescan),
        web.get(api + "/workflows/{slug}/plan", plan),
        web.get(api + "/workflows/{slug}", workflow_detail),
        web.post(api + "/workflows/{slug}/setup", setup),
        web.get(api + "/activity", activity),
        web.post(api + "/activity/{op_id}/{action}", op_action),
        web.post(api + "/activity/clear-done", clear_done),
        web.post(api + "/downloads", add_download),
        web.post(api + "/jobs/cancel", cancel_job),
        web.get(api + "/settings", settings),
        web.post(api + "/manager/install", install_manager),
        web.post(api + "/settings/credentials", set_credentials),
        web.post(api + "/restart", restart),
        web.get(api + "/update", update_status),
        web.post(api + "/update", update_apply),
        web.post(api + "/workflows/restore-bundled", restore_bundled),
        # static assets last (catch-all under the prefix)
        web.get(PREFIX + "/{path:.*}", ui_static),
    ]
    return r


async def _body(request) -> dict:
    try:
        b = await request.json()
        return b if isinstance(b, dict) else {}
    except Exception:
        return {}
