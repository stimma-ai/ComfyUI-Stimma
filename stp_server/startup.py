"""STP server startup — hooks into ComfyUI's PromptServer.

All routes are added during plugin import (before the aiohttp app starts).
The provider itself runs as a background asyncio task on ComfyUI's event loop.
"""

import asyncio
import glob
import logging
import os
import shutil
import sys
import tempfile
from typing import Optional

from .config import Config, load_config
from .comfy_client import Comfy, parse_addresses
from .workflow_install import sync_bundled_workflows

logger = logging.getLogger(__name__)

_provider = None


def _detect_comfyui_address() -> str:
    """Auto-detect the ComfyUI server address from the running process."""
    try:
        from server import PromptServer
        instance = PromptServer.instance
        if instance and hasattr(instance, "port"):
            host = getattr(instance, "address", "127.0.0.1") or "127.0.0.1"
            return f"{host}:{instance.port}"
    except (ImportError, AttributeError):
        pass

    # Fallback: parse from sys.argv
    port = "8188"
    host = "127.0.0.1"
    args = sys.argv
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            port = args[i + 1]
        elif arg == "--listen" and i + 1 < len(args):
            listen_addr = args[i + 1]
            if listen_addr and listen_addr != "0.0.0.0":
                host = listen_addr

    return f"{host}:{port}"


def _get_comfyui_addresses(config: Config) -> list:
    """Get ComfyUI addresses from config or auto-detect."""
    if config.comfyui.addresses:
        return parse_addresses(config.comfyui.addresses)
    addr = _detect_comfyui_address()
    logger.info(f"Auto-detected ComfyUI address: {addr}")
    return [addr]


_STP_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, PUT, HEAD, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Expose-Headers": "ETag",
    "Access-Control-Max-Age": "3600",
}


def _make_stp_cors_bypass_middleware():
    """
    Build a middleware that runs ahead of ComfyUI's `origin_only` check for
    our `/stp-v1/*` paths only. Other paths pass through to the rest of the
    chain unchanged, so ComfyUI's CSRF / origin protection still applies
    to ComfyUI's own endpoints.

    Why this is needed: ComfyUI's default middleware returns 403 on any
    cross-origin request, which kills the Stimma renderer's direct PUT
    to /stp-v1/assets/<id> from `tauri://localhost` to a LAN ComfyUI
    host like `gpu-box:8188`. Without this bypass users would have
    to start ComfyUI with `--enable-cors-header "*"` — a footgun we
    avoid by handling CORS entirely inside the plugin.

    The bypass is scoped to /stp-v1/*. STP's own authentication is
    enforced at the WebSocket handshake, not on individual asset PUTs;
    anyone who can reach the host can already drop bytes via WS, so
    opening the asset endpoint is consistent with the existing trust
    model.
    """
    from aiohttp import web

    @web.middleware
    async def _stp_cors_bypass(request, handler):
        if not request.path.startswith("/stp-v1/"):
            return await handler(request)

        # CORS preflight — answer directly, no further middleware needed.
        if request.method == "OPTIONS":
            return web.Response(status=204, headers=_STP_CORS_HEADERS)

        # Resolve the matched route and invoke its handler directly,
        # bypassing the remainder of the middleware chain (which is what
        # would otherwise apply ComfyUI's origin_only 403 to us).
        match_info = await request.app.router.resolve(request)
        if match_info.http_exception is not None:
            raise match_info.http_exception
        response = await match_info.handler(request)
        for k, v in _STP_CORS_HEADERS.items():
            response.headers[k] = v
        return response

    return _stp_cors_bypass


def setup_stp_server():
    """Hook STP routes into ComfyUI's PromptServer.

    Must be called during plugin import (before the aiohttp app starts).
    Adds the WebSocket endpoint and asset routes, then schedules
    the provider to start once the HTTP server is running.
    """
    try:
        from server import PromptServer
        app = PromptServer.instance.app
    except (ImportError, AttributeError) as e:
        logger.warning(f"ComfyUI PromptServer not available, STP disabled: {e}")
        return

    # Prepend the CORS-bypass middleware so it runs ahead of ComfyUI's
    # origin_only check on /stp-v1/* paths. Must happen before the app
    # starts (middlewares freeze after on_startup). aiohttp stores the
    # mutable list as `app._middlewares` even after wrapping it as a
    # FrozenList; `insert(0, ...)` works while the app hasn't started.
    try:
        app.middlewares.insert(0, _make_stp_cors_bypass_middleware())
        logger.info("Installed STP CORS-bypass middleware ahead of ComfyUI's chain")
    except Exception as e:
        logger.warning(f"Could not install STP CORS-bypass middleware: {e}")

    # Increase body size limit for large asset uploads (LoRAs can be 500MB+)
    # ComfyUI defaults to 100MB which is too small
    _STP_MAX_UPLOAD = 4 * 1024 * 1024 * 1024  # 4GB
    if hasattr(app, "_client_max_size") and app._client_max_size < _STP_MAX_UPLOAD:
        old_limit = app._client_max_size
        app._client_max_size = _STP_MAX_UPLOAD
        logger.info(f"Increased ComfyUI body size limit: {old_limit} -> {_STP_MAX_UPLOAD} bytes")

    from .transport import ComfyUITransport
    from stimma_tools_protocol.assets import LocalAssetServer

    config = load_config()

    # Install bundled workflows into ComfyUI's user workflow directory
    try:
        sync_bundled_workflows()
    except Exception as e:
        logger.warning(f"Failed to sync bundled workflows: {e}")

    # Create transport — adds /stp-v1 WebSocket route to ComfyUI's app
    transport = ComfyUITransport(app)

    # Clean up stale asset temp dirs from previous sessions
    for old_dir in glob.glob(os.path.join(tempfile.gettempdir(), "stimma-provider-assets-*")):
        try:
            shutil.rmtree(old_dir)
            logger.info(f"Cleaned up stale asset dir: {old_dir}")
        except OSError:
            pass

    # Privacy: sweep orphaned job uploads from ComfyUI's input directory.
    # The executor deletes its uploads after each job, but a hard crash can
    # leave them behind. 24h threshold — jobs can legitimately run for hours.
    try:
        import time as _time
        import folder_paths
        input_dir = folder_paths.get_input_directory()
        cutoff = _time.time() - 24 * 3600
        swept = 0
        for pattern in ("stimma_upload_*", "comfy_input_*"):
            for path in glob.glob(os.path.join(input_dir, pattern)):
                try:
                    if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        swept += 1
                except OSError:
                    pass
        if swept:
            logger.info(f"Swept {swept} orphaned Stimma upload(s) from ComfyUI input dir")
    except Exception as e:
        logger.warning(f"Stale upload sweep failed: {e}")

    # Create asset server and add routes to ComfyUI's app
    asset_server = LocalAssetServer()
    transport.add_routes(asset_server.get_aiohttp_routes("/stp-v1/assets"))

    logger.info("STP routes added to ComfyUI server")

    # Schedule provider startup after the HTTP server is running
    async def _on_startup(app):
        asyncio.create_task(_run_provider(config, transport, asset_server))

    app.on_startup.append(_on_startup)


async def _run_provider(config: Config, transport, asset_server):
    """Background task: start and run the STP provider on ComfyUI's event loop."""
    global _provider

    from .provider import StimmaPluginProvider

    # Set up ComfyUI client
    addresses = _get_comfyui_addresses(config)
    comfy_client = Comfy(addresses)

    provider = StimmaPluginProvider(
        config=config,
        transport=transport,
        comfy_client=comfy_client,
    )
    _provider = provider

    try:
        await provider.start(asset_manager=asset_server)
        logger.info("STP provider running on ComfyUI's server")
        await provider.run()
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.exception(f"STP provider error: {e}")
    finally:
        try:
            await provider.stop()
        except Exception:
            pass
        try:
            asset_server.cleanup()
        except Exception:
            pass
        logger.info("STP provider stopped")
