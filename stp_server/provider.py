"""StimmaPluginProvider — STP provider that serves discovered ComfyUI workflows."""

import asyncio
import logging
import os
from typing import Optional, Dict, Any, List

from stimma_tools_protocol.provider import Provider, ProviderConfig
from stimma_tools_protocol.protocol import JsonRpcResponse
from stimma_tools_protocol.tool import ToolRegistry
from stimma_tools_protocol.transport import Transport

from .config import Config
from .comfy_client import Comfy
from .version import PRODUCT_NAME, PRODUCT_VERSION

logger = logging.getLogger(__name__)


class StimmaPluginProvider(Provider):
    """STP provider for ComfyUI-Stimma.

    Manages dynamic tool discovery, LoRA enumeration, and workflow execution.
    """

    def __init__(
        self,
        config: Config,
        transport: Transport,
        comfy_client: Comfy,
    ):
        max_concurrent = max(1, len(comfy_client.instances))
        provider_config = ProviderConfig(
            provider_id=config.provider.id,
            provider_name=config.provider.name,
            server=f"{PRODUCT_NAME}/{PRODUCT_VERSION}",
            max_concurrent=max_concurrent,
            supports_cancel=True,
        )
        self._tool_registry = ToolRegistry()
        super().__init__(provider_config, transport, tool_registry=self._tool_registry)

        self._plugin_config = config
        self._comfy_client = comfy_client
        self._object_info: Optional[Dict[str, Any]] = None
        self._discovered_workflows: Dict[str, Any] = {}  # slug → DiscoveredWorkflow
        self._previous_slugs: set = set()
        self._watcher_task: Optional[asyncio.Task] = None
        self._workflow_snapshot: Dict[str, float] = {}  # filepath → mtime
        self._deps_snapshot: Optional[str] = None  # hash of models + custom_nodes dirs
        self._initial_discovery_done = False
        self._pending_uploads: Dict[str, dict] = {}  # upload_id → {asset_id, filename, subdir}

    @property
    def plugin_config(self) -> Config:
        return self._plugin_config

    @property
    def comfy_client(self) -> Comfy:
        return self._comfy_client

    @property
    def object_info(self) -> Optional[Dict[str, Any]]:
        return self._object_info

    def _ensure_watcher_started(self):
        """Start the filesystem watcher once."""
        if self._watcher_task is None or self._watcher_task.done():
            self._watcher_task = asyncio.create_task(self._watch_workflows())

    async def start(self, asset_manager=None) -> None:
        """Start provider and begin background filesystem watching immediately."""
        await super().start(asset_manager=asset_manager)
        self._ensure_watcher_started()

    async def stop(self) -> None:
        """Stop provider and cancel background filesystem watching."""
        if self._watcher_task and not self._watcher_task.done():
            self._watcher_task.cancel()
            try:
                await self._watcher_task
            except asyncio.CancelledError:
                pass
        await super().stop()

    async def discover_and_register_tools(self):
        """Scan for workflows, build tools, and register them."""
        # Import here to avoid circular imports
        from .discovery import discover_workflows
        from .tool_builder import build_tools_from_workflows

        # Refresh object_info for dynamic enums (samplers, schedulers, LoRAs)
        try:
            self._object_info = await self._comfy_client.get_object_info()
        except Exception as e:
            logger.error(
                f"\033[31m\u2718 Failed to fetch ComfyUI object_info: {e}\033[0m"
            )
            if self._object_info is None:
                self._object_info = {}

        # Discover workflows
        workflows = discover_workflows(self._plugin_config, object_info=self._object_info)
        self._discovered_workflows = {w.tool_info["slug"]: w for w in workflows if w.tool_info.get("slug")}

        # Only register workflows that validated cleanly — skip any with missing
        # nodes or models so the user knows to install them.
        registerable = [w for w in workflows if not w.warnings]

        # Build Tool objects
        tools = build_tools_from_workflows(
            registerable,
            self._object_info,
            self._plugin_config,
            self,
        )

        # Register
        self._tool_registry.clear()
        for tool in tools:
            self._tool_registry.register(tool)

        # Check if tool list changed
        current_slugs = {t.slug for t in tools}
        if current_slugs != self._previous_slugs:
            self._previous_slugs = current_slugs
            return True  # Tools changed
        return False

    async def _handle_tools_list(self, request):
        """Re-enumerate tools, LoRAs, and property values on every tools.list.

        Stimma sends exactly one tools.list immediately after each
        (re)connection handshake, so re-running discovery here guarantees a
        freshly-connected client always sees the *current* LoRA list and
        dynamic property enums (samplers, schedulers, dropdowns) — never a
        stale catalog cached from an earlier session. This is what makes a
        LoRA copied onto the box after ComfyUI started show up on reconnect
        without needing a ComfyUI restart.

        discover_and_register_tools() refetches /object_info but preserves the
        previous value if the ComfyUI fetch fails, and only clears+re-registers
        the tool registry once a build succeeds — so a transient hiccup rebuilds
        the same catalog rather than wiping it, and the next connection
        self-heals.
        """
        first = not self._initial_discovery_done
        try:
            await self.discover_and_register_tools()
        except Exception:
            logger.error(
                "\033[1m\033[35m[STP]\033[0m "
                "\033[31mDiscovery failed during tools.list — "
                "returning last known tools\033[0m",
                exc_info=True,
            )

        if first:
            self._initial_discovery_done = True
            n = len(self._tool_registry.list())
            logger.info(
                f"\033[1m\033[35m[STP]\033[0m Provider ready — "
                f"\033[1m{n}\033[0m tool(s) registered"
            )
        return await super()._handle_tools_list(request)

    async def on_refresh(self):
        """Handle tools.refresh — rescan workflows and update tools."""
        changed = await self.discover_and_register_tools()
        if changed:
            logger.info("Tool list changed during refresh, notifying Stimma")
            await self.notify_tools_changed()

    def get_workflow(self, slug: str):
        """Get a discovered workflow by tool slug."""
        return self._discovered_workflows.get(slug)

    def _snapshot_workflow_files(self) -> Dict[str, float]:
        """Get mtime snapshot of all workflow files that should trigger refresh.

        Includes ComfyUI workflow directories (discovery targets) and the plugin's
        bundled workflows source directory so source edits can re-run sync.
        """
        from .discovery import _get_comfyui_workflow_dirs, _is_scannable_json

        snapshot = {}
        dirs = list(_get_comfyui_workflow_dirs())
        dirs.extend(self._plugin_config.discovery.extra_workflow_dirs)

        plugin_wf = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workflows")
        if os.path.isdir(plugin_wf):
            dirs.append(plugin_wf)

        for directory in dirs:
            if not os.path.isdir(directory):
                continue
            for root, _dirs, files in os.walk(directory):
                for f in files:
                    if not _is_scannable_json(f):
                        continue
                    path = os.path.join(root, f)
                    try:
                        snapshot[path] = os.path.getmtime(path)
                    except OSError:
                        pass
        return snapshot

    def _snapshot_comfyui_deps(self) -> str:
        """Lightweight fingerprint of models/ and custom_nodes/ directories.

        Returns a string hash that changes when models or custom nodes are
        added or removed.  Model subfolders are walked recursively (by name,
        not mtime) so nested additions like loras/flux/foo.safetensors are
        caught; custom_nodes uses a top-level listing.  os.walk reads dir
        entries via os.scandir without stat-ing files, so it stays cheap
        enough to call every poll cycle.
        """
        import hashlib
        parts = []

        try:
            import folder_paths
            base = folder_paths.base_path
        except (ImportError, AttributeError):
            return ""

        # custom_nodes: top-level directory listing
        cn_dir = os.path.join(base, "custom_nodes")
        if os.path.isdir(cn_dir):
            try:
                entries = sorted(os.listdir(cn_dir))
                parts.append(f"cn:{','.join(entries)}")
            except OSError:
                pass

        # models: check each known subfolder's file listing
        # folder_paths tracks all model directories — iterate the ones we care about
        model_subdirs = [
            "checkpoints", "unet", "diffusion_models", "loras",
            "vae", "clip", "controlnet", "embeddings",
        ]
        for subdir in model_subdirs:
            try:
                dirs = folder_paths.get_folder_paths(subdir)
            except Exception:
                continue
            for d in dirs:
                if not os.path.isdir(d):
                    continue
                try:
                    # Recursive listing so models added in nested folders
                    # (e.g. loras/flux/foo.safetensors) are detected. A
                    # top-level os.listdir() misses them because the parent
                    # subfolder already exists — and LoRAs are almost always
                    # organized in subdirs (matching path_filter like "flux/**").
                    # os.walk uses os.scandir and we never read/stat files, so
                    # this stays cheap even on large model libraries.
                    names = []
                    for root, _subdirs, files in os.walk(d):
                        rel_root = os.path.relpath(root, d)
                        for f in files:
                            names.append(
                                f if rel_root == "." else os.path.join(rel_root, f)
                            )
                    parts.append(f"{subdir}:{','.join(sorted(names))}")
                except OSError:
                    pass

        h = hashlib.md5("|".join(parts).encode()).hexdigest()
        return h

    async def _watch_workflows(self):
        """Poll workflow directories and ComfyUI deps for changes, re-discover when needed."""
        self._workflow_snapshot = self._snapshot_workflow_files()
        self._deps_snapshot = self._snapshot_comfyui_deps()
        interval = self._plugin_config.discovery.watch_interval
        if interval <= 0:
            logger.info("Workflow file watching disabled (watch_interval <= 0)")
            return

        logger.info(
            f"\033[1m\033[35m[STP]\033[0m File watcher active (every {interval}s)"
        )

        while True:
            await asyncio.sleep(interval)
            try:
                new_snapshot = self._snapshot_workflow_files()
                new_deps = self._snapshot_comfyui_deps()
                workflows_changed = new_snapshot != self._workflow_snapshot
                deps_changed = new_deps != self._deps_snapshot

                if workflows_changed or deps_changed:
                    if workflows_changed:
                        plugin_wf = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workflows")
                        changed_paths = set(new_snapshot.keys()) ^ set(self._workflow_snapshot.keys())
                        for path, mtime in new_snapshot.items():
                            if self._workflow_snapshot.get(path) != mtime:
                                changed_paths.add(path)
                        self._workflow_snapshot = new_snapshot
                        logger.info(
                            "\033[1m\033[35m[STP]\033[0m "
                            "Workflow files changed, re-scanning..."
                        )
                        if any(path.startswith(plugin_wf + os.sep) for path in changed_paths):
                            from .workflow_install import sync_bundled_workflows
                            sync_bundled_workflows()
                    if deps_changed:
                        self._deps_snapshot = new_deps
                        logger.info(
                            "\033[1m\033[35m[STP]\033[0m "
                            "Models or custom nodes changed, re-scanning..."
                        )
                    await self.discover_and_register_tools()
                    await self.notify_tools_changed()
                    logger.info(
                        f"\033[1m\033[35m[STP]\033[0m "
                        f"Notified Stimma — "
                        f"\033[1m{len(self._tool_registry.list())}\033[0m tool(s)"
                    )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(
                    f"\033[1m\033[35m[STP]\033[0m \033[31mWatcher error: {e}\033[0m",
                    exc_info=True,
                )

    async def _handle_request(self, request):
        """Override to intercept upload methods before base class routing."""
        if request.method == "tools.upload":
            return await self._handle_tools_upload(request)
        elif request.method == "tools.upload_complete":
            return await self._handle_tools_upload_complete(request)
        return await super()._handle_request(request)

    async def _handle_tools_upload(self, request):
        """Handle tools.upload — accept an incoming LoRA file upload."""
        params = request.params or {}
        upload_id = params.get("upload_id", "")
        tool_id = params.get("tool_id", "")
        filename = params.get("filename", "")

        # Look up the workflow to find path_filter
        workflow = self._discovered_workflows.get(tool_id)
        if not workflow:
            return JsonRpcResponse.success(request.id, {
                "accepted": False,
                "error": f"Unknown tool: {tool_id}",
            })

        # Find the lora node's path_filter
        path_filter = ""
        for node in workflow.lora_nodes:
            path_filter = node.get("inputs", {}).get("path_filter", "")
            break

        subdir = self._get_upload_subdir(path_filter)

        # Preserve extension from the original filename
        _, ext = os.path.splitext(filename)
        asset_id = f"{upload_id}{ext}"

        self._pending_uploads[upload_id] = {
            "asset_id": asset_id,
            "filename": filename,
            "subdir": subdir,
        }

        logger.info(
            f"\033[1m\033[35m[STP]\033[0m Upload accepted: {filename} → {subdir}/"
        )

        return JsonRpcResponse.success(request.id, {
            "accepted": True,
            "asset_id": asset_id,
        })

    async def _handle_tools_upload_complete(self, request):
        """Handle tools.upload_complete — install the uploaded LoRA file."""
        params = request.params or {}
        upload_id = params.get("upload_id", "")

        upload = self._pending_uploads.pop(upload_id, None)
        if not upload:
            return JsonRpcResponse.success(request.id, {
                "success": False,
                "error": f"Unknown upload_id: {upload_id}",
            })

        asset_id = upload["asset_id"]
        filename = upload["filename"]
        subdir = upload["subdir"]

        try:
            # Read file bytes from asset storage
            data = await self.assets.download(asset_id)

            # Get LoRA base directory from ComfyUI
            import folder_paths
            lora_dirs = folder_paths.get_folder_paths("loras")
            base_dir = lora_dirs[0]

            # Write to target location
            if subdir:
                dest_dir = os.path.join(base_dir, subdir)
            else:
                dest_dir = base_dir
            os.makedirs(dest_dir, exist_ok=True)

            dest_path = os.path.join(dest_dir, filename)
            with open(dest_path, "wb") as f:
                f.write(data)

            # Clean up the asset copy
            try:
                await self.assets.delete(asset_id)
            except Exception:
                pass

            installed_path = f"{subdir}/{filename}" if subdir else filename

            logger.info(
                f"\033[1m\033[35m[STP]\033[0m LoRA installed: {installed_path}"
            )

            return JsonRpcResponse.success(request.id, {
                "success": True,
                "installed_path": installed_path,
            })

        except Exception as e:
            logger.error(
                f"\033[1m\033[35m[STP]\033[0m \033[31mUpload install failed: {e}\033[0m",
                exc_info=True,
            )
            return JsonRpcResponse.success(request.id, {
                "success": False,
                "error": str(e),
            })

    @staticmethod
    def _get_upload_subdir(path_filter: str) -> str:
        """Extract directory prefix from a path_filter pattern.

        Examples:
            "flux/**" → "flux"
            "wan/**;flux/**" → "wan"
            "**" or "" → ""
        """
        if not path_filter or not path_filter.strip():
            return ""
        first_pattern = path_filter.split(";")[0].strip()
        prefix = first_pattern.split("*")[0].rstrip("/")
        return prefix
