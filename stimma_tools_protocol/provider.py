"""
Provider base class for STP providers.

Handles registration, tool listing, heartbeat handling, and job execution.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Awaitable
import asyncio
import logging
from contextlib import asynccontextmanager

from stimma_tools_protocol.protocol import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcNotification,
    JsonRpcError,
    JsonRpcErrorCode,
    ProviderRegistration,
    RegistrationResponse,
    ExecuteRequest,
    QueueStatus,
    ProgressNotification,
    ExecutionResult,
    STPError,
    STPErrorCode,
)
from stimma_tools_protocol.transport import Transport, MessageHandler
from stimma_tools_protocol.tool import Tool, ToolRegistry, get_registry
from stimma_tools_protocol.assets import AssetManager, LocalAssetServer

logger = logging.getLogger(__name__)


_MEDIA_TYPE_BY_EXT = {
    "png": "image", "jpg": "image", "jpeg": "image", "gif": "image", "webp": "image",
    "mp4": "video", "webm": "video", "mov": "video",
    "mp3": "audio", "wav": "audio", "ogg": "audio",
}


def _infer_media_type(asset_id: str) -> str:
    ext = asset_id.rsplit(".", 1)[-1].lower() if "." in asset_id else ""
    return _MEDIA_TYPE_BY_EXT.get(ext, "document")


def normalize_output(result: Any) -> dict:
    """Normalize a tool's return value to the STP output envelope: {"assets": [...]}.

    Accepts either the envelope form ({"assets": [...]}) or the ergonomic single-asset form
    ({"asset_id": "..."}); other top-level keys are preserved.
    """
    if not isinstance(result, dict):
        return {"assets": []}
    if "assets" in result:
        return result
    out = dict(result)
    asset_id = out.pop("asset_id", None)
    out["assets"] = (
        [{"asset_id": asset_id, "type": _infer_media_type(asset_id), "role": "primary"}]
        if asset_id else []
    )
    return out


@dataclass
class ProviderConfig:
    """Configuration for an STP provider."""
    provider_id: str
    provider_name: str
    server: Optional[str] = None  # Software identifier, e.g. "ComfyUI-Stimma/1.2.3"
    max_concurrent: int = 1
    supports_cancel: bool = True  # exposed as the `cancel` capability
    asset_endpoint: str = "/stp-v1/assets"  # Provider's asset endpoint path


@dataclass
class ExecutionContext:
    """Context passed to tool execution."""
    request_id: str
    tool: Tool
    parameters: dict
    assets: AssetManager
    _provider: "Provider"

    async def report_progress(self, progress: float) -> None:
        """Report progress (0.0 to 1.0) for this execution."""
        await self._provider._send_progress(self.request_id, progress)


@dataclass
class Job:
    """A queued or running job."""
    request_id: str
    tool: Tool
    parameters: dict
    task: Optional[asyncio.Task] = None
    cancelled: bool = False


class Provider:
    """
    Base class for STP providers.

    Handles the STP protocol lifecycle:
    - Registration with Stimma
    - Tool listing and refresh
    - Job execution with progress reporting
    - Queue management
    - Heartbeat handling
    """

    def __init__(
        self,
        config: ProviderConfig,
        transport: Transport,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self._config = config
        self._transport = transport
        self._registry = tool_registry or get_registry()
        self._handler: Optional[MessageHandler] = None
        self._assets: Optional[AssetManager] = None
        self._session_id: Optional[str] = None

        # Job management
        self._queued_jobs: List[Job] = []
        self._running_jobs: Dict[str, Job] = {}
        self._job_semaphore = asyncio.Semaphore(config.max_concurrent)
        # Signalled whenever a job is enqueued so the processor wakes
        # immediately instead of polling on a fixed interval.
        self._job_available = asyncio.Event()
        self._job_processor_task: Optional[asyncio.Task] = None
        self._message_loop_task: Optional[asyncio.Task] = None

        # Shutdown
        self._shutdown_event = asyncio.Event()

    @property
    def config(self) -> ProviderConfig:
        """Get the provider configuration."""
        return self._config

    @property
    def assets(self) -> AssetManager:
        """Get the asset manager."""
        if not self._assets:
            raise RuntimeError("Provider not started - no asset manager available")
        return self._assets

    async def start(self, asset_manager: Optional[AssetManager] = None) -> None:
        """
        Start the provider.

        Args:
            asset_manager: Optional pre-configured asset manager
        """
        logger.info(f"Starting provider: {self._config.provider_name}")

        # For transports that serve HTTP (WebSocket, ComfyUI hook), set up asset serving
        if hasattr(self._transport, 'add_routes'):
            if asset_manager is None:
                # Create local asset server and add its routes
                self._assets = LocalAssetServer()
                asset_routes = self._assets.get_aiohttp_routes(self._config.asset_endpoint)
                self._transport.add_routes(asset_routes)
                logger.info(f"Asset endpoint: {self._config.asset_endpoint}")
            else:
                # Asset manager (and routes) already set up externally
                self._assets = asset_manager

            # Set up callback for new client connections (WebSocket reconnect support)
            if hasattr(self._transport, '_on_client_connected'):
                self._transport._on_client_connected = self._on_new_client
        else:
            # Stdio transport - use provided asset manager
            self._assets = asset_manager

        # Start transport
        await self._transport.start()

        # Set up message handler
        self._handler = MessageHandler(
            transport=self._transport,
            on_request=self._handle_request,
            on_notification=self._handle_notification,
        )

        # Start message receiving loop in background BEFORE registration
        self._message_loop_task = asyncio.create_task(self._run_message_loop())

        # Send registration
        await self._register()

        # Start job processor
        self._job_processor_task = asyncio.create_task(self._process_jobs())

        logger.info(f"Provider started: {self._config.provider_name}")

    async def _on_new_client(self) -> None:
        """Handle a new WebSocket client connection - re-register."""
        logger.info("New client connected, re-registering...")
        try:
            await self._register()
            logger.info("Re-registration complete")
        except Exception as e:
            logger.error(f"Re-registration failed: {e}")

    async def stop(self) -> None:
        """Stop the provider gracefully."""
        logger.info(f"Stopping provider: {self._config.provider_name}")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel all running jobs
        for job in list(self._running_jobs.values()):
            if job.task and not job.task.done():
                job.task.cancel()

        # Wait for job processor to finish
        if self._job_processor_task:
            self._job_processor_task.cancel()
            try:
                await self._job_processor_task
            except asyncio.CancelledError:
                pass

        # Send disconnect notification before stopping transport
        if self._handler:
            try:
                await self._handler.send_notification(
                    "provider.disconnect",
                    {"reason": "shutdown"},
                )
            except Exception:
                pass

        # Stop transport BEFORE awaiting message loop - this unblocks readline()
        await self._transport.stop()

        # Now cancel and await message loop (should exit quickly since transport stopped)
        if self._message_loop_task:
            self._message_loop_task.cancel()
            try:
                await self._message_loop_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Provider stopped: {self._config.provider_name}")

    async def run(self) -> None:
        """Run the provider until shutdown."""
        if not self._handler:
            raise RuntimeError("Provider not started")

        try:
            # Wait for message loop to complete (happens on transport close or shutdown)
            if self._message_loop_task:
                await self._message_loop_task
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _run_message_loop(self) -> None:
        """Background task to receive and handle messages."""
        if not self._handler:
            return

        try:
            await self._handler.run()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Message loop error: {e}")

    async def _register(self) -> None:
        """Register with Stimma."""
        if not self._handler:
            raise RuntimeError("Handler not initialized")

        # Include asset_endpoint for HTTP-serving transports
        asset_endpoint = None
        if hasattr(self._transport, 'add_routes'):
            asset_endpoint = self._config.asset_endpoint

        registration = ProviderRegistration(
            provider_id=self._config.provider_id,
            provider_name=self._config.provider_name,
            server=self._config.server,
            max_concurrent=self._config.max_concurrent,
            capabilities={"cancel": self._config.supports_cancel},
            asset_endpoint=asset_endpoint,
        )

        response = await self._handler.send_request(
            "provider.register",
            registration.to_dict(),
        )

        if response.error:
            raise RuntimeError(f"Registration failed: {response.error.message}")

        result = RegistrationResponse.from_dict(response.result)
        self._session_id = result.session_id

        logger.info(f"Registered with session ID: {self._session_id}")

    async def _handle_request(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Handle an incoming JSON-RPC request."""
        method = request.method
        params = request.params or {}

        try:
            if method == "tools.list":
                return await self._handle_tools_list(request)
            elif method == "tools.refresh":
                return await self._handle_tools_refresh(request)
            elif method == "tools.execute":
                return await self._handle_tools_execute(request, params)
            elif method == "tools.cancel":
                return await self._handle_tools_cancel(request, params)
            else:
                error = JsonRpcError.from_code(JsonRpcErrorCode.METHOD_NOT_FOUND)
                return JsonRpcResponse.failure(request.id, error)
        except Exception as e:
            logger.exception(f"Error handling request {method}: {e}")
            error = JsonRpcError.from_code(JsonRpcErrorCode.INTERNAL_ERROR, str(e))
            return JsonRpcResponse.failure(request.id, error)

    async def _handle_notification(self, notification: JsonRpcNotification) -> None:
        """Handle an incoming JSON-RPC notification."""
        # Currently no notifications to handle from Stimma
        logger.debug(f"Received notification: {notification.method}")

    async def _handle_tools_list(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Handle tools.list request."""
        tools = self._registry.list_descriptors()
        return JsonRpcResponse.success(request.id, {
            "tools": [t.to_dict() for t in tools]
        })

    async def _handle_tools_refresh(self, request: JsonRpcRequest) -> JsonRpcResponse:
        """Handle tools.refresh request."""
        # Subclasses can override on_refresh to update tool list
        await self.on_refresh()
        tools = self._registry.list_descriptors()
        return JsonRpcResponse.success(request.id, {
            "tools": [t.to_dict() for t in tools]
        })

    async def _handle_tools_execute(
        self,
        request: JsonRpcRequest,
        params: dict,
    ) -> JsonRpcResponse:
        """Handle tools.execute request."""
        exec_request = ExecuteRequest.from_dict(params)

        # Find the tool
        tool = self._registry.get(exec_request.tool_id)
        if not tool:
            error = JsonRpcError(
                code=-32000,
                message=f"Tool not found: {exec_request.tool_id}",
            )
            return JsonRpcResponse.failure(request.id, error)

        # Queue the job
        job = Job(
            request_id=exec_request.request_id,
            tool=tool,
            parameters=exec_request.parameters,
        )
        self._queued_jobs.append(job)
        self._job_available.set()

        # Send queue status
        await self._send_queue_status()

        return JsonRpcResponse.success(request.id, {"accepted": True})

    async def _handle_tools_cancel(
        self,
        request: JsonRpcRequest,
        params: dict,
    ) -> JsonRpcResponse:
        """Handle tools.cancel request."""
        request_id = params.get("request_id")

        # Check queued jobs
        for i, job in enumerate(self._queued_jobs):
            if job.request_id == request_id:
                self._queued_jobs.pop(i)
                await self._send_queue_status()
                return JsonRpcResponse.success(request.id, {"cancelled": True})

        # Check running jobs
        if request_id in self._running_jobs:
            job = self._running_jobs[request_id]
            job.cancelled = True
            if job.task and not job.task.done():
                job.task.cancel()
            return JsonRpcResponse.success(request.id, {"cancelled": True})

        return JsonRpcResponse.success(request.id, {
            "cancelled": False,
            "reason": "not_found",
        })


    async def _process_jobs(self) -> None:
        """Background task to process queued jobs."""
        while not self._shutdown_event.is_set():
            if not self._queued_jobs:
                # Wake the instant a job is enqueued; the timeout just lets us
                # re-check the shutdown flag periodically.
                self._job_available.clear()
                try:
                    await asyncio.wait_for(self._job_available.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                continue

            # Get next job
            job = self._queued_jobs.pop(0)

            # Wait for capacity
            await self._job_semaphore.acquire()

            # Start execution
            self._running_jobs[job.request_id] = job
            job.task = asyncio.create_task(self._execute_job(job))
            await self._send_queue_status()

    def _apply_defaults(self, tool: Tool, params: dict, param_list: list) -> dict:
        """Apply schema defaults to parameters.

        This ensures tool functions receive fully-populated parameter dicts
        without needing to repeat defaults in the function body.

        Args:
            tool: The tool being executed
            params: The parameters dict from the request
            param_list: List of ToolParameter definitions to apply

        Returns:
            Dict with schema defaults merged in for missing keys
        """
        result = dict(params)  # Copy to avoid mutation
        for p in param_list:
            if p.name not in result and p.default is not None:
                result[p.name] = p.default
        return result

    async def _execute_job(self, job: Job) -> None:
        """Execute a single job."""
        try:
            if not self._assets:
                raise RuntimeError("Asset manager not configured")

            # Apply schema defaults to parameters
            parameters = self._apply_defaults(job.tool, job.parameters, job.tool.parameters)

            context = ExecutionContext(
                request_id=job.request_id,
                tool=job.tool,
                parameters=parameters,
                assets=self._assets,
                _provider=self,
            )

            # Execute the tool
            result = await job.tool.function(context, parameters)

            # Send result
            if not job.cancelled:
                await self._send_result(
                    request_id=job.request_id,
                    success=True,
                    output=normalize_output(result),
                )

        except asyncio.CancelledError:
            if job.cancelled:
                await self._send_result(
                    request_id=job.request_id,
                    success=False,
                    error={"code": "CANCELLED", "message": "Job was cancelled"},
                )
            raise

        except STPError as e:
            await self._send_result(
                request_id=job.request_id,
                success=False,
                error=e.to_dict(),
            )

        except Exception as e:
            logger.exception(f"Job {job.request_id} failed: {e}")
            await self._send_result(
                request_id=job.request_id,
                success=False,
                error={"code": "EXECUTION_FAILED", "message": str(e)},
            )

        finally:
            # Cleanup
            self._running_jobs.pop(job.request_id, None)
            self._job_semaphore.release()
            await self._send_queue_status()

    async def _send_queue_status(self) -> None:
        """Send queue status notification."""
        if not self._handler:
            return

        status = QueueStatus(
            queued=len(self._queued_jobs),
            running=len(self._running_jobs),
            capacity=self._config.max_concurrent,
        )
        await self._handler.send_notification("queue.status", status.to_dict())

    async def _send_progress(self, request_id: str, progress: float) -> None:
        """Send progress notification for a job."""
        if not self._handler:
            return

        notification = ProgressNotification(
            request_id=request_id,
            progress=progress,
        )
        await self._handler.send_notification("tools.progress", notification.to_dict())

    async def _send_result(
        self,
        request_id: str,
        success: bool,
        output: Optional[dict] = None,
        error: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Send result notification for a completed job."""
        if not self._handler:
            logger.error(
                f"Cannot send result for {request_id}: no handler connected "
                f"(success={success}, error={error})"
            )
            return

        result = ExecutionResult(
            request_id=request_id,
            success=success,
            output=output,
            error=error,
            metadata=metadata or {},
        )
        try:
            await self._handler.send_notification("tools.result", result.to_dict())
        except Exception as e:
            logger.error(
                f"Failed to send result for {request_id}: {e} "
                f"(success={success}, error={error})"
            )

    async def on_refresh(self) -> None:
        """
        Called when tools.refresh is requested.

        Override this to dynamically update the tool registry,
        e.g., re-query available models.
        """
        pass

    async def notify_tools_changed(self) -> None:
        """
        Notify Stimma that the tool list has changed.

        Stimma will re-call tools.list to get the updated tool set.
        Use this when tools are added, removed, or modified dynamically
        (e.g., user saves a new workflow).
        """
        if self._handler:
            await self._handler.send_notification("tools.changed")
