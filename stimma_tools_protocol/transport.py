"""
Transport layer for STP communication.

Supports:
- Stdio: Newline-delimited JSON on stdin/stdout
- WebSocket: WebSocket server for remote providers
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Callable, Awaitable
import asyncio
import sys
import json
import logging

from stimma_tools_protocol.protocol import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcNotification,
    JsonRpcError,
    JsonRpcErrorCode,
    parse_message,
)

logger = logging.getLogger(__name__)


class Transport(ABC):
    """Abstract base class for STP transports."""

    @abstractmethod
    async def start(self) -> None:
        """Start the transport."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the transport."""
        pass

    @abstractmethod
    async def send(self, message: str) -> None:
        """Send a message."""
        pass

    @abstractmethod
    async def receive(self) -> AsyncIterator[str]:
        """Receive messages as an async iterator."""
        pass

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the transport is running."""
        pass


class StdioTransport(Transport):
    """
    Stdio transport for subprocess mode.

    - Reads newline-delimited JSON from stdin
    - Writes newline-delimited JSON to stdout
    - All logging goes to stderr
    """

    def __init__(self):
        self._running = False
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the stdio transport."""
        self._running = True
        logger.debug("Stdio transport started")

    async def stop(self) -> None:
        """Stop the stdio transport."""
        self._running = False
        # Close stdin to unblock any readline() in the thread pool
        try:
            sys.stdin.close()
        except Exception:
            pass
        logger.debug("Stdio transport stopped")

    async def send(self, message: str) -> None:
        """Send a message to stdout."""
        async with self._write_lock:
            line = message.rstrip("\n") + "\n"
            sys.stdout.write(line)
            sys.stdout.flush()

    async def receive(self) -> AsyncIterator[str]:
        """Receive messages from stdin."""
        while self._running:
            try:
                # Read in thread so it doesn't block the event loop
                # and can be interrupted by signals
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    # EOF
                    logger.debug("Stdin EOF received")
                    break

                message = line.strip()
                if message:
                    yield message
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading from stdin: {e}")
                break

    @property
    def is_running(self) -> bool:
        return self._running


class WebSocketTransport(Transport):
    """
    WebSocket transport for remote providers using aiohttp.

    Runs an HTTP server that:
    - Handles WebSocket connections at /tools-rpc (or configured path)
    - Can serve additional HTTP routes (e.g., asset endpoints)

    Supports multiple clients connecting sequentially or concurrently.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        auth_token: Optional[str] = None,
        ws_path: str = "/stp-v1",
    ):
        self._host = host
        self._port = port
        self._auth_token = auth_token
        self._ws_path = ws_path
        self._running = False
        self._app = None
        self._runner = None
        self._site = None
        self._clients: set = set()
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._first_client_connected: asyncio.Event = asyncio.Event()
        # Callback for new client connections (for re-registration)
        self._on_client_connected: Optional[Callable[[], Awaitable[None]]] = None
        # Additional routes to add (e.g., asset routes)
        self._additional_routes: list = []

    def add_routes(self, routes: list) -> None:
        """
        Add additional HTTP routes to the server.

        Must be called before start().

        Args:
            routes: List of aiohttp route definitions
        """
        self._additional_routes.extend(routes)

    async def start(self) -> None:
        """Start the aiohttp server with WebSocket and HTTP support."""
        try:
            from aiohttp import web
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp package required for WebSocket transport")

        async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
            """Handle WebSocket connections."""
            # Validate auth if configured
            if self._auth_token:
                auth_header = request.headers.get("Authorization", "")
                if not auth_header.startswith("Bearer "):
                    return web.Response(status=401, text="Missing authorization")
                token = auth_header[7:]  # Strip "Bearer "
                if token != self._auth_token:
                    return web.Response(status=401, text="Invalid authorization")

            ws = web.WebSocketResponse()
            await ws.prepare(request)

            self._clients.add(ws)
            client_addr = request.remote
            logger.info(f"WebSocket client connected: {client_addr}")

            # Signal that first client has connected
            is_first = not self._first_client_connected.is_set()
            if is_first:
                self._first_client_connected.set()
            else:
                # New client after first - notify for re-registration
                if self._on_client_connected:
                    logger.info("New client connected, triggering re-registration")
                    asyncio.create_task(self._on_client_connected())

            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await self._message_queue.put(msg.data)
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        await self._message_queue.put(msg.data.decode("utf-8"))
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {ws.exception()}")
                        break
            except Exception as e:
                logger.debug(f"WebSocket connection closed: {e}")
            finally:
                self._clients.discard(ws)
                logger.info(f"WebSocket client disconnected: {client_addr}")

            return ws

        # Create aiohttp app with large body size limit for asset uploads
        # Default is 1MB which is too small for images/videos
        self._app = web.Application(client_max_size=1024 * 1024 * 1024)  # 1GB

        # Add WebSocket route
        self._app.router.add_get(self._ws_path, websocket_handler)

        # Add any additional routes (e.g., asset routes)
        for route in self._additional_routes:
            self._app.router.add_route(route.method, route.path, route.handler)

        # Start the server
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()

        self._running = True
        # Get actual port if 0 was specified
        actual_port = self._site._server.sockets[0].getsockname()[1] if self._site._server else self._port
        self._port = actual_port

        logger.info(f"Server listening on http://{self._host}:{actual_port}")
        logger.info(f"WebSocket endpoint: ws://{self._host}:{actual_port}{self._ws_path}")
        logger.info("Waiting for first client to connect...")

        # Wait for first client to connect before returning
        await self._first_client_connected.wait()
        logger.info("First client connected, transport ready")

    async def stop(self) -> None:
        """Stop the aiohttp server."""
        self._running = False

        # Close all WebSocket connections
        for client in list(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()

        # Stop the server
        if self._runner:
            await self._runner.cleanup()

        logger.debug("WebSocket transport stopped")

    async def send(self, message: str) -> None:
        """Send a message to all connected WebSocket clients."""
        async with self._write_lock:
            if not self._clients:
                logger.warning("No WebSocket clients connected")
                return

            # Log outgoing message
            try:
                parsed = json.loads(message)
                method = parsed.get("method", parsed.get("result", {}).get("method") if isinstance(parsed.get("result"), dict) else None)
                msg_id = parsed.get("id")
                logger.debug(f"WS SEND: method={method}, id={msg_id}")
            except Exception:
                pass

            # Send to all clients
            for client in list(self._clients):
                try:
                    await client.send_str(message)
                except Exception as e:
                    logger.error(f"Error sending to client: {e}")
                    self._clients.discard(client)

    async def receive(self) -> AsyncIterator[str]:
        """Receive messages from connected WebSocket clients."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                # Log incoming message
                try:
                    parsed = json.loads(message)
                    method = parsed.get("method")
                    msg_id = parsed.get("id")
                    logger.debug(f"WS RECV: method={method}, id={msg_id}")
                except Exception:
                    pass

                yield message
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def port(self) -> int:
        """Get the actual port."""
        return self._port

    @property
    def app(self):
        """Get the aiohttp app (for adding custom routes after start if needed)."""
        return self._app


class MessageHandler:
    """
    Handles JSON-RPC message routing and response generation.
    """

    def __init__(
        self,
        transport: Transport,
        on_request: Optional[Callable[[JsonRpcRequest], Awaitable[JsonRpcResponse]]] = None,
        on_notification: Optional[Callable[[JsonRpcNotification], Awaitable[None]]] = None,
    ):
        self._transport = transport
        self._on_request = on_request
        self._on_notification = on_notification
        self._pending_responses: dict[str | int, asyncio.Future] = {}

    async def send_request(self, method: str, params: Optional[dict] = None) -> JsonRpcResponse:
        """Send a request and wait for response."""
        import uuid
        request_id = str(uuid.uuid4())
        request = JsonRpcRequest(method=method, id=request_id, params=params)

        future: asyncio.Future = asyncio.Future()
        self._pending_responses[request_id] = future

        await self._transport.send(request.to_json())

        try:
            return await future
        finally:
            self._pending_responses.pop(request_id, None)

    async def send_response(self, response: JsonRpcResponse) -> None:
        """Send a response."""
        await self._transport.send(response.to_json())

    async def send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """Send a notification (no response expected)."""
        notification = JsonRpcNotification(method=method, params=params)
        await self._transport.send(notification.to_json())

    async def handle_message(self, message: str) -> None:
        """Handle an incoming message."""
        try:
            parsed = parse_message(message)
        except ValueError as e:
            # Send parse error response
            error = JsonRpcError.from_code(JsonRpcErrorCode.PARSE_ERROR, str(e))
            response = JsonRpcResponse.failure(None, error)
            await self._transport.send(response.to_json())
            return

        if isinstance(parsed, JsonRpcResponse):
            # Handle response to a pending request
            if parsed.id in self._pending_responses:
                future = self._pending_responses.pop(parsed.id)
                future.set_result(parsed)
        elif isinstance(parsed, JsonRpcRequest):
            # Handle incoming request
            if self._on_request:
                try:
                    response = await self._on_request(parsed)
                    await self._transport.send(response.to_json())
                except Exception as e:
                    logger.exception(f"Error handling request: {e}")
                    error = JsonRpcError.from_code(JsonRpcErrorCode.INTERNAL_ERROR, str(e))
                    response = JsonRpcResponse.failure(parsed.id, error)
                    await self._transport.send(response.to_json())
            else:
                error = JsonRpcError.from_code(JsonRpcErrorCode.METHOD_NOT_FOUND)
                response = JsonRpcResponse.failure(parsed.id, error)
                await self._transport.send(response.to_json())
        elif isinstance(parsed, JsonRpcNotification):
            # Handle notification
            if self._on_notification:
                try:
                    await self._on_notification(parsed)
                except Exception as e:
                    logger.exception(f"Error handling notification: {e}")

    def close(self) -> None:
        """Cancel all pending request futures."""
        for future in self._pending_responses.values():
            if not future.done():
                future.cancel()
        self._pending_responses.clear()

    async def run(self) -> None:
        """Run the message handling loop."""
        try:
            async for message in self._transport.receive():
                await self.handle_message(message)
        finally:
            # Cancel any pending requests when transport closes
            self.close()
