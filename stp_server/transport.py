"""ComfyUI transport — hooks STP WebSocket endpoint into ComfyUI's PromptServer."""

import asyncio
import json
import logging
from typing import AsyncIterator, Optional, Callable, Awaitable

from stimma_tools_protocol.transport import Transport

logger = logging.getLogger(__name__)


class ComfyUITransport(Transport):
    """
    Transport that adds an STP WebSocket endpoint to ComfyUI's existing aiohttp app.

    Instead of running its own HTTP server, this hooks into PromptServer.instance.app
    so the STP endpoint lives on the same port as ComfyUI (e.g., :8188/stp-v1).

    Routes must be added during the plugin import phase (before aiohttp freezes the router).
    """

    def __init__(self, app, ws_path: str = "/stp-v1"):
        self._app = app
        self._ws_path = ws_path
        self._running = False
        self._clients: set = set()
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._first_client_connected: asyncio.Event = asyncio.Event()
        self._on_client_connected: Optional[Callable[[], Awaitable[None]]] = None

        # Add WebSocket route immediately (before app starts / freezes router)
        app.router.add_get(ws_path, self._websocket_handler)
        logger.info(f"Added STP WebSocket route: {ws_path}")

    def add_routes(self, routes: list) -> None:
        """Add additional HTTP routes to ComfyUI's app.

        Must be called before the aiohttp app starts (during plugin import).
        """
        for route in routes:
            self._app.router.add_route(route.method, route.path, route.handler)

    async def start(self) -> None:
        """Start the transport. Waits for the first STP client to connect."""
        self._running = True
        logger.info(f"STP transport ready, waiting for client on {self._ws_path}...")
        await self._first_client_connected.wait()
        logger.info("First STP client connected")

    async def stop(self) -> None:
        """Stop the transport (close WS connections, don't stop ComfyUI's server)."""
        self._running = False
        for client in list(self._clients):
            try:
                await client.close()
            except Exception:
                pass
        self._clients.clear()
        logger.debug("ComfyUI transport stopped")

    async def send(self, message: str) -> None:
        """Send a message to all connected STP clients."""
        async with self._write_lock:
            # Parse message for logging
            method = None
            try:
                parsed = json.loads(message)
                method = parsed.get("method")
                msg_id = parsed.get("id")
                logger.debug(f"STP SEND: method={method}, id={msg_id}")
            except Exception:
                pass

            if not self._clients:
                if method == "tools.result":
                    logger.error(
                        f"No STP clients connected — tools.result DROPPED. "
                        f"This will cause a stuck slot in Stimma."
                    )
                else:
                    logger.warning(f"No STP clients connected (dropping {method})")
                return

            for client in list(self._clients):
                try:
                    await client.send_str(message)
                except Exception as e:
                    logger.error(f"Error sending to STP client: {e}")
                    self._clients.discard(client)

    async def receive(self) -> AsyncIterator[str]:
        """Receive messages from connected STP clients."""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                try:
                    parsed = json.loads(message)
                    method = parsed.get("method")
                    msg_id = parsed.get("id")
                    logger.debug(f"STP RECV: method={method}, id={msg_id}")
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

    async def _websocket_handler(self, request):
        """Handle WebSocket connections on ComfyUI's server."""
        from aiohttp import web
        import aiohttp

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._clients.add(ws)
        client_addr = request.remote
        logger.info(f"STP client connected: {client_addr}")

        is_first = not self._first_client_connected.is_set()
        if is_first:
            self._first_client_connected.set()
        elif self._on_client_connected:
            asyncio.create_task(self._on_client_connected())

        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._message_queue.put(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await self._message_queue.put(msg.data.decode("utf-8"))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"STP WebSocket error: {ws.exception()}")
                    break
        except Exception as e:
            logger.debug(f"STP WebSocket connection closed: {e}")
        finally:
            self._clients.discard(ws)
            logger.info(f"STP client disconnected: {client_addr}")

        return ws
