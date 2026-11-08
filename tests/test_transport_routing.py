"""Two Stimma hosts on one ComfyUI each get only their own replies and results.

Every host numbers its JSON-RPC requests from 1, so a reply broadcast to all
clients can resolve another host's request with the same id.

Run with: python tests/test_transport_routing.py (needs aiohttp)
"""

import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aiohttp  # noqa: E402
from aiohttp import web  # noqa: E402

from stimma_tools_protocol.provider import Provider, ProviderConfig  # noqa: E402
from stimma_tools_protocol.tool import Tool, ToolRegistry  # noqa: E402
from stp_server.transport import ComfyUITransport  # noqa: E402


class Host:
    """A fake Stimma host: answers provider.register and records frames."""

    def __init__(self, ws):
        self.ws = ws
        self.frames = []
        self.registered = asyncio.Event()
        self.task = asyncio.create_task(self._read())

    async def _read(self):
        async for msg in self.ws:
            frame = json.loads(msg.data)
            self.frames.append(frame)
            if frame.get("method") == "provider.register":
                await self.ws.send_str(json.dumps({
                    "jsonrpc": "2.0",
                    "id": frame["id"],
                    "result": {"session_id": "s", "stimma_version": "test"},
                }))
                self.registered.set()

    async def request(self, rpc_id, method, params):
        await self.ws.send_str(json.dumps(
            {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}
        ))

    def replies(self):
        return [f for f in self.frames if "method" not in f]

    def notifications(self, method):
        return [f["params"] for f in self.frames if f.get("method") == method]


class TransportRoutingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def echo(context, parameters):
            await context.report_progress(0.5)
            return {"label": parameters["label"]}

        registry = ToolRegistry()
        registry.register(Tool(
            slug="echo", display_name="Echo", task_type="text-to-image", function=echo,
        ))

        app = web.Application()
        self.transport = ComfyUITransport(app)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/stp-v1"

        self.provider = Provider(
            ProviderConfig(provider_id="p", provider_name="P"), self.transport, registry,
        )
        self.session = aiohttp.ClientSession()
        start = asyncio.create_task(self.provider.start(asset_manager=MagicMock()))
        self.a = Host(await self.session.ws_connect(url))
        await start
        self.b = Host(await self.session.ws_connect(url))
        await asyncio.wait_for(self.b.registered.wait(), 5)

    async def asyncTearDown(self):
        await self.provider.stop()
        await self.session.close()
        await self.runner.cleanup()

    async def test_replies_and_job_notifications_reach_only_the_sender(self):
        # Both hosts use id 1, as two real backends would.
        await self.a.request(1, "tools.execute", {
            "request_id": "exec-a", "tool_id": "echo", "parameters": {"label": "a"},
        })
        await self.b.request(1, "tools.execute", {
            "request_id": "exec-b", "tool_id": "echo", "parameters": {"label": "b"},
        })

        async def both_done():
            while not (self.a.notifications("tools.result") and self.b.notifications("tools.result")):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(both_done(), 5)
        await asyncio.sleep(0.1)  # let any stray frame arrive

        for host, mine in ((self.a, "exec-a"), (self.b, "exec-b")):
            self.assertEqual(
                [r for r in host.replies() if r["id"] == 1],
                [{"jsonrpc": "2.0", "id": 1, "result": {"accepted": True}}],
            )
            self.assertEqual([r["request_id"] for r in host.notifications("tools.result")], [mine])
            self.assertEqual([p["request_id"] for p in host.notifications("tools.progress")], [mine])
            # Queue status still goes to everyone.
            self.assertTrue(host.notifications("queue.status"))

        self.assertEqual(self.a.notifications("tools.result")[0]["output"]["label"], "a")
        self.assertEqual(self.b.notifications("tools.result")[0]["output"]["label"], "b")


if __name__ == "__main__":
    unittest.main()
