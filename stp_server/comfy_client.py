"""Multi-instance ComfyUI client for load balancing across GPUs."""

import uuid
import json
import logging
import os
import asyncio
import tempfile
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, AsyncIterator

import aiohttp

logger = logging.getLogger(__name__)


def parse_addresses(addresses) -> List[str]:
    """Parse various address input formats into a list of individual addresses.

    Handles: list, single string, comma-separated, port ranges (host:8188-8191).
    """
    if isinstance(addresses, str):
        addresses = [addr.strip() for addr in addresses.split(",")]
    elif not isinstance(addresses, list):
        addresses = [str(addresses)]

    expanded = []
    for addr in addresses:
        if ":" not in addr:
            expanded.append(addr)
            continue
        host, port_spec = addr.rsplit(":", 1)
        if "-" in port_spec:
            start, end = map(int, port_spec.split("-"))
            expanded.extend(f"{host}:{port}" for port in range(start, end + 1))
        else:
            expanded.append(addr)

    return expanded


class SingleComfy:
    """Client for a single ComfyUI instance."""

    def __init__(self, addr: str):
        self.addr = addr
        self.client_id = str(uuid.uuid4())
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get (or lazily create) a keep-alive HTTP session for this instance.

        Reusing one session across requests keeps the TCP connection to ComfyUI
        warm instead of doing a fresh connect/teardown on every /prompt,
        /history, /object_info and /upload call. Created lazily so it binds to
        the running STP event loop.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(auto_decompress=False)
        return self._session

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        """Make an HTTP request to the ComfyUI server."""
        session = await self._get_session()
        url = f"http://{self.addr}{path}"
        async with session.request(method, url, **kwargs) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(
                    f"ComfyUI {method} {path} failed ({resp.status}): {error_text}"
                )
            raw = await resp.read()
            try:
                return json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return raw

    async def queue_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a workflow prompt for execution."""
        data = json.dumps({"prompt": prompt, "client_id": self.client_id})
        result = await self._request(
            "POST", "/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        # Check for validation/prompt errors in response body
        if "error" in result:
            error_info = result["error"]
            if isinstance(error_info, dict):
                error_msg = error_info.get("message", str(error_info))
            else:
                error_msg = str(error_info)
            node_errors = result.get("node_errors", {})
            if node_errors:
                # Summarize first few node errors
                details = []
                for nid, nerr in list(node_errors.items())[:3]:
                    errs = nerr.get("errors", []) if isinstance(nerr, dict) else []
                    for e in errs[:1]:
                        details.append(
                            f"node #{nid}: {e.get('message', str(e))}"
                        )
                if details:
                    error_msg += " — " + "; ".join(details)
            raise RuntimeError(f"ComfyUI prompt validation error: {error_msg}")
        if not result.get("prompt_id"):
            raise RuntimeError(f"ComfyUI response missing prompt_id: {result}")
        logger.debug(f"Queued prompt {result['prompt_id']} on {self.addr}")
        return result

    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """Get execution history for a prompt."""
        return await self._request("GET", f"/history/{prompt_id}")

    async def delete_history(self, prompt_id: str) -> None:
        """Delete a prompt's history entry (holds prompt text + output refs)."""
        await self._request("POST", "/history", json={"delete": [prompt_id]})

    async def get_object_info(self) -> Dict[str, Any]:
        """Get available nodes, models, samplers, etc."""
        return await self._request("GET", "/object_info")

    async def upload_image(
        self, image_path: str, image_type: str = "input", overwrite: bool = True
    ) -> str:
        """Upload an image to ComfyUI's input directory."""
        from pathlib import Path

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        filename = Path(image_path).name

        session = await self._get_session()
        with open(image_path, "rb") as f:
            form_data = aiohttp.FormData()
            form_data.add_field("image", f, filename=filename, content_type="image/png")
            form_data.add_field("type", image_type)
            form_data.add_field("overwrite", str(overwrite).lower())

            async with session.post(
                f"http://{self.addr}/upload/image", data=form_data
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Image upload failed ({resp.status}): {error_text}")
                raw = await resp.read()
                response = json.loads(raw.decode("utf-8"))
                return response.get("name", filename)

    async def upload_video(self, video_path: str, overwrite: bool = True) -> str:
        """Upload a video to ComfyUI's input directory."""
        from pathlib import Path

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        filename = Path(video_path).name
        ext = Path(video_path).suffix.lower()
        content_type_map = {
            ".mp4": "video/mp4", ".webm": "video/webm",
            ".mov": "video/quicktime", ".avi": "video/x-msvideo",
        }
        content_type = content_type_map.get(ext, "video/mp4")

        session = await self._get_session()
        with open(video_path, "rb") as f:
            form_data = aiohttp.FormData()
            form_data.add_field("image", f, filename=filename, content_type=content_type)
            form_data.add_field("type", "input")
            form_data.add_field("overwrite", str(overwrite).lower())

            async with session.post(
                f"http://{self.addr}/upload/image", data=form_data
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise RuntimeError(f"Video upload failed ({resp.status}): {error_text}")
                raw = await resp.read()
                response = json.loads(raw.decode("utf-8"))
                return response.get("name", filename)

    async def interrupt(self) -> bool:
        """Interrupt current execution."""
        try:
            session = await self._get_session()
            async with session.post(f"http://{self.addr}/interrupt") as resp:
                return resp.status == 200
        except aiohttp.ClientError:
            return False

    async def clear_queue(self) -> bool:
        """Clear all pending prompts."""
        try:
            session = await self._get_session()
            async with session.post(
                f"http://{self.addr}/queue", json={"clear": True}
            ) as resp:
                return resp.status == 200
        except aiohttp.ClientError:
            return False

    async def connect_ws(self):
        """Create a websocket connection for progress monitoring."""
        session = aiohttp.ClientSession()
        try:
            ws = await session.ws_connect(
                f"http://{self.addr}/ws?clientId={self.client_id}",
                compress=0,
                heartbeat=30.0,
            )
        except Exception:
            await session.close()
            raise
        ws._session = session  # attach so we can close it later
        return ws


class Comfy:
    """Multi-instance ComfyUI client.

    Exposes an `acquire()` context manager that hands out a `SingleComfy`
    instance for the full lifetime of a job (uploads → queue → monitor →
    capture). The instance is returned to the pool when the context exits,
    so the next job in line picks a *truly* idle GPU instead of dispatching
    based on POST latency.
    """

    def __init__(self, addresses):
        self.addresses = parse_addresses(addresses)
        self.instances = [SingleComfy(addr) for addr in self.addresses]
        self._available: asyncio.Queue = asyncio.Queue()
        for inst in self.instances:
            self._available.put_nowait(inst)
        logger.info(
            f"ComfyUI client initialized with {len(self.instances)} instance(s): {self.addresses}"
        )

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[SingleComfy]:
        """Acquire an instance for the full duration of a job.

        Blocks until an instance is free. Routes uploads, queueing, and
        monitoring to the same instance, and prevents a second job from
        landing on this instance until the current one finishes.
        """
        if not self.instances:
            raise RuntimeError("No ComfyUI instances configured")
        instance = await self._available.get()
        try:
            yield instance
        finally:
            self._available.put_nowait(instance)

    async def get_object_info(self) -> Dict[str, Any]:
        """Get object info (samplers/schedulers/models). Uses the first instance.

        All instances are expected to expose equivalent node/model catalogs,
        so it doesn't matter which one we ask.
        """
        if not self.instances:
            raise RuntimeError("No ComfyUI instances configured")
        return await self.instances[0].get_object_info()

    async def interrupt_all(self) -> int:
        """Interrupt all instances. Returns count of successful interrupts."""
        results = await asyncio.gather(
            *[inst.interrupt() for inst in self.instances],
            return_exceptions=True,
        )
        return sum(1 for r in results if r is True)
