"""Exercise model downloads and disk preflight through a real HTTP server."""

import hashlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stp_server.manage import downloads
from stp_server.manage.ops import OperationRegistry, STATE_DONE, STATE_FAILED


class TestModelDownloads(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.payload = b"model download regression fixture" * 1024
        self.sha256 = hashlib.sha256(self.payload).hexdigest()
        self.dest = self.root / "loras" / "krea2" / "krea2_darkbrush.safetensors"
        self.part = Path(str(self.dest) + ".part")
        self.ranges = []
        self.ignore_range = False
        app = web.Application()
        app.router.add_get("/model", self.serve_model)
        self.server = TestServer(app)
        await self.server.start_server()
        self.addAsyncCleanup(self.server.close)
        self.ops = OperationRegistry(self.root / "operations.json")
        self.manager = downloads.DownloadManager(self.ops)
        self.addAsyncCleanup(self.manager.stop)

    async def serve_model(self, request):
        requested_range = request.headers.get("Range")
        self.ranges.append(requested_range)
        if requested_range and not self.ignore_range:
            start = int(requested_range.removeprefix("bytes=").removesuffix("-"))
            return web.Response(
                status=206,
                body=self.payload[start:],
                headers={"Content-Range": f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"},
            )
        return web.Response(body=self.payload)

    def seed_partial_download(self):
        self.dest.parent.mkdir(parents=True)
        self.part.write_bytes(self.payload[:len(self.payload) // 2])

    async def download(self):
        operation = self.manager.enqueue(
            filename="krea2/krea2_darkbrush.safetensors",
            url=str(self.server.make_url("/model")),
            dest_path=str(self.dest),
            sha256=self.sha256,
        )
        await self.manager._tasks[operation.id]
        return operation

    def assert_downloaded(self, operation):
        self.assertEqual(operation.state, STATE_DONE, operation.error)
        self.assertEqual(self.dest.read_bytes(), self.payload)
        self.assertFalse(self.part.exists())
        self.assertEqual(operation.meta["bytes"], len(self.payload))
        self.assertEqual(operation.meta["size"], len(self.payload))
        self.assertIsNone(operation.error)

    async def test_downloads_and_verifies_without_os_statvfs(self):
        with patch.object(downloads, "os", types.SimpleNamespace(replace=os.replace)):
            operation = await self.download()

        self.assert_downloaded(operation)
        self.assertEqual(self.ranges, [None])
        restored = OperationRegistry(self.root / "operations.json")
        self.assertEqual(restored.get(operation.id).state, STATE_DONE)

    async def test_insufficient_space_fails_before_writing(self):
        with patch("shutil.disk_usage", return_value=types.SimpleNamespace(free=0)) as disk_usage:
            operation = await self.download()

        self.assertEqual(operation.state, STATE_FAILED)
        self.assertEqual(operation.error_kind, "disk")
        self.assertIn("Not enough disk space", operation.error)
        self.assertEqual(operation.fix, {"action": "retry"})
        self.assertFalse(self.dest.exists())
        self.assertFalse(self.part.exists())
        self.assertEqual(Path(disk_usage.call_args.args[0]), self.dest.parent)

    async def test_resuming_requires_space_only_for_remaining_bytes(self):
        self.seed_partial_download()
        have = self.part.stat().st_size
        free = len(self.payload) - have + 512 * 1024 * 1024

        with patch("shutil.disk_usage", return_value=types.SimpleNamespace(free=free)):
            operation = await self.download()

        self.assert_downloaded(operation)
        self.assertEqual(self.ranges, [f"bytes={have}-"])

    async def test_insufficient_space_preserves_partial_download(self):
        self.seed_partial_download()
        original = self.part.read_bytes()

        with patch("shutil.disk_usage", return_value=types.SimpleNamespace(free=0)):
            operation = await self.download()

        self.assertEqual(operation.state, STATE_FAILED)
        self.assertEqual(operation.error_kind, "disk")
        self.assertEqual(self.part.read_bytes(), original)
        self.assertFalse(self.dest.exists())

    async def test_ignored_range_requires_space_for_full_download(self):
        self.seed_partial_download()
        self.ignore_range = True
        free = len(self.payload) - self.part.stat().st_size + 512 * 1024 * 1024

        with patch("shutil.disk_usage", return_value=types.SimpleNamespace(free=free)):
            operation = await self.download()

        self.assertEqual(operation.state, STATE_FAILED)
        self.assertEqual(operation.error_kind, "disk")
        self.assertFalse(self.dest.exists())

    async def test_disk_usage_oserror_does_not_block_download(self):
        with patch("shutil.disk_usage", side_effect=OSError("Disk statistics unavailable")):
            operation = await self.download()

        self.assert_downloaded(operation)


if __name__ == "__main__":
    unittest.main()
