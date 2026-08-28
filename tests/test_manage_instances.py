"""Focused tests for host GPU telemetry."""

import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stp_server.manage.instances import InstanceMonitor, InstanceStatus, gpu_stats, merge_comfy_gpu_memory
from stp_server.manage.manager import Manager, _running_jobs_view
from stp_server.manage.ops import Operation, STATE_FAILED, STATE_PAUSED, STATE_RUNNING


class TestGpuStats(unittest.TestCase):
    def test_keeps_gpu_when_nvidia_smi_memory_is_unavailable(self):
        output = "0, GPU-1, NVIDIA GB10, 7, [N/A], [N/A], 48\n"
        completed = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
        with patch("stp_server.manage.instances.shutil.which", return_value="nvidia-smi"), \
             patch("stp_server.manage.instances.subprocess.run", return_value=completed):
            gpus = gpu_stats()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["name"], "NVIDIA GB10")
        self.assertEqual(gpus[0]["util"], 7)
        self.assertEqual(gpus[0]["temp"], 48)
        self.assertIsNone(gpus[0]["mem_total"])

    def test_fills_unified_memory_from_comfyui_system_stats(self):
        gpus = [{"index": 0, "name": "NVIDIA GB10", "util": 3,
                 "mem_used": None, "mem_total": None, "temp": 47}]
        devices = [{"type": "cuda", "index": 0, "name": "cuda:0 NVIDIA GB10",
                    "vram_total": 128_000, "vram_free": 28_000}]

        merged = merge_comfy_gpu_memory(gpus, devices)

        self.assertEqual(merged[0]["mem_total"], 128_000)
        self.assertEqual(merged[0]["mem_used"], 100_000)
        self.assertTrue(merged[0]["unified_memory"])


class TestInstanceStartupState(unittest.TestCase):
    def test_unconfirmed_instance_is_checking_not_down(self):
        status = InstanceStatus(addr="127.0.0.1:8188", local=True)
        monitor = object.__new__(InstanceMonitor)
        monitor._statuses = {status.addr: status}

        summary = monitor.summary()

        self.assertEqual(summary["checking"], [status.addr])
        self.assertEqual(summary["down"], [])

    def test_instance_becomes_down_after_startup_retries(self):
        status = InstanceStatus(addr="127.0.0.1:8188", local=True, poll_attempts=15)
        monitor = object.__new__(InstanceMonitor)
        monitor._statuses = {status.addr: status}

        summary = monitor.summary()

        self.assertEqual(summary["checking"], [])
        self.assertEqual(summary["down"], [status.addr])

    def test_provider_is_not_degraded_while_only_instance_is_checking(self):
        manager = object.__new__(Manager)
        manager.instances = types.SimpleNamespace(summary=lambda: {
            "total": 1, "healthy": 0, "checking": ["127.0.0.1:8188"], "down": [],
        })
        manager._restart_needed = []
        manager._dismissed_failures = set()
        manager.ops = types.SimpleNamespace(all=lambda: [])

        self.assertEqual(manager.provider_state(), ("ready", None))

    def _manager_with_ops(self, ops):
        manager = object.__new__(Manager)
        manager.instances = types.SimpleNamespace(summary=lambda: {
            "total": 1, "healthy": 1, "checking": [], "down": [],
        })
        manager._restart_needed = []
        manager._dismissed_failures = set()
        manager.ops = types.SimpleNamespace(all=lambda: ops)
        return manager

    def test_running_management_operation_is_in_progress(self):
        op = Operation(id="download", kind="download", title="Download MiniMax H3", state=STATE_RUNNING)
        manager = self._manager_with_ops([op])

        self.assertEqual(manager.provider_state(), ("in_progress", "Download MiniMax H3"))

    def test_update_attention_does_not_change_provider_health(self):
        manager = self._manager_with_ops([])

        with patch(
            "stp_server.manage.manager.updater.cached_status",
            return_value={"update_available": True},
        ):
            self.assertEqual(manager.provider_state(), ("ready", None))
            self.assertEqual(manager.provider_attention(), "update_available")

    def test_newer_checkout_requires_comfyui_restart(self):
        manager = self._manager_with_ops([])

        with patch(
            "stp_server.manage.manager.updater.cached_status",
            return_value={"restart_required": True},
        ):
            self.assertEqual(
                manager.provider_state(),
                ("warning", "ComfyUI-Stimma updated. Restart ComfyUI to apply."),
            )
            self.assertEqual(manager.restart_reasons(), ["update"])

    def test_paused_or_failed_operation_does_not_degrade_provider(self):
        running = Operation(id="running", kind="download", title="Download model", state=STATE_RUNNING)
        paused = Operation(id="paused", kind="download", title="Download other model", state=STATE_PAUSED)
        manager = self._manager_with_ops([running, paused])
        self.assertEqual(manager.provider_state(), ("in_progress", "Download model"))

        failed = Operation(id="failed", kind="install_node", title="Install nodes", state=STATE_FAILED)
        manager = self._manager_with_ops([running, failed])
        self.assertEqual(manager.provider_state(), ("in_progress", "Download model"))

        manager = self._manager_with_ops([paused, failed])
        self.assertEqual(manager.provider_state(), ("ready", None))


class TestManagedUpdate(unittest.IsolatedAsyncioTestCase):
    async def test_update_restarts_comfyui_after_fast_forward(self):
        manager = object.__new__(Manager)
        operation = MagicMock()
        manager.ops = MagicMock()
        manager.ops.create.return_value = operation
        manager.provider = types.SimpleNamespace(push_state=AsyncMock())
        manager.restart = AsyncMock(return_value={"ok": True})

        with patch(
            "stp_server.manage.manager.updater.apply_update",
            new=AsyncMock(),
        ):
            result = await manager.apply_update()

        self.assertEqual(result, {"ok": True, "restarting": True})
        manager.restart.assert_awaited_once_with()
        manager.provider.push_state.assert_awaited_once_with(force=True)


class TestRunningJobsView(unittest.TestCase):
    def test_live_progress_overrides_sample_and_settled_job_disappears(self):
        status = types.SimpleNamespace(
            addr="127.0.0.1:8188",
            pending=2,
            running=[
                {"prompt_id": "live", "ours": True, "title": "Old", "progress": 0.2},
                {"prompt_id": "done", "ours": True, "title": "Done", "progress": 0.9},
                {"prompt_id": "external", "ours": False, "title": "External", "progress": None},
            ],
        )
        live = {
            "live": {"title": "Current", "request_id": "r1", "started_at": 1, "progress": 0.6},
            "new": {"title": "New", "request_id": "r2", "started_at": 2,
                    "progress": 0.1, "addr": "127.0.0.1:8188"},
        }

        running, pending = _running_jobs_view([status], live)

        self.assertEqual(pending, 2)
        by_id = {job["prompt_id"]: job for job in running}
        self.assertEqual(set(by_id), {"live", "new", "external"})
        self.assertEqual(by_id["live"]["title"], "Current")
        self.assertEqual(by_id["live"]["progress"], 0.6)


if __name__ == "__main__":
    unittest.main()
