"""Focused tests for host GPU telemetry."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stp_server.manage.instances import gpu_stats, merge_comfy_gpu_memory


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


if __name__ == "__main__":
    unittest.main()
