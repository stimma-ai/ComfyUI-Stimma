"""Focused tests for ComfyUI-Manager detection and installation."""

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stp_server.manage import nodes


class _Stdout:
    def __init__(self, lines=()):
        self._lines = [f"{line}\n".encode() for line in lines]

    async def readline(self):
        return self._lines.pop(0) if self._lines else b""


class _Process:
    def __init__(self, returncode=0):
        self.stdout = _Stdout(["Cloning"])
        self._returncode = returncode

    async def wait(self):
        return self._returncode


class TestComfyUIManager(unittest.TestCase):
    def test_detects_packaged_manager_and_uses_its_module_cli(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                base = Path(td) / "ComfyUI"
                base.mkdir()
                site = Path(td) / "site-packages"
                package = site / "comfyui_manager"
                package.mkdir(parents=True)
                cli = site / "cm_cli"
                cli.mkdir()
                (cli / "__main__.py").write_text("")
                dist = types.SimpleNamespace(locate_file=lambda path: site / path)
                calls = []

                async def fake_exec(*args, **kwargs):
                    calls.append((args, kwargs))
                    return _Process()

                with patch.dict(sys.modules, {"folder_paths": types.SimpleNamespace(base_path=str(base))}), \
                     patch.object(nodes.metadata, "distribution", return_value=dist), \
                     patch.object(nodes.metadata, "version", return_value="4.2.2"), \
                     patch.object(nodes.asyncio, "create_subprocess_exec", fake_exec):
                    self.assertTrue(nodes.has_manager())
                    self.assertEqual(nodes.manager_dir(), package)
                    self.assertEqual(nodes.manager_version(), "4.2.2")
                    await nodes.install_pack("https://example.com/nodes", lambda _line: None)

                self.assertEqual(calls[0][0], (sys.executable, "-m", "cm_cli", "install", "https://example.com/nodes"))
                self.assertEqual(calls[0][1]["env"]["COMFYUI_PATH"], str(base))

        asyncio.run(run())

    def test_package_without_cli_is_not_install_capable(self):
        with tempfile.TemporaryDirectory() as td:
            site = Path(td)
            (site / "comfyui_manager").mkdir()
            dist = types.SimpleNamespace(locate_file=lambda path: site / path)
            with patch.dict(sys.modules, {"folder_paths": types.SimpleNamespace(base_path=td)}), \
                 patch.object(nodes.metadata, "distribution", return_value=dist):
                self.assertFalse(nodes.has_manager())

    def test_bundled_pack_override_resolves_before_manager_index(self):
        async def run():
            with patch.object(nodes, "_ensure_node_map", side_effect=AssertionError("index should not be needed")):
                return await nodes.lookup_pack("SpectrumApplyMiniMaxH3")

        result = asyncio.run(run())

        self.assertEqual(result["title"], "Spectrum for MiniMax H3")
        self.assertEqual(
            result["url"],
            "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3",
        )

    def test_detection_requires_cm_cli_and_reads_version(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            manager = base / "custom_nodes" / "comfyui-manager"
            manager.mkdir(parents=True)
            (manager / "pyproject.toml").write_text('[project]\nversion = "3.42.1"\n')
            folder_paths = types.SimpleNamespace(base_path=str(base))
            with patch.dict(sys.modules, {"folder_paths": folder_paths}):
                self.assertFalse(nodes.has_manager())
                (manager / "cm-cli.py").write_text("")
                self.assertTrue(nodes.has_manager())
                self.assertEqual(nodes.manager_version(), "3.42.1")

    def test_install_clones_to_documented_directory_atomically(self):
        async def run():
            with tempfile.TemporaryDirectory() as td:
                base = Path(td)
                (base / "custom_nodes").mkdir()
                folder_paths = types.SimpleNamespace(base_path=str(base))
                calls = []

                async def fake_exec(*args, **kwargs):
                    calls.append((args, kwargs))
                    temp = Path(args[-1])
                    temp.mkdir()
                    (temp / "cm-cli.py").write_text("")
                    return _Process()

                with patch.dict(sys.modules, {"folder_paths": folder_paths}), \
                     patch.object(nodes.asyncio, "create_subprocess_exec", fake_exec):
                    result = await nodes.install_manager(lambda _line: None)

                self.assertEqual(result, base / "custom_nodes" / "comfyui-manager")
                self.assertTrue((result / "cm-cli.py").exists())
                self.assertEqual(calls[0][0][:4], ("git", "clone", "--depth", "1"))
                self.assertFalse(any(p.name.startswith(".stimma-manager-install-") for p in result.parent.iterdir()))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
