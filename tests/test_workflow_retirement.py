"""Retire unchanged bundled workflows without deleting user work."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "workflow_install", Path(__file__).resolve().parents[1] / "stp_server/workflow_install.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RetirementTests(unittest.TestCase):
    def run_migration(self, modified=False, tracked=True):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "plugin/workflows").mkdir(parents=True)
            dest = root / "user/default/workflows/Stimma"
            dest.mkdir(parents=True)
            name = "Stimma-Qwen-Image-2.1-Edit.json"
            original = b'{"nodes":[]}'
            payload = b'{"nodes":["user edit"]}' if modified else original
            (dest / name).write_bytes(payload)
            entries = {name: {"hash": hashlib.sha256(original).hexdigest()}} if tracked else {}
            manifest = dest / ".stimma-manifest.json"
            manifest.write_text(json.dumps({"version": 1, "files": entries}))
            folders = types.SimpleNamespace(get_user_directory=lambda: str(root / "user"))
            with patch.dict(sys.modules, {"folder_paths": folders}), patch.object(
                module, "__file__", str(root / "plugin/stp_server/workflow_install.py")
            ):
                module.sync_bundled_workflows()
                module.sync_bundled_workflows()  # Idempotent on subsequent starts.
            exists = (dest / name).exists()
            if modified or not tracked:
                self.assertEqual((dest / name).read_bytes(), payload)
            return exists, json.loads(manifest.read_text())["files"]

    def test_removes_unchanged_managed_copy(self):
        exists, entries = self.run_migration()
        self.assertFalse(exists)
        self.assertEqual(entries, {})

    def test_preserves_user_modified_copy(self):
        exists, entries = self.run_migration(modified=True)
        self.assertTrue(exists)
        self.assertIn("Stimma-Qwen-Image-2.1-Edit.json", entries)

    def test_preserves_untracked_copy(self):
        exists, _ = self.run_migration(tracked=False)
        self.assertTrue(exists)


if __name__ == "__main__":
    unittest.main()
