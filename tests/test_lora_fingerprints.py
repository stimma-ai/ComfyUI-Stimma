"""LoRA fingerprints identify a file's weights and never block lookups."""

import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stp_server.lora_fingerprints import (  # noqa: E402
    CHUNK_COUNT,
    CHUNK_SIZE,
    LoraFingerprints,
    fingerprint_file,
    pair_fingerprint,
)


def write_lora(path, data, metadata=None):
    header = json.dumps({"__metadata__": metadata or {}}).encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(data)


class FingerprintFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def fp(self, name, data, metadata=None):
        path = self.dir / name
        write_lora(path, data, metadata)
        return fingerprint_file(str(path))

    def test_same_weights_at_two_paths_match(self):
        data = os.urandom(5000)
        self.assertEqual(self.fp("a/one.safetensors", data), self.fp("b/two.safetensors", data))

    def test_metadata_edit_keeps_identity(self):
        data = os.urandom(5000)
        self.assertEqual(self.fp("a.safetensors", data), self.fp("b.safetensors", data, {"title": "renamed"}))

    def test_weight_change_differs_in_each_sampled_chunk(self):
        size = (CHUNK_COUNT + 2) * CHUNK_SIZE
        data = bytearray(os.urandom(size))
        base = self.fp("base.safetensors", bytes(data))
        for i in range(CHUNK_COUNT):
            changed = bytearray(data)
            changed[((size - CHUNK_SIZE) * i) // (CHUNK_COUNT - 1)] ^= 0xFF
            self.assertNotEqual(base, self.fp(f"c{i}.safetensors", bytes(changed)))

    def test_size_is_part_of_identity(self):
        self.assertNotEqual(self.fp("a.safetensors", b"\0" * 100), self.fp("b.safetensors", b"\0" * 101))

    def test_files_without_identity_are_null(self):
        (self.dir / "x.ckpt").write_bytes(b"12345678")
        self.assertIsNone(fingerprint_file(str(self.dir / "x.ckpt")))
        (self.dir / "short.safetensors").write_bytes(b"123")
        self.assertIsNone(fingerprint_file(str(self.dir / "short.safetensors")))
        (self.dir / "bad.safetensors").write_bytes(struct.pack("<Q", 10**9) + b"data")
        self.assertIsNone(fingerprint_file(str(self.dir / "bad.safetensors")))
        self.assertIsNone(fingerprint_file(str(self.dir / "missing.safetensors")))
        self.assertIsNone(self.fp("empty.safetensors", b""))

    def test_pair_is_order_independent_and_null_with_a_missing_side(self):
        self.assertEqual(pair_fingerprint("a", "b"), pair_fingerprint("b", "a"))
        self.assertTrue(pair_fingerprint("a", "b").startswith("stfp1p:"))
        self.assertIsNone(pair_fingerprint("a", None))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.cache = str(self.dir / "cache" / "fp.json")
        self.files = {}

    def add(self, name, data=None):
        path = self.dir / "loras" / name
        write_lora(path, data if data is not None else os.urandom(500))
        self.files[name] = str(path)

    def service(self):
        return LoraFingerprints(cache_path=self.cache, list_files=lambda: list(self.files.items()))

    def settle(self, service, names):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            found, pending = service.lookup(names)
            if pending == 0 and not service._scanning:
                return found
            time.sleep(0.01)
        self.fail("worker did not settle")

    def test_lookup_is_pending_then_resolves_and_persists(self):
        self.add("a.safetensors")
        service = self.service()
        found, pending = service.lookup(["a.safetensors"])
        self.assertEqual((found["a.safetensors"], pending), (None, 1))
        value = self.settle(service, ["a.safetensors"])["a.safetensors"]
        self.assertTrue(value.startswith("stfp1:"))
        self.assertEqual(json.loads(Path(self.cache).read_text())["entries"]["a.safetensors"]["fingerprint"], value)

    def test_cache_hit_avoids_reading_file(self):
        self.add("a.safetensors")
        first = self.service()
        first.kick()
        value = self.settle(first, ["a.safetensors"])["a.safetensors"]

        import stp_server.lora_fingerprints as module
        original = module.fingerprint_file
        module.fingerprint_file = MagicMock(side_effect=AssertionError("cache missed"))
        self.addCleanup(setattr, module, "fingerprint_file", original)
        second = self.service()
        self.assertEqual(self.settle(second, ["a.safetensors"])["a.safetensors"], value)

    def test_changed_file_is_rehashed_and_removed_file_pruned(self):
        self.add("a.safetensors")
        self.add("b.safetensors")
        service = self.service()
        before = self.settle(service, ["a.safetensors", "b.safetensors"])
        self.add("a.safetensors", os.urandom(800))
        del self.files["b.safetensors"]
        service.kick()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and service.lookup(["a.safetensors"])[0]["a.safetensors"] == before["a.safetensors"]:
            time.sleep(0.01)
        self.assertNotEqual(service.lookup(["a.safetensors"])[0]["a.safetensors"], before["a.safetensors"])
        self.settle(service, ["a.safetensors"])
        self.assertNotIn("b.safetensors", service._entries)

    def test_overwritten_file_stays_pending_until_rescanned(self):
        self.add("a.safetensors")
        service = self.service()
        before = self.settle(service, ["a.safetensors"])["a.safetensors"]
        self.add("a.safetensors", os.urandom(800))

        # Hold the worker inside the listing, so the rescan is visibly in progress.
        release = threading.Event()
        list_files = service._list_files

        def slow_list():
            release.wait(5)
            return list_files()

        service._list_files = slow_list
        service._last_complete = time.monotonic() - 1000
        found, pending = service.lookup(["a.safetensors"])
        self.assertEqual((found["a.safetensors"], pending), (before, 1))
        release.set()
        after = self.settle(service, ["a.safetensors"])["a.safetensors"]
        self.assertNotEqual(after, before)

    def test_name_the_scan_cannot_find_stops_being_pending(self):
        service = self.service()
        self.assertEqual(self.settle(service, ["gone.safetensors"])["gone.safetensors"], None)

    def test_unfingerprintable_file_is_null_not_pending(self):
        self.files["x.ckpt"] = str(self.dir / "x.ckpt")
        Path(self.files["x.ckpt"]).write_bytes(b"12345678")
        service = self.service()
        self.assertIsNone(self.settle(service, ["x.ckpt"])["x.ckpt"])


class RpcTests(unittest.TestCase):
    def call(self, tool, params):
        from stp_server.provider import StimmaPluginProvider

        stub = types.SimpleNamespace(
            _tool_registry=types.SimpleNamespace(get=lambda slug: tool if slug == "t" else None),
            _lora_fingerprints=types.SimpleNamespace(
                lookup=lambda names: ({n: None if n == "new" else "fp-" + n for n in names}, names.count("new"))
            ),
        )
        request = types.SimpleNamespace(id=1, params=params)
        return StimmaPluginProvider._handle_file_fingerprints(stub, request)

    def tool(self, items):
        return types.SimpleNamespace(parameters=[types.SimpleNamespace(name="loras", items=items)])

    def test_returns_fingerprint_per_enum_value(self):
        tool = self.tool({"properties": {"path": {"enum": ["a", "new"]}}})
        response = self.call(tool, {"tool_id": "t", "parameter": "loras"})
        self.assertEqual(response.result, {"fingerprints": {"a": "fp-a", "new": None}, "pending": 1})

    def test_paired_value_hashes_its_two_files(self):
        items = {
            "properties": {"path": {"enum": ["pair"]}},
            "x-pairs": [{"display": "pair", "high": "h", "low": "l"}],
        }
        response = self.call(self.tool(items), {"tool_id": "t", "parameter": "loras"})
        self.assertEqual(response.result["fingerprints"]["pair"], pair_fingerprint("fp-h", "fp-l"))

    def test_unknown_tool_or_parameter_is_invalid(self):
        tool = self.tool({})
        self.assertEqual(self.call(tool, {"tool_id": "nope", "parameter": "loras"}).error.code, -32602)
        self.assertEqual(self.call(tool, {"tool_id": "t", "parameter": "nope"}).error.code, -32602)


if __name__ == "__main__":
    unittest.main()
