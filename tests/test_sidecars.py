"""Generic publication and optional companion transport: no model dependencies."""
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

# Import extension modules outside patch.dict: restoring sys.modules must not
# unload/reimport torch between the two real video-output checks.
try:
    import torch
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stp_server.executor import _capture_output


class SidecarCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_outputs_keep_two_unrelated_formats_and_optional_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = []
            for i, fmt in enumerate(("opaque.one", "unrelated.two")):
                folder = root / str(i)
                folder.mkdir()
                (folder / "stimma_output_0000.mp4").write_bytes(f"render{i}".encode())
                (folder / "raw.bin").write_bytes(f"state{i}".encode())
                assets.append({"file": "stimma_output_0000.mp4", "type": "video", "sidecars": [
                    {"file": "raw.bin", "format": fmt, "format_version": i+1, "descriptor": {"custom": [i]}},
                    {"file": "missing.bin", "format": "optional", "format_version": 1}]})
                (folder / "out.manifest.json").write_text(json.dumps({"assets": [assets[-1]]}))
            context = SimpleNamespace(assets=SimpleNamespace(upload=AsyncMock(side_effect=lambda data, ext: data.decode()+ext)),
                _provider=SimpleNamespace(host_wants_media_sidecars=True))
            result = await _capture_output(tmp, {}, context, None, "job")
            assert len(result["assets"]) == 2
            for i, asset in enumerate(result["assets"]):
                assert asset["asset_id"] == f"render{i}.mp4"
                assert asset["sidecars"][0]["asset_id"] == f"state{i}.bin"
                assert asset["sidecars"][0]["descriptor"] == {"custom": [i]}
                assert asset["sidecars"][1]["status"] == "unavailable"
            context._provider.host_wants_media_sidecars = False
            legacy = await _capture_output(tmp, {}, context, None, "job")
            assert all("sidecars" not in asset for asset in legacy["assets"])




class SidecarInputAndRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_companion_is_delivered_to_adapter_as_unavailable(self):
        from stp_server.executor import _inject_fields
        wf = SimpleNamespace(tool_info={"task_types": ["reference-to-video"]}, field_nodes=[
            {"node_id": "pin", "class_type": "StimmaKeyframesParam", "name": "keyframes", "inputs": {}}])
        context = SimpleNamespace(request_id="test", assets=SimpleNamespace(download=AsyncMock(side_effect=OSError("missing"))),
                                 input_sidecars={"render.mp4": [{"asset_id": "raw.bin", "format": "other", "format_version": 1, "media_mapping": {"preserved": True}}]})
        prompt = {"pin": {"inputs": {}}}
        with patch("stp_server.executor._download_and_upload_video", AsyncMock(return_value="healthy.mp4")):
            await _inject_fields(prompt, wf, {"keyframes": [{"media": "render.mp4", "at": 0}]}, context, None)
        [pin] = json.loads(prompt["pin"]["inputs"]["_stimma_keyframes"])
        assert pin["file"] == "healthy.mp4"
        assert pin["sidecars"][0]["status"] == "unavailable"
        assert pin["sidecars"][0]["media_mapping"] == {"preserved": True}

    async def test_completed_capture_retry_never_queues_a_prompt(self):
        from stp_server.executor import execute_workflow
        class Lease:
            async def __aenter__(self): return SimpleNamespace(queue_prompt=AsyncMock(side_effect=AssertionError("resampled")))
            async def __aexit__(self, *args): pass
        with tempfile.TemporaryDirectory() as tmp:
            wf = SimpleNamespace(warnings=[], api_prompt={}, tool_info={"slug": "test"})
            key = hashlib.sha256(json.dumps(["retry", {}], sort_keys=True).encode()).hexdigest()
            receipt = Path(tmp) / "stimma-capture-receipts" / (key + ".json")
            receipt.parent.mkdir()
            rendered = Path(tmp) / "rendered"
            rendered.mkdir()
            (rendered / "stimma_output_0000.mp4").write_bytes(b"completed generation")
            receipt.write_text(json.dumps({"output_dir": str(rendered), "prompt_id": "original", "generation_time": 321}))
            context = SimpleNamespace(request_id="retry", assets=SimpleNamespace(upload=AsyncMock(return_value="result.mp4")),
                                      _provider=SimpleNamespace(host_wants_media_sidecars=True))
            provider = SimpleNamespace(object_info={}, comfy_client=SimpleNamespace(acquire=lambda: Lease()))
            with patch("tempfile.gettempdir", return_value=tmp):
                result = await execute_workflow(context, wf, {"media": "new-wire-reference"}, provider)
            assert result["assets"][0]["asset_id"] == "result.mp4"
            assert result["generation_time"] == 321
            assert not receipt.exists()


class VideoMuxTests(unittest.TestCase):
    def test_short_audio_does_not_drop_final_video_frame(self):
        import importlib
        import types
        import numpy as np
        import subprocess
        package = types.ModuleType("publication_nodes")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "nodes")]
        folder_paths = types.ModuleType("folder_paths")
        with patch.dict(sys.modules, {"publication_nodes": package, "folder_paths": folder_paths}), tempfile.TemporaryDirectory() as tmp:
            module = importlib.import_module("publication_nodes.outputs")
            audio = {"waveform": np.zeros((1, 2, int(6.575*32000)), dtype=np.float32), "sample_rate": 32000}
            result = module.StimmaVideoOutput().execute(np.zeros((158, 48, 64, 3), dtype=np.float32), fps=24, audio=audio, _stimma_output_dir=tmp)
            path = Path(tmp) / result["ui"]["videos"][0]["filename"]
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=nb_frames", "-of", "json", str(path)]))
            assert info["streams"][0]["nb_frames"] == "158"

class OptionalPublicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_manifest_write_failure_still_captures_healthy_video(self):
        import importlib
        import types
        import numpy as np
        package = types.ModuleType("failed_publication_nodes")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "nodes")]
        original_write = Path.write_text
        def fail_manifest(path, *args, **kwargs):
            if path.name.endswith(".manifest.tmp"):
                raise OSError("manifest storage unavailable")
            return original_write(path, *args, **kwargs)
        with patch.dict(sys.modules, {"failed_publication_nodes": package, "folder_paths": types.ModuleType("folder_paths")}), tempfile.TemporaryDirectory() as tmp:
            module = importlib.import_module("failed_publication_nodes.outputs")
            with patch.object(Path, "write_text", fail_manifest):
                result = module.StimmaVideoOutput().execute(np.zeros((24, 48, 64, 3), dtype=np.float32), fps=24, _stimma_output_dir=tmp)
            assert "stimma_manifests" not in result["ui"]
            context = SimpleNamespace(assets=SimpleNamespace(upload=AsyncMock(return_value="healthy.mp4")),
                                      _provider=SimpleNamespace(host_wants_media_sidecars=True))
            captured = await _capture_output(tmp, {}, context, None, "job")
            assert captured["assets"] == [{"asset_id": "healthy.mp4", "type": "video", "role": "primary"}]
            assert context.assets.upload.await_args.args[0][4:8] == b"ftyp"


if __name__ == "__main__":
    unittest.main()
