"""Audio serialization, STP capture, and bundled audio workflow contracts."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_node_module(filename):
    spec = importlib.util.spec_from_file_location(filename, ROOT / "nodes" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


outputs = load_node_module("outputs.py")
params = load_node_module("params.py")
from stp_server.discovery import _convert_ui_to_api, _extract_stimma_nodes, DiscoveredWorkflow
from stp_server.executor import _capture_output, _capture_from_history, _inject_output_dir
from stp_server.tool_builder import _build_single_tool
from stp_server.transport import ComfyUITransport
from stp_server.comfy_client import SingleComfy
from stimma_tools_protocol.provider import normalize_output


class AudioOutputTests(unittest.TestCase):
    def test_stereo_sample_order_rate_and_clipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.wav"
            data = np.array([[[0.0, 0.5, 2.0], [-0.5, -2.0, 0.0]]], dtype=np.float32)
            outputs._write_audio_wav({"waveform": data, "sample_rate": 44100}, str(path))
            with wave.open(str(path)) as wav:
                self.assertEqual((wav.getnchannels(), wav.getframerate(), wav.getnframes()), (2, 44100, 3))
                self.assertEqual(np.frombuffer(wav.readframes(3), dtype="<i2").tolist(),
                                 [0, -16383, 16383, -32767, 32767, 0])

    def test_output_node_writes_wav_to_job_directory(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"folder_paths": types.ModuleType("folder_paths")}):
            result = outputs.StimmaAudioOutput().execute(
                {"waveform": np.ones((1, 1, 480), dtype=np.float32) * .1, "sample_rate": 48000},
                _stimma_output_dir=tmp,
            )
            filename = result["ui"]["audio"][0]["filename"]
            with wave.open(str(Path(tmp) / filename)) as wav:
                self.assertEqual((wav.getnchannels(), wav.getframerate(), wav.getnframes()), (1, 48000, 480))

    def test_invalid_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for data in [np.empty((0, 1, 4)), np.empty((1, 1, 0)), np.zeros((1, 2, 3, 4)), np.array([np.nan])]:
                with self.subTest(shape=data.shape):
                    self.assertIsNone(outputs._write_audio_wav({"waveform": data, "sample_rate": 48000}, str(Path(tmp) / "bad.wav")))

    def test_audio_extensions_are_typed_as_audio(self):
        for ext in ("wav", "mp3", "flac", "ogg", "opus", "m4a"):
            self.assertEqual(normalize_output({"asset_id": "clip." + ext})["assets"][0]["type"], "audio")

    def test_float_format_appends_without_shifting_existing_widgets(self):
        spec = params.StimmaFloatParam.INPUT_TYPES()["required"]
        self.assertEqual(list(spec), ["name", "value", "minimum", "maximum", "step", "ui_control", "ui_order", "ui_description", "ui_format"])
        self.assertEqual(params.StimmaFloatParam().execute("cfg", 4., 0., 20., .1, "slider", 1, "Guidance"), (4.,))


class AudioCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_targets_only_its_prompt(self):
        instance = SingleComfy("worker:8188")
        instance._request = AsyncMock(side_effect=[b"", b"", {"queue_running": [[1, "other-job"]], "queue_pending": []}])
        await instance.cancel_prompt("audio-job")
        calls = instance._request.await_args_list
        self.assertEqual(calls[0].args, ("POST", "/queue"))
        self.assertEqual(calls[0].kwargs["json"], {"delete": ["audio-job"]})
        self.assertEqual(calls[1].args, ("POST", "/interrupt"))
        self.assertEqual(calls[1].kwargs["json"], {"prompt_id": "audio-job"})

    async def test_temp_output_upload_preserves_bytes_and_extension(self):
        context = MagicMock()
        context.assets.upload = AsyncMock(return_value="result.wav")
        with tempfile.TemporaryDirectory() as tmp:
            data = b"RIFF-test-wave"
            (Path(tmp) / "stimma_output_0000.wav").write_bytes(data)
            result = await _capture_output(tmp, {}, context, MagicMock(), "job")
        context.assets.upload.assert_awaited_once_with(data, ".wav")
        self.assertEqual(result, {"asset_id": "result.wav"})

    async def test_history_audio_download_preserves_temp_type(self):
        instance = MagicMock(addr="worker:8188")
        instance.get_history = AsyncMock(return_value={"job": {"outputs": {"7": {"audio": [
            {"filename": "audio.flac", "subfolder": "preview", "type": "temp"}
        ]}}}})
        context = MagicMock()
        context.assets.upload = AsyncMock(return_value="result.flac")
        response = MagicMock(status=200)
        response.read = AsyncMock(return_value=b"fLaC-audio")
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.get.return_value = response
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        with patch("aiohttp.ClientSession", return_value=session):
            result = await _capture_from_history(instance, "job", context)
        session.get.assert_called_once_with("http://worker:8188/view", params={"filename": "audio.flac", "type": "temp", "subfolder": "preview"})
        context.assets.upload.assert_awaited_once_with(b"fLaC-audio", ".flac")
        self.assertEqual(result, {"asset_id": "result.flac"})

    async def test_registration_precedes_broadcasts_on_new_connections(self):
        transport = ComfyUITransport(MagicMock())
        established, new = MagicMock(), MagicMock()
        established.send_str = AsyncMock()
        new.send_str = AsyncMock()
        transport._clients.update([established, new])
        transport._awaiting_registration.add(new)
        progress = json.dumps({"method": "tools.progress"})
        register = json.dumps({"method": "provider.register"})
        await transport.send(progress)
        new.send_str.assert_not_awaited()
        established.send_str.assert_awaited_once_with(progress)
        await transport.send(register)
        await transport.send(progress)
        self.assertEqual([c.args[0] for c in new.send_str.await_args_list], [register, progress])


class AudioWorkflowTests(unittest.TestCase):
    def test_bundled_graphs_expose_controls_and_audio_output(self):
        # Metadata nodes use real definitions. Other graph sockets are recovered
        # from the saved UI workflow; live ComfyUI tests validate native schemas.
        info = {}
        for name in ("StimmaIntParam", "StimmaFloatParam", "StimmaStringParam", "StimmaBoolParam"):
            info[name] = {"input": getattr(params, name).INPUT_TYPES()}
        info["StimmaPromptParam"] = {"input": {"required": dict(name=("STRING",), default_text=("STRING",), required=("BOOLEAN",), ui_order=("INT",), ui_description=("STRING",))}}
        info["StimmaSeedParam"] = {"input": {"required": dict(name=("STRING",), value=("INT",), ui_order=("INT",))}}
        info["StimmaToolInfo"] = {"input": load_node_module("tool_info.py").StimmaToolInfo.INPUT_TYPES()}
        info["StimmaLayoutGroup"] = {"input": load_node_module("layout.py").StimmaLayoutGroup.INPUT_TYPES()}
        info["StimmaAudioOutput"] = {"input": outputs.StimmaAudioOutput.INPUT_TYPES()}
        for filename, default_steps, max_duration in [
            ("Stimma-Stable-Audio-3-Medium.json", 8, 380),
            ("Stimma-Stable-Audio-3-Medium-Base.json", 50, 380),
            ("Stimma-MOSS-SoundEffect-v2.json", 100, 30),
        ]:
            with self.subTest(workflow=filename):
                data = json.loads((ROOT / "workflows" / filename).read_text())
                api = _convert_ui_to_api(data, info)
                extracted = _extract_stimma_nodes(api)
                workflow = DiscoveredWorkflow(file_path=filename, api_prompt=api, **extracted)
                descriptor = _build_single_tool(workflow, info, object(), object()).to_descriptor()
                props = descriptor.parameter_schema["properties"]
                self.assertIn("text-to-audio", descriptor.task_types)
                self.assertEqual(props["steps"]["default"], default_steps)
                self.assertEqual(props["duration"]["maximum"], max_duration)
                self.assertEqual(props["duration"]["x-format"], "seconds")
                self.assertIn("seed", props)
                self.assertEqual(len(workflow.output_nodes), 1)
                output = workflow.output_nodes[0]
                self.assertEqual(output["class_type"], "StimmaAudioOutput")
                _inject_output_dir(api, workflow, "job-output")
                self.assertEqual(api[output["node_id"]]["inputs"]["_stimma_output_dir"], "job-output")
                self.assertEqual("negative_prompt" in props, "-Base" in filename or "MOSS" in filename)
                group = descriptor.metadata["layout"][0]
                self.assertTrue(group["collapsed"])
                self.assertTrue(all(p["name"] in props for p in group["params"]))


if __name__ == "__main__":
    unittest.main()
