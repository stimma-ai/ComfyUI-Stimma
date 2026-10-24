"""Qwen speech controls, reference routing, and directory-based model setup."""

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audio_runtime import qwen_tts_dependencies
from stp_server.discovery import (
    DiscoveredWorkflow, _collect_model_hints, _convert_ui_to_api,
    _extract_stimma_nodes, _resolve_stimma_links, _validate_workflow,
)
from stp_server.manage.resolve import resolve_source
from stp_server.tool_builder import _build_single_tool


class QwenTTSTests(unittest.TestCase):
    def test_workflows_keep_spoken_text_separate_from_direction_and_reference(self):
        info = json.loads((ROOT / "tests/fixtures/tts-object-info.json").read_text())
        for variant in ("Custom-Voice", "Voice-Design", "Voice-Clone"):
            with self.subTest(variant=variant):
                path = ROOT / f"workflows/Stimma-Qwen3-TTS-{variant}.json"
                data = json.loads(path.read_text())
                api = _convert_ui_to_api(data, info)
                self.assertEqual(_validate_workflow(api, info), [])
                workflow = DiscoveredWorkflow(file_path=str(path), api_prompt=api, **_extract_stimma_nodes(api))
                descriptor = _build_single_tool(workflow, info, object(), object()).to_descriptor()
                props = descriptor.parameter_schema["properties"]
                self.assertEqual(descriptor.task_types, ["text-to-speech"])
                self.assertEqual("input_audios" in props, variant == "Voice-Clone")
                self.assertEqual("instruct" in props, variant != "Voice-Clone")
                self.assertEqual("speaker" in props, variant == "Custom-Voice")
                self.assertIn("English", props["language"]["enum"])
                for group in descriptor.metadata["layout"]:
                    self.assertTrue(all(p["name"] in props for p in group["params"]))
                _resolve_stimma_links(api)
                generate = next(n for n in api.values() if n["class_type"].startswith("FB_Qwen3TTS"))
                self.assertEqual(generate["inputs"]["model_choice"], "1.7B")
                self.assertTrue(generate["inputs"]["unload_model_after_generate"])
                if variant == "Voice-Clone":
                    transcript = api[generate["inputs"]["ref_text"][0]]
                    self.assertEqual(transcript["class_type"], "Qwen3ASRTranscribe")
                    self.assertEqual(transcript["inputs"]["audio"], generate["inputs"]["ref_audio"])

    def test_model_subdirectory_hints_do_not_collide_on_shared_basenames(self):
        data = json.loads((ROOT / "workflows/Stimma-Qwen3-TTS-Custom-Voice.json").read_text())
        hints = _collect_model_hints(data)
        for node in data["nodes"]:
            for model in node.get("properties", {}).get("models", []):
                source = resolve_source(model["name"], hints)
                self.assertEqual(source["url"], model["url"])
                self.assertEqual(source["sha256"], model["hash"])

    def test_first_install_tracks_every_file_and_extra_model_paths(self):
        info = json.loads((ROOT / "tests/fixtures/tts-object-info.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alternate = root / "shared-models"
            alternate.mkdir()
            folder_paths = types.SimpleNamespace(models_dir=str(root), get_folder_paths=lambda _: [str(alternate)])
            prompt = {"1": {"class_type": "FB_Qwen3TTSCustomVoice", "inputs": {"model_choice": "1.7B"}}}
            with patch.dict(sys.modules, {"folder_paths": folder_paths}):
                issues = []
                self.assertEqual(len(_validate_workflow(prompt, info, issues)), 15)
                for issue in issues:
                    source = resolve_source(issue["name"])
                    self.assertEqual(source["folder"], "qwen-tts")
                    self.assertEqual(len(source["sha256"]), 64)
                    dest = alternate / issue["name"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.touch()
                self.assertEqual(_validate_workflow(prompt, info), [])
            deps = qwen_tts_dependencies([alternate], "FB_Qwen3TTSCustomVoice")
            self.assertTrue(all(d["installed"] for d in deps))


if __name__ == "__main__":
    unittest.main()
