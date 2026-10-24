"""Drama 3 workflow conversion, voice routing, and first-install setup."""

import asyncio
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stp_server.discovery import (
    DiscoveredWorkflow, _convert_ui_to_api, _extract_stimma_nodes,
    _resolve_stimma_links, _validate_workflow,
)
from stp_server.manage import nodes, resolve
from stp_server.tool_builder import _build_single_tool


def object_info():
    # Compact live /object_info contracts from ComfyUI-Drama3 67a733a.
    return json.loads((ROOT / "tests/fixtures/tts-object-info.json").read_text())


class Drama3Tests(unittest.TestCase):
    def test_graphs_convert_and_route_each_reference_to_its_speaker(self):
        for variant, voices in (("TTS", 0), ("Voice-Clone", 1), ("Dialogue", 2)):
            with self.subTest(variant=variant):
                path = ROOT / f"workflows/Stimma-Fish-Audio-Drama-3-{variant}.json"
                data = json.loads(path.read_text())
                by_id = {n["id"]: n for n in data["nodes"]}
                for lid, source, slot, target, input_slot, _ in data["links"]:
                    self.assertIn(lid, by_id[source]["outputs"][slot]["links"])
                    self.assertEqual(by_id[target]["inputs"][input_slot]["link"], lid)
                info = object_info()
                api = _convert_ui_to_api(data, info)
                self.assertEqual(_validate_workflow(api, info), [])
                workflow = DiscoveredWorkflow(file_path=str(path), api_prompt=api, **_extract_stimma_nodes(api))
                descriptor = _build_single_tool(workflow, info, object(), object()).to_descriptor()
                self.assertEqual(descriptor.task_types, ["text-to-speech"])
                props = descriptor.parameter_schema["properties"]
                self.assertEqual("input_audios" in props, bool(voices))
                if voices:
                    self.assertIn("input_audios", descriptor.parameter_schema["required"])
                    self.assertIn("speaker 0", props["input_audios"]["description"])
                for group in descriptor.metadata["layout"]:
                    self.assertTrue(all(p["name"] in props for p in group["params"]))

                prompt_id = next(k for k, n in api.items() if n["class_type"] == "StimmaPromptParam")
                api[prompt_id]["inputs"]["default_text"] = "Only these words are spoken."
                _resolve_stimma_links(api)
                generate = next(n["inputs"] for n in api.values() if n["class_type"] == "Drama3Generate")
                self.assertEqual(generate["text"], "Only these words are spoken.")
                voice_links = {k: v for k, v in generate.items() if k.startswith("voices.")}
                self.assertEqual(set(voice_links), {f"voices.voice{i}" for i in range(voices)})
                reference_ids = set()
                for link in voice_links.values():
                    encode = api[link[0]]
                    self.assertEqual(encode["class_type"], "Drama3EncodeVoice")
                    audio = encode["inputs"]["audio"]
                    reference_ids.add(audio[0])
                    transcribe = api[encode["inputs"]["transcript"][0]]
                    self.assertEqual(transcribe["class_type"], "Qwen3ASRTranscribe")
                    self.assertEqual(transcribe["inputs"]["audio"], audio)
                self.assertEqual(len(reference_ids), voices)
                self.assertEqual(workflow.output_nodes[0]["class_type"], "StimmaAudioOutput")

    def test_empty_model_folders_produce_downloadable_dependencies(self):
        info = object_info()
        for typ, field, folder in (
            ("Drama3Loader", "model_name", "diffusion_models"),
            ("Drama3CodecLoader", "codec_name", "vae"),
            ("Qwen3ASRLoader", "model_name", "asr"),
        ):
            spec = info[typ]["input"]["required"][field]
            filename = spec[1]["options"][0]
            empty = copy.deepcopy(info)
            empty[typ]["input"]["required"][field][1]["options"] = []
            issues = []
            warnings = _validate_workflow({"1": {"class_type": typ, "inputs": {field: filename}}}, empty, issues)
            self.assertEqual(len(warnings), 1)
            self.assertEqual(issues[0]["folder"], folder)
            source = resolve.resolve_source(filename)
            self.assertEqual(source["folder"], folder)
            self.assertEqual(len(source["sha256"]), 64)
            self.assertNotIn("/main/", source["url"])

    def test_setup_finds_one_pack_without_a_manager_index_update(self):
        async def check():
            for typ in object_info():
                if typ.startswith(("Drama3", "Qwen3ASR")):
                    pack = await nodes.lookup_pack(typ)
                    self.assertEqual(pack["url"], "https://github.com/PedroMarinhoDev/ComfyUI-Drama3")

        with patch.object(nodes, "_ensure_node_map", side_effect=AssertionError("no index needed")):
            asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
