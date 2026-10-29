"""Focused tests for descriptor construction from Stimma workflow fields.

Run: python tests/test_tool_builder.py
"""

import json
import os
import sys
import types
import unittest
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

package = types.ModuleType("stp_server")
package.__path__ = [os.path.join(ROOT, "stp_server")]
sys.modules["stp_server"] = package
config_module = types.ModuleType("stp_server.config")
config_module.Config = type("Config", (), {})
sys.modules["stp_server.config"] = config_module

from stp_server.discovery import (
    DiscoveredWorkflow,
    _convert_ui_to_api,
    _extract_stimma_nodes,
    _validate_workflow,
)
from stp_server.tool_builder import _build_single_tool


def field(node_id, class_type, inputs):
    return {"node_id": node_id, "class_type": class_type, "inputs": inputs}


class TestReferenceToVideoDescriptor(unittest.TestCase):
    def test_bundled_image_resolution_contract(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("resolution_params", Path(ROOT) / "nodes/params.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        required = module.StimmaResolutionParam.INPUT_TYPES()["required"]
        self.assertEqual(list(required)[:7], ["width", "height", "min_size", "max_size", "step", "supported_resolutions", "ui_order"])
        self.assertEqual(list(required)[7], "slider_max_pixels")
        count = 0
        for path in (Path(ROOT) / "workflows").glob("*.json"):
            nodes = json.loads(path.read_text())["nodes"]
            if not any(n["type"] == "StimmaToolInfo" and "text-to-image" in n["widgets_values"][2] for n in nodes):
                continue
            for node in nodes:
                if node["type"] != "StimmaResolutionParam":
                    continue
                converted = _convert_ui_to_api({"nodes": [node], "links": []},
                    {"StimmaResolutionParam": {"input": {"required": required}}})
                inputs = converted[str(node["id"])]["inputs"]
                workflow = DiscoveredWorkflow(file_path=path.name, api_prompt={},
                    tool_info={"slug": "resolution-test", "display_name": "Resolution", "task_types": ["text-to-image"], "description": ""},
                    field_nodes=[field("size", "StimmaResolutionParam", inputs)])
                descriptor = _build_single_tool(workflow, object_info=None, config=object(), provider=object()).to_descriptor()
                props = descriptor.parameter_schema["properties"]
                self.assertEqual(props["width"]["maximum"], 4096, path.name)
                self.assertEqual(props["height"]["maximum"], 4096, path.name)
                self.assertEqual(props["width"]["x-resolution-slider-max-pixels"], 4194304, path.name)
                self.assertNotIn("x-supported_resolutions", props["width"], path.name)
                self.assertEqual(module.StimmaResolutionParam().execute(3136, 1344, **{k: v for k, v in inputs.items() if k not in ("width", "height")}), (3136, 1344))
                count += 1
        self.assertEqual(count, 20)
        self.assertEqual(module.StimmaResolutionParam().execute(1024, 1024, 256, 2048, 64, "", 3), (1024, 1024))

    def test_typed_sections_are_optional_but_one_is_required(self):
        fields = [field("prompt", "StimmaPromptParam", {
            "name": "prompt", "default_text": "", "required": True, "ui_order": 0,
        })]
        fields.extend(
            field(f"image-{i}", "StimmaImageParam", {
                "required": False, "ui_control": "image_picker", "ui_order": i,
            })
            for i in range(1, 3)
        )
        fields.extend(
            field(f"video-{i}", "StimmaVideoParam", {
                "required": False, "ui_control": "video_picker", "ui_order": 10 + i,
            })
            for i in range(1, 3)
        )
        fields.extend(
            field(f"audio-{i}", "StimmaAudioParam", {
                "required": False, "ui_control": "audio_picker", "ui_order": 20 + i,
                "ui_label": "Standalone Reference Audio", "audio_role": "reference",
            })
            for i in range(1, 3)
        )
        workflow = DiscoveredWorkflow(
            file_path="reference.json",
            api_prompt={},
            tool_info={
                "slug": "h3-reference",
                "display_name": "H3 Reference",
                "task_types": ["reference-to-video"],
                "description": "",
            },
            field_nodes=fields,
        )

        descriptor = _build_single_tool(
            workflow, object_info=None, config=object(), provider=object()
        ).to_descriptor()
        schema = descriptor.parameter_schema

        self.assertEqual(schema["anyOf"], [
            {"required": ["input_images"], "properties": {"input_images": {"minItems": 1}}},
            {"required": ["input_videos"], "properties": {"input_videos": {"minItems": 1}}},
            {"required": ["input_audios"], "properties": {"input_audios": {"minItems": 1}}},
        ])
        self.assertNotIn("input_images", schema["required"])
        self.assertEqual(schema["properties"]["input_images"]["x-min-items"], 0)
        self.assertEqual(schema["properties"]["input_images"]["x-max-items"], 2)
        self.assertEqual(schema["properties"]["input_videos"]["x-min-items"], 0)
        self.assertEqual(schema["properties"]["input_videos"]["x-max-items"], 2)
        self.assertEqual(schema["properties"]["input_audios"]["x-min-items"], 0)
        self.assertEqual(schema["properties"]["input_audios"]["x-max-items"], 2)
        self.assertIn("immediately before", schema["properties"]["input_videos"]["description"])
        self.assertIn("Numbered after", schema["properties"]["input_audios"]["description"])


class TestSavedStimmaWidgetCompatibility(unittest.TestCase):
    def test_optional_tool_identity_values_are_not_dropped(self):
        workflow = {
            "nodes": [{
                "id": 1,
                "type": "StimmaToolInfo",
                # ComfyUI persists the required widgets here, but optional
                # widget values may exist only in widgets_values.
                "inputs": [
                    {"name": "slug", "widget": {"name": "slug"}, "link": None},
                    {"name": "display_name", "widget": {"name": "display_name"}, "link": None},
                    {"name": "task_types", "widget": {"name": "task_types"}, "link": None},
                    {"name": "badges", "widget": {"name": "badges"}, "link": None},
                    {"name": "description", "widget": {"name": "description"}, "link": None},
                ],
                "widgets_values": [
                    "ideogram4-t2i", "Ideogram 4.0", "text-to-image", "",
                    "Structured captions", "ideogram", "ideogram-v4",
                ],
            }],
            "links": [],
        }
        object_info = {
            "StimmaToolInfo": {"input": {
                "required": {
                    "slug": ("STRING", {}),
                    "display_name": ("STRING", {}),
                    "task_types": ("STRING", {}),
                    "badges": ("STRING", {}),
                    "description": ("STRING", {}),
                },
                "optional": {
                    "model_vendor": ("STRING", {}),
                    "model": ("STRING", {}),
                },
            }},
        }

        api_prompt = _convert_ui_to_api(workflow, object_info)
        self.assertEqual(api_prompt["1"]["inputs"]["model_vendor"], "ideogram")
        self.assertEqual(api_prompt["1"]["inputs"]["model"], "ideogram-v4")

    def test_new_widget_does_not_shift_legacy_image_param_values(self):
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "StimmaToolInfo",
                    "inputs": [
                        {"name": "slug", "widget": {"name": "slug"}, "link": None},
                        {"name": "display_name", "widget": {"name": "display_name"}, "link": None},
                        {"name": "task_types", "widget": {"name": "task_types"}, "link": None},
                    ],
                    "widgets_values": ["legacy-i2v", "Legacy I2V", "image-to-video"],
                },
                {
                    "id": 2,
                    "type": "StimmaImageParam",
                    "inputs": [
                        {"name": "image", "widget": {"name": "image"}, "link": None},
                        {"name": "controlnet_types", "widget": {"name": "controlnet_types"}, "link": None},
                        {"name": "ui_control", "widget": {"name": "ui_control"}, "link": None},
                        {"name": "ui_order", "widget": {"name": "ui_order"}, "link": None},
                        {"name": "allow_prep", "widget": {"name": "allow_prep"}, "link": None},
                    ],
                    "widgets_values": ["example.png", "", "video_frame_picker", 5, True],
                },
                {
                    "id": 3,
                    "type": "StimmaImageParam",
                    "inputs": [
                        {"name": "image", "widget": {"name": "image"}, "link": None},
                        {"name": "controlnet_types", "widget": {"name": "controlnet_types"}, "link": None},
                        {"name": "ui_control", "widget": {"name": "ui_control"}, "link": None},
                        {"name": "ui_order", "widget": {"name": "ui_order"}, "link": None},
                        {"name": "allow_prep", "widget": {"name": "allow_prep"}, "link": None},
                    ],
                    "widgets_values": ["example.png", "", "video_frame_picker", 6, True],
                },
            ],
            "links": [],
        }
        object_info = {
            "StimmaToolInfo": {"input": {"required": {
                "slug": ("STRING", {}),
                "display_name": ("STRING", {}),
                "task_types": ("STRING", {}),
            }}},
            "StimmaImageParam": {"input": {"required": {
                "image": (["example.png"],),
                "required": ("BOOLEAN", {"default": True}),
                "controlnet_types": ("STRING", {}),
                "ui_control": (["image_picker", "video_frame_picker"],),
                "ui_order": ("INT", {}),
                "allow_prep": ("BOOLEAN", {}),
            }}},
        }

        api_prompt = _convert_ui_to_api(workflow, object_info)
        self.assertEqual(api_prompt["2"]["inputs"]["ui_control"], "video_frame_picker")
        self.assertEqual(api_prompt["2"]["inputs"]["ui_order"], 5)
        self.assertNotIn("required", api_prompt["2"]["inputs"])

        extracted = _extract_stimma_nodes(api_prompt)
        image_fields = [
            node for node in extracted["field_nodes"]
            if node["class_type"] == "StimmaImageParam"
        ]
        self.assertEqual(
            [(node["inputs"]["required"], node["inputs"]["ui_order"]) for node in image_fields],
            [(True, 5), (False, 6)],
        )


class TestTopLevelRerouteConversion(unittest.TestCase):
    @staticmethod
    def _base_nodes():
        return [
            {
                "id": 1,
                "type": "StimmaToolInfo",
                "widgets_values": {
                    "slug": "reroute-test",
                    "display_name": "Reroute Test",
                    "task_types": "text-to-image",
                },
            },
            {
                "id": 2,
                "type": "StimmaPromptParam",
                "widgets_values": {
                    "name": "prompt",
                    "default_text": "hello",
                    "required": True,
                    "ui_order": 0,
                },
                "outputs": [{"name": "STRING", "type": "STRING", "links": []}],
            },
        ]

    @staticmethod
    def _object_info():
        return {
            "StimmaToolInfo": {},
            "StimmaPromptParam": {},
            "SomeConsumer": {
                "input": {"required": {"text": ("STRING", {})}}
            },
        }

    def test_array_link_reroute_is_removed_and_rewired(self):
        nodes = self._base_nodes() + [
            {
                "id": 3,
                "type": "Reroute",
                "inputs": [{"name": "", "type": "*", "link": 10}],
                "outputs": [{"name": "", "type": "*", "links": [11]}],
            },
            {
                "id": 4,
                "type": "SomeConsumer",
                "inputs": [{"name": "text", "type": "STRING", "link": 11}],
            },
        ]
        workflow = {
            "nodes": nodes,
            "links": [
                [10, 2, 0, 3, 0, "*"],
                [11, 3, 0, 4, 0, "STRING"],
            ],
        }

        prompt = _convert_ui_to_api(workflow, self._object_info())

        self.assertNotIn("3", prompt)
        self.assertEqual(prompt["4"]["inputs"]["text"], ["2", 0])
        self.assertEqual(_validate_workflow(prompt, self._object_info()), [])

    def test_object_link_reroute_chain_rewires_every_consumer(self):
        nodes = self._base_nodes() + [
            {
                "id": "r1",
                "type": "Reroute",
                "inputs": [{"name": "", "type": "*", "link": "a"}],
                "outputs": [{"name": "", "type": "*", "links": ["b"]}],
            },
            {
                "id": "r2",
                "type": "Reroute",
                "inputs": [{"name": "", "type": "*", "link": "b"}],
                "outputs": [{"name": "", "type": "*", "links": ["c", "d"]}],
            },
            {
                "id": 5,
                "type": "SomeConsumer",
                "inputs": [{"name": "text", "type": "STRING", "link": "c"}],
            },
            {
                "id": 6,
                "type": "SomeConsumer",
                "inputs": [{"name": "text", "type": "STRING", "link": "d"}],
            },
        ]

        def link(link_id, source, target):
            return {
                "id": link_id,
                "origin_id": source,
                "origin_slot": 0,
                "target_id": target,
                "target_slot": 0,
                "type": "STRING",
            }

        workflow = {
            "nodes": nodes,
            "links": [
                link("a", 2, "r1"),
                link("b", "r1", "r2"),
                link("c", "r2", 5),
                link("d", "r2", 6),
            ],
        }

        prompt = _convert_ui_to_api(workflow, self._object_info())

        self.assertNotIn("r1", prompt)
        self.assertNotIn("r2", prompt)
        self.assertEqual(prompt["5"]["inputs"]["text"], ["2", 0])
        self.assertEqual(prompt["6"]["inputs"]["text"], ["2", 0])

    def test_dangling_and_cyclic_reroutes_do_not_leave_invalid_inputs(self):
        nodes = self._base_nodes() + [
            {
                "id": 3,
                "type": "Reroute",
                "inputs": [{"name": "", "type": "*", "link": 10}],
                "outputs": [{"name": "", "type": "*", "links": [11, 12]}],
            },
            {
                "id": 4,
                "type": "Reroute",
                "inputs": [{"name": "", "type": "*", "link": 11}],
                "outputs": [{"name": "", "type": "*", "links": [10]}],
            },
            {
                "id": 5,
                "type": "SomeConsumer",
                "inputs": [{"name": "text", "type": "STRING", "link": 12}],
            },
            {
                "id": 6,
                "type": "SomeConsumer",
                "inputs": [{"name": "text", "type": "STRING", "link": 13}],
            },
            {
                "id": 7,
                "type": "Reroute",
                "inputs": [{"name": "", "type": "*", "link": None}],
                "outputs": [{"name": "", "type": "*", "links": [13]}],
            },
        ]
        workflow = {
            "nodes": nodes,
            "links": [
                [10, 4, 0, 3, 0, "*"],
                [11, 3, 0, 4, 0, "*"],
                [12, 3, 0, 5, 0, "STRING"],
                [13, 7, 0, 6, 0, "STRING"],
            ],
        }

        prompt = _convert_ui_to_api(workflow, self._object_info())

        self.assertNotIn("text", prompt["5"]["inputs"])
        self.assertNotIn("text", prompt["6"]["inputs"])


class TestReferenceToVideoWorkflow(unittest.TestCase):
    def test_links_and_h3_socket_order_match_native_model_presentation(self):
        workflow_path = Path(ROOT) / "workflows" / "Stimma-MiniMax-H3-R2V.json"
        workflow = json.loads(workflow_path.read_text())
        nodes = {node["id"]: node for node in workflow["nodes"]}

        for link_id, source, source_slot, target, target_slot, _type in workflow["links"]:
            self.assertIn(link_id, nodes[source]["outputs"][source_slot]["links"])
            self.assertEqual(nodes[target]["inputs"][target_slot]["link"], link_id)

        h3 = next(
            node for node in workflow["nodes"]
            if node["type"] == "MiniMaxH3ReferenceToVideo"
        )
        expected = ["clip", "vae", "audio_vae", "prompt", "width", "height", "length", "ref_image_size"]
        expected.extend(f"ref_images.ref_image_{index}" for index in range(9))
        for index in range(3):
            expected.extend((
                f"ref_videos.ref_video_{index}",
                f"ref_video_audios.ref_video_audio_{index}",
            ))
        expected.extend(f"ref_audios.ref_audio_{index}" for index in range(3))

        self.assertEqual([item["name"] for item in h3["inputs"]], expected)

        field_nodes = [
            node for node in workflow["nodes"]
            if node["type"] in {"StimmaImageParam", "StimmaVideoParam", "StimmaAudioParam"}
        ]
        self.assertEqual(sum(node["type"] == "StimmaImageParam" for node in field_nodes), 9)
        self.assertEqual(sum(node["type"] == "StimmaVideoParam" for node in field_nodes), 3)
        self.assertEqual(sum(node["type"] == "StimmaAudioParam" for node in field_nodes), 3)
        for node in field_nodes:
            # Slot 1 is the new required widget for typed visual fields and was
            # already the required widget for audio fields.
            self.assertIs(node["widgets_values"][1], False)
        for node in field_nodes:
            if node["type"] == "StimmaVideoParam":
                self.assertEqual(node["inputs"][-1]["name"], "target_fps")
                self.assertEqual(node["widgets_values"][-1], 24)

    def test_performance_controls_feed_reference_model_and_both_consumers(self):
        workflow_path = Path(ROOT) / "workflows" / "Stimma-MiniMax-H3-R2V.json"
        workflow = json.loads(workflow_path.read_text())
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}

        loader = next(
            node for node in workflow["nodes"]
            if node["type"] == "UNETLoader"
            and node["widgets_values"][0] == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        )
        precision_compare = next(
            node for node in workflow["nodes"]
            if node["type"] == "StringCompare"
        )
        model_switch = next(
            node for node in workflow["nodes"]
            if node["type"] == "ComfySwitchNode"
        )
        backend = next(
            node for node in workflow["nodes"]
            if node["type"] == "ModelAttentionBackend"
        )
        sol = next(
            node for node in workflow["nodes"]
            if node["type"] == "BlockSparseAttention"
        )
        lora = next(
            node for node in workflow["nodes"]
            if node["type"] == "StimmaLoraLoader"
        )

        parameter_nodes = {
            node["widgets_values"][0]: node
            for node in workflow["nodes"]
            if node["type"] in {
                "StimmaBoolParam",
                "StimmaDropdownParam",
                "StimmaFloatParam",
                "StimmaIntParam",
                "StimmaStringParam",
            }
        }
        expected_parameters = {
            "model_precision",
            "sol_sparsity",
            "sol_start",
            "sol_end",
        }
        self.assertTrue(expected_parameters.issubset(parameter_nodes))
        self.assertEqual(
            parameter_nodes["model_precision"]["widgets_values"][1],
            "INT8 ConvRot",
        )
        self.assertNotIn("spectrum", json.dumps(workflow).lower())

        precision_link = links[precision_compare["inputs"][0]["link"]]
        self.assertEqual(precision_link[1], parameter_nodes["model_precision"]["id"])
        self.assertEqual(precision_link[3], precision_compare["id"])

        lora_input_link = links[lora["inputs"][0]["link"]]
        self.assertEqual(lora_input_link[1], model_switch["id"])
        self.assertEqual(lora_input_link[3], lora["id"])
        self.assertEqual(lora["widgets_values"][0], "minimax-h3/**")

        int8_switch_link = links[model_switch["inputs"][0]["link"]]
        self.assertEqual(int8_switch_link[1], loader["id"])
        self.assertEqual(int8_switch_link[3], model_switch["id"])

        model_link = links[backend["inputs"][0]["link"]]
        self.assertEqual(model_link[1], lora["id"])
        self.assertEqual(model_link[3], backend["id"])

        sol_model_link = links[sol["inputs"][0]["link"]]
        self.assertEqual(sol_model_link[1], backend["id"])
        self.assertEqual(sol_model_link[3], sol["id"])

        consumer_types = {
            nodes[links[link_id][3]]["type"]
            for link_id in sol["outputs"][0]["links"]
        }
        self.assertEqual(consumer_types, {"BasicScheduler", "BasicGuider"})

        sol_inputs = {
            "selection.tau": "sol_sparsity",
            "start_percent": "sol_start",
            "end_percent": "sol_end",
        }
        for sol_input, parameter_name in sol_inputs.items():
            input_slot = next(
                item for item in sol["inputs"] if item["name"] == sol_input
            )
            link = links[input_slot["link"]]
            self.assertEqual(link[1], parameter_nodes[parameter_name]["id"])
            self.assertEqual(link[3], sol["id"])


class TestMiniMaxH3TurboWorkflows(unittest.TestCase):
    def test_turbo_preserves_generation_features_without_spectrum_controls(self):
        workflow_dir = Path(ROOT) / "workflows"
        paths = sorted(workflow_dir.glob("Stimma-MiniMax-H3-*-Turbo.json"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            with self.subTest(workflow=path.name):
                turbo = json.loads(path.read_text())
                base = json.loads(path.with_name(path.name.replace("-Turbo", "")).read_text())
                info = next(n for n in turbo["nodes"] if n["type"] == "StimmaToolInfo")
                self.assertTrue(info["widgets_values"][0].endswith("-turbo"))
                self.assertTrue(info["widgets_values"][1].endswith(" ⚡"))
                self.assertNotEqual(base["id"], turbo["id"])
                base_nodes = {n["id"]: n for n in base["nodes"]}
                turbo_nodes = {n["id"]: n for n in turbo["nodes"]}
                for nid, node in base_nodes.items():
                    if not node["type"].startswith("Stimma") or node["type"] == "StimmaToolInfo":
                        continue
                    expected = json.loads(json.dumps(node))
                    values = expected.get("widgets_values", [])
                    if values and (str(values[0]).startswith("spectrum") or values[0] == "Spectrum Tuning"):
                        continue
                    if values and values[0] == "Performance":
                        values[1] = "model_precision"
                    if values and values[0] == "steps":
                        values[1] = 8
                    elif values and values[0] == "sampler":
                        values[1] = "euler"
                    self.assertEqual(expected, turbo_nodes[nid])
                for workflow in (base, turbo):
                    loaders = [n for g in [workflow] + workflow.get("definitions", {}).get("subgraphs", [])
                               for n in g["nodes"] if n["type"] == "StimmaLoraLoader"]
                    self.assertEqual(len(loaders), 1)
                    self.assertEqual(loaders[0]["widgets_values"][0], "minimax-h3/**")
                    self.assertEqual(len(loaders[0]["widgets_values"]), 22)

    def test_lightx_sol_chain_and_model_registry(self):
        workflow_dir = Path(ROOT) / "workflows"
        registry = json.loads((Path(ROOT) / "models.json").read_text())["models"]
        self.assertFalse(any("ckpt850" in name for name in registry))
        for task in ("I2V", "T2V", "R2V"):
            with self.subTest(task=task):
                path = workflow_dir / f"Stimma-MiniMax-H3-{task}-Turbo.json"
                workflow = json.loads(path.read_text())
                graphs = [workflow] + workflow.get("definitions", {}).get("subgraphs", [])
                graph = next(g for g in graphs if any(n["type"] == "BlockSparseAttention" for n in g["nodes"]))
                nodes = {n["type"]: n for n in graph["nodes"]}
                self.assertNotIn("MiniMaxH3TurboLoRA", nodes)
                self.assertNotIn("MiniMaxH3TurboSampler", nodes)
                self.assertNotIn("spectrum", json.dumps(workflow).lower())
                self.assertEqual(nodes["MiniMaxH3SigmaShift"]["widgets_values"], [6.0, 3.0])
                self.assertEqual(nodes["KSamplerSelect"]["widgets_values"], ["euler"])
                self.assertEqual(nodes["BlockSparseAttention"]["widgets_values"],
                                 ["sol-attn", 1.0, .2, .9, "0, 49", 0, 256, "exact_kv_and_rows", False])
                adapter = nodes["LoraLoaderModelOnly"]["widgets_values"]
                family = "ref2v" if task == "R2V" else "fl2v"
                filename = f"minimax_h3_{family}_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
                self.assertEqual(adapter, [filename, 1.0])
                self.assertEqual(registry[filename]["source"]["repo"], "lightx2v/Minimax-h3-Turbo")
                self.assertIn(path.name, registry[filename]["used_by"])
                self.assertEqual(len(registry[filename]["sha256"]), 64)
                links = {l["id"] if isinstance(l, dict) else l[0]: l for l in graph["links"]}
                for left, right in zip(
                    ("StimmaLoraLoader", "LoraLoaderModelOnly", "MiniMaxH3SigmaShift", "ModelAttentionBackend"),
                    ("LoraLoaderModelOnly", "MiniMaxH3SigmaShift", "ModelAttentionBackend", "BlockSparseAttention"),
                ):
                    model_input = next(i for i in nodes[right]["inputs"] if i["name"] == "model")
                    link = links[model_input["link"]]
                    source = link["origin_id"] if isinstance(link, dict) else link[1]
                    self.assertEqual(source, nodes[left]["id"])


    def test_sol_controls_are_connected_to_all_h3_attention_graphs(self):
        for task in ("I2V", "T2V", "R2V"):
            for suffix in ("", "-Turbo"):
                with self.subTest(task=task, suffix=suffix):
                    w = json.loads((Path(ROOT) / "workflows" / f"Stimma-MiniMax-H3-{task}{suffix}.json").read_text())
                    self.assertNotIn("spectrum", json.dumps(w).lower())
                    self.assertFalse(any(
                        n["type"] == "MiniMaxH3MemoryEfficientSageAttentionPatch"
                        for g in [w] + w.get("definitions", {}).get("subgraphs", [])
                        for n in g["nodes"]
                    ))
                    params = {n["widgets_values"][0]: n for n in w["nodes"]
                              if n["type"] == "StimmaFloatParam"}
                    graph = next(g for g in [w] + w.get("definitions", {}).get("subgraphs", [])
                                 if any(n["type"] == "BlockSparseAttention" for n in g["nodes"]))
                    sol = next(n for n in graph["nodes"] if n["type"] == "BlockSparseAttention")
                    backend = next(n for n in graph["nodes"] if n["type"] == "ModelAttentionBackend")
                    links = {l["id"] if isinstance(l, dict) else l[0]: l for l in graph["links"]}
                    for name, target, default in (("sol_sparsity", "selection.tau", 1.0),
                                                   ("sol_start", "start_percent", .2),
                                                   ("sol_end", "end_percent", .9)):
                        self.assertEqual(params[name]["widgets_values"][1], default)
                        slot = next(i for i in sol["inputs"] if i["name"] == target)
                        link = links[slot["link"]]
                        if isinstance(link, dict):
                            self.assertEqual(link["origin_id"], graph["inputNode"]["id"])
                            self.assertEqual(graph["inputs"][link["origin_slot"]]["name"], name)
                            instance = next(n for n in w["nodes"] if n["type"] == graph["id"])
                            inp = next(i for i in instance["inputs"] if i["name"] == name)
                            root_link = next(l for l in w["links"] if l[0] == inp["link"])
                            self.assertEqual(root_link[1], params[name]["id"])
                        else:
                            self.assertEqual(link[1], params[name]["id"])
                    by_id = {n["id"]: n for n in graph["nodes"]}
                    sol_model_link = links[sol["inputs"][0]["link"]]
                    sol_model_source = sol_model_link["origin_id"] if isinstance(sol_model_link, dict) else sol_model_link[1]
                    self.assertEqual(sol_model_source, backend["id"])
                    consumers = set()
                    for lid in sol["outputs"][0]["links"]:
                        link = links[lid]
                        target = link["target_id"] if isinstance(link, dict) else link[3]
                        consumers.add(by_id[target]["type"])
                    self.assertEqual(consumers, {"BasicGuider", "BasicScheduler"})


class TestLTXLoraWorkflows(unittest.TestCase):
    def test_ltx_25_prompt_enhancement_defaults_to_passthrough_after_expansion(self):
        object_info = {
            "PrimitiveBoolean": {"input": {"required": {"value": ["BOOLEAN"]}}},
            "StimmaBoolParam": {"input": {"required": {
                "name": ["STRING"], "value": ["BOOLEAN"],
                "ui_order": ["INT"], "ui_description": ["STRING"],
            }}},
        }
        paths = sorted((Path(ROOT) / "workflows").glob("Stimma-LTX2.5-*.json"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                api_prompt = _convert_ui_to_api(workflow, object_info)
                enhancers = {
                    node_id for node_id, node in api_prompt.items()
                    if node["class_type"] == "TextGenerateLTX2Prompt"
                }
                self.assertEqual(len(enhancers), 1)

                def default_value(value):
                    if not isinstance(value, list):
                        return value
                    node = api_prompt[value[0]]
                    self.assertIn(node["class_type"], {"PrimitiveBoolean", "StimmaBoolParam"})
                    return default_value(node["inputs"]["value"])

                switches = [
                    node for node in api_prompt.values()
                    if node["class_type"] == "ComfySwitchNode"
                    and node["inputs"].get("on_true", [None])[0] in enhancers
                ]
                self.assertEqual(len(switches), 1)
                self.assertIs(default_value(switches[0]["inputs"]["switch"]), False)
                self.assertNotIn(switches[0]["inputs"]["on_false"][0], enhancers)

    def test_every_ltx_family_workflow_exposes_the_generation_lora_path(self):
        workflow_dir = Path(ROOT) / "workflows"
        paths = sorted(workflow_dir.glob("Stimma-LTX*.json"))
        paths.extend(sorted(workflow_dir.glob("Stimma-Sulphur*.json")))
        self.assertEqual(len(paths), 12)

        for path in paths:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                graphs = [workflow]
                graphs.extend(
                    (workflow.get("definitions") or {}).get("subgraphs") or []
                )
                matching = [
                    (graph, node)
                    for graph in graphs
                    for node in graph.get("nodes", [])
                    if node.get("type") == "StimmaLoraLoader"
                ]
                self.assertEqual(len(matching), 1)
                graph, loader = matching[0]
                expected_filter = (
                    "ltx-25/**" if "LTX2.5" in path.name else "ltx-23/**"
                )
                self.assertEqual(loader["widgets_values"][0], expected_filter)

                nodes = {node["id"]: node for node in graph["nodes"]}
                if graph["links"] and isinstance(graph["links"][0], list):
                    links = {link[0]: link for link in graph["links"]}

                    def endpoints(link_id):
                        link = links[link_id]
                        return link[1], link[2], link[3], link[4]
                else:
                    links = {link["id"]: link for link in graph["links"]}

                    def endpoints(link_id):
                        link = links[link_id]
                        return (
                            link["origin_id"], link["origin_slot"],
                            link["target_id"], link["target_slot"],
                        )

                for slot, link_type in ((0, "MODEL"), (1, "CLIP")):
                    input_link = links[loader["inputs"][slot]["link"]]
                    actual_type = (
                        input_link[5]
                        if isinstance(input_link, list)
                        else input_link["type"]
                    )
                    self.assertEqual(actual_type, link_type)
                    self.assertEqual(
                        endpoints(loader["inputs"][slot]["link"])[2],
                        loader["id"],
                    )

                model_targets = {
                    nodes[endpoints(link_id)[2]]["type"]
                    for link_id in loader["outputs"][0]["links"]
                }
                self.assertTrue(model_targets)
                self.assertTrue(model_targets <= {
                    "CFGGuider", "LTXVDualCFGGuider", "LoraLoaderModelOnly",
                })
                clip_targets = {
                    nodes[endpoints(link_id)[2]]["type"]
                    for link_id in loader["outputs"][1]["links"]
                }
                self.assertEqual(clip_targets, {"CLIPTextEncode"})


class TestAgentInstructions(unittest.TestCase):
    INSTRUCTION_NODES = {
        "nodes/tool_info.py": ["StimmaToolInfo"],
        "nodes/fields.py": [
            "StimmaImageParam", "StimmaImagesParam", "StimmaVideoParam",
            "StimmaVideosParam", "StimmaAudioParam",
        ],
    }

    @staticmethod
    def _widget_order(path, class_name):
        import ast
        tree = ast.parse((Path(ROOT) / path).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "INPUT_TYPES")
        ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
        order = []
        for section, body in zip(ret.value.keys, ret.value.values):
            if section.value in ("required", "optional"):
                order.extend(k.value for k in body.keys)
        return order

    def test_instructions_is_the_last_widget(self):
        # Widgets are positional in saved workflows; appending is the only safe change.
        for path, classes in self.INSTRUCTION_NODES.items():
            for class_name in classes:
                order = self._widget_order(path, class_name)
                self.assertEqual(order[-1], "instructions", class_name)
                self.assertEqual(order.count("instructions"), 1, class_name)

    def test_instructions_reach_the_descriptor(self):
        fields = [
            field("first", "StimmaImageParam", {
                "required": True, "ui_control": "video_frame_picker", "ui_order": 1,
                "instructions": "First frame.",
            }),
            field("last", "StimmaImageParam", {
                "required": False, "ui_control": "video_frame_picker", "ui_order": 2,
                "instructions": " Last frame. ",
            }),
        ]
        workflow = DiscoveredWorkflow(
            file_path="flf.json",
            api_prompt={},
            tool_info={
                "slug": "flf", "display_name": "FLF", "task_types": ["image-to-video"],
                "description": "Start and end frames.",
                "instructions": "Pass input_images=[first, last].",
            },
            field_nodes=fields,
        )
        descriptor = _build_single_tool(
            workflow, object_info=None, config=object(), provider=object()
        ).to_descriptor().to_dict()

        self.assertEqual(descriptor["metadata"]["description"], "Start and end frames.")
        self.assertEqual(descriptor["metadata"]["instructions"], "Pass input_images=[first, last].")
        self.assertEqual(
            descriptor["parameter_schema"]["properties"]["input_images"]["description"],
            "Item 1: First frame. Item 2: Last frame.",
        )

    def test_flf_workflows_describe_their_frames(self):
        for name in (
            "Stimma-MiniMax-H3-I2V.json", "Stimma-MiniMax-H3-I2V-Turbo.json",
            "Stimma-Wan22-FLF2V.json", "Stimma-Wan22-FLF2V-Lightning.json",
        ):
            nodes = json.loads((Path(ROOT) / "workflows" / name).read_text())["nodes"]
            tool = next(n for n in nodes if n["type"] == "StimmaToolInfo")
            self.assertEqual(len(tool["widgets_values"]), 8, name)
            self.assertIn("input_images=[first_frame, last_frame]", tool["widgets_values"][7], name)
            frames = sorted(
                (n for n in nodes if n["type"] == "StimmaImageParam"),
                key=lambda n: n["widgets_values"][4],
            )
            # Current layout: image, required, controlnet_types, ui_control,
            # ui_order, allow_prep, instructions.
            self.assertEqual([n["widgets_values"][1] for n in frames], [True, False], name)
            self.assertEqual({n["widgets_values"][3] for n in frames}, {"video_frame_picker"}, name)
            self.assertTrue(all(n["widgets_values"][6] for n in frames), name)
            for node in frames:
                saved = [i["name"] for i in node["inputs"] if i.get("widget")]
                self.assertEqual(saved, [
                    "image", "required", "controlnet_types", "ui_control",
                    "ui_order", "allow_prep", "instructions",
                ], name)


class TestKeyframes(unittest.TestCase):
    KF_INPUTS = {
        "name": "keyframes", "max_items": 6, "positions": "any", "fps": 24,
        "clip_min_frames": 5, "clip_step_frames": 17, "allow_sound": True,
        "timeline_param": "duration", "ui_order": 15, "instructions": "Pin things.",
    }

    def _schema(self, kf_inputs, task_types=("reference-to-video",)):
        workflow = DiscoveredWorkflow(
            file_path="kf.json", api_prompt={},
            tool_info={"slug": "kf", "display_name": "KF", "task_types": list(task_types), "description": ""},
            field_nodes=[
                field("img", "StimmaImageParam", {"required": False, "ui_control": "image_picker", "ui_order": 1}),
                field("kf", "StimmaKeyframesParam", kf_inputs),
            ],
        )
        return _build_single_tool(workflow, object_info=None, config=object(), provider=object()).to_descriptor().parameter_schema

    def test_schema_shape(self):
        schema = self._schema(self.KF_INPUTS)
        kf = schema["properties"]["keyframes"]
        self.assertEqual(kf["x-control"], "keyframes")
        self.assertEqual(kf["x-max-items"], 6)
        self.assertEqual(kf["x-positions"], "any")
        self.assertEqual(kf["x-timeline"], {"duration": "duration", "fps": 24})
        self.assertEqual(kf["x-clip-frames"], {"min": 5, "step": 17})
        self.assertEqual(kf["description"], "Pin things.")
        items = kf["items"]
        self.assertEqual(items["required"], ["media", "at"])
        self.assertEqual(items["properties"]["media"]["x-media"], ["image", "video"])
        self.assertEqual(set(items["properties"]), {"media", "at", "from", "length", "sound"})
        self.assertNotIn("keyframes", schema.get("required", []))
        self.assertIn({"required": ["keyframes"], "properties": {"keyframes": {"minItems": 1}}}, schema["anyOf"])

    def test_stills_only_omits_clip_fields(self):
        kf = self._schema(dict(self.KF_INPUTS, clip_min_frames=0), task_types=("image-to-video",))["properties"]["keyframes"]
        self.assertEqual(set(kf["items"]["properties"]), {"media", "at", "from"})
        self.assertNotIn("x-clip-frames", kf)

    def test_h3_reference_workflows_wire_keyframes_into_the_guider(self):
        for name in ("Stimma-MiniMax-H3-R2V.json", "Stimma-MiniMax-H3-R2V-Turbo.json"):
            wf = json.loads((Path(ROOT) / "workflows" / name).read_text())
            nodes = {n["id"]: n for n in wf["nodes"]}
            links = {l[0]: l for l in wf["links"]}
            for link_id, src, src_slot, dst, dst_slot, _ in wf["links"]:
                self.assertIn(link_id, nodes[src]["outputs"][src_slot]["links"], name)
                self.assertEqual(nodes[dst]["inputs"][dst_slot]["link"], link_id, name)
            adapter = next(n for n in wf["nodes"] if n["type"] == "StimmaApplyKeyframesH3")
            sources = {i["name"]: nodes[links[i["link"]][1]]["type"] for i in adapter["inputs"]}
            self.assertEqual(sources, {
                "positive": "MiniMaxH3ReferenceToVideo", "latent": "MiniMaxH3ReferenceToVideo",
                "vae": "VAELoader", "audio_vae": "VAELoader", "keyframes": "StimmaKeyframesParam",
                "duration": "StimmaFloatParam",
            }, name)
            out = links[adapter["outputs"][0]["links"][0]]
            self.assertEqual(nodes[out[3]]["type"], "BasicGuider", name)
            kf = next(n for n in wf["nodes"] if n["type"] == "StimmaKeyframesParam")
            saved = [i["name"] for i in kf["inputs"] if i.get("widget")]
            self.assertEqual(saved[-1], "instructions", name)
            self.assertEqual(len(kf["widgets_values"]), len(saved), name)
            self.assertEqual(wf["last_link_id"], max(links), name)
            self.assertEqual(wf["last_node_id"], max(nodes), name)


if __name__ == "__main__":
    unittest.main()
