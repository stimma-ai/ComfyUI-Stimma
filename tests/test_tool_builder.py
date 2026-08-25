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
        sage = next(
            node for node in workflow["nodes"]
            if node["type"] == "MiniMaxH3MemoryEfficientSageAttentionPatch"
        )
        spectrum = next(
            node for node in workflow["nodes"]
            if node["type"] == "SpectrumApplyMiniMaxH3"
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
            "spectrum",
            "spectrum_blend",
            "spectrum_degree",
            "spectrum_ridge",
            "spectrum_window",
            "spectrum_flex_window",
            "spectrum_warmup_steps",
            "spectrum_tail_steps",
            "spectrum_max_history",
            "spectrum_history_storage",
            "spectrum_debug",
        }
        self.assertTrue(expected_parameters.issubset(parameter_nodes))
        self.assertEqual(
            parameter_nodes["model_precision"]["widgets_values"][1],
            "INT8 ConvRot",
        )
        self.assertIs(parameter_nodes["spectrum"]["widgets_values"][1], False)

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

        model_link = links[sage["inputs"][0]["link"]]
        self.assertEqual(model_link[1], lora["id"])
        self.assertEqual(model_link[3], sage["id"])

        spectrum_model_link = links[spectrum["inputs"][0]["link"]]
        self.assertEqual(spectrum_model_link[1], sage["id"])
        self.assertEqual(spectrum_model_link[3], spectrum["id"])

        consumer_types = {
            nodes[links[link_id][3]]["type"]
            for link_id in spectrum["outputs"][0]["links"]
        }
        self.assertEqual(consumer_types, {"BasicScheduler", "BasicGuider"})

        spectrum_inputs = {
            "enabled": "spectrum",
            "blend_weight": "spectrum_blend",
            "degree": "spectrum_degree",
            "ridge_lambda": "spectrum_ridge",
            "window_size": "spectrum_window",
            "flex_window": "spectrum_flex_window",
            "warmup_steps": "spectrum_warmup_steps",
            "tail_actual_steps": "spectrum_tail_steps",
            "max_history": "spectrum_max_history",
            "history_storage": "spectrum_history_storage",
            "debug": "spectrum_debug",
        }
        for spectrum_input, parameter_name in spectrum_inputs.items():
            input_slot = next(
                item for item in spectrum["inputs"] if item["name"] == spectrum_input
            )
            link = links[input_slot["link"]]
            self.assertEqual(link[1], parameter_nodes[parameter_name]["id"])
            self.assertEqual(link[3], spectrum["id"])


class TestMiniMaxH3TurboWorkflows(unittest.TestCase):
    def test_turbo_variants_use_the_fixed_four_step_path(self):
        workflow_dir = Path(ROOT) / "workflows"
        paths = sorted(workflow_dir.glob("Stimma-MiniMax-H3-*-Turbo.json"))
        self.assertEqual(len(paths), 3)

        for path in paths:
            with self.subTest(workflow=path.name):
                workflow = json.loads(path.read_text())
                tool_info = next(
                    node for node in workflow["nodes"]
                    if node["type"] == "StimmaToolInfo"
                )
                self.assertTrue(tool_info["widgets_values"][0].endswith("-turbo"))
                self.assertTrue(tool_info["widgets_values"][1].endswith(" ⚡"))
                self.assertNotIn("Turbo", tool_info["widgets_values"][1])

                parameters = {
                    node["widgets_values"][0]: node
                    for node in workflow["nodes"]
                    if node["type"] in {
                        "StimmaBoolParam",
                        "StimmaDropdownParam",
                        "StimmaFloatParam",
                        "StimmaIntParam",
                    }
                }
                self.assertNotIn("sampler", parameters)
                self.assertEqual(parameters["steps"]["widgets_values"][1:3], [4, 4])

                graphs = [workflow]
                graphs.extend(workflow.get("definitions", {}).get("subgraphs", []))
                graph = next(
                    item for item in graphs
                    if any(node["type"] == "MiniMaxH3TurboLoRA" for node in item["nodes"])
                )
                nodes = {node["id"]: node for node in graph["nodes"]}
                generic_lora = next(
                    node for node in graph["nodes"]
                    if node["type"] == "StimmaLoraLoader"
                )
                turbo_lora = next(
                    node for node in graph["nodes"]
                    if node["type"] == "MiniMaxH3TurboLoRA"
                )
                sage = next(
                    node for node in graph["nodes"]
                    if node["type"] == "MiniMaxH3MemoryEfficientSageAttentionPatch"
                )
                self.assertEqual(
                    turbo_lora["widgets_values"],
                    ["minimax_h3_turbo_4step_ema_ckpt850.safetensors", 1, False],
                )
                self.assertTrue(any(
                    node["type"] == "MiniMaxH3TurboSampler"
                    for node in graph["nodes"]
                ))

                if graph["links"] and isinstance(graph["links"][0], list):
                    links = {link[0]: link for link in graph["links"]}
                    turbo_input = links[turbo_lora["inputs"][0]["link"]]
                    sage_input = links[sage["inputs"][0]["link"]]
                    self.assertEqual(nodes[turbo_input[1]]["id"], generic_lora["id"])
                    self.assertEqual(turbo_input[3], turbo_lora["id"])
                    self.assertEqual(sage_input[1], turbo_lora["id"])
                    self.assertEqual(sage_input[3], sage["id"])
                else:
                    links = {link["id"]: link for link in graph["links"]}
                    turbo_input = links[turbo_lora["inputs"][0]["link"]]
                    sage_input = links[sage["inputs"][0]["link"]]
                    self.assertEqual(turbo_input["origin_id"], generic_lora["id"])
                    self.assertEqual(turbo_input["target_id"], turbo_lora["id"])
                    self.assertEqual(sage_input["origin_id"], turbo_lora["id"])
                    self.assertEqual(sage_input["target_id"], sage["id"])


class TestLTXLoraWorkflows(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
