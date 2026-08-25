"""Tests for resolving bundled workflow model names against ComfyUI paths."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stp_server.discovery import (
    _default_optional_switch_nodes,
    _match_path_filter,
    _resolve_model_combo_value,
    _validate_workflow,
)
from stp_server.tool_builder import (
    _build_checkpoint_parameter,
    _build_lora_parameter,
    _match_lora_filter,
)


class TestModelResolution(unittest.TestCase):
    def test_forward_slash_filter_matches_windows_model_path(self):
        self.assertTrue(_match_lora_filter(
            r"anima\Studio_Ghibli.safetensors", "anima/**"
        ))
        self.assertTrue(_match_path_filter(
            r"sdxl\sd_xl_base_1.0.safetensors", "sdxl/**"
        ))

    def test_backslash_filter_matches_forward_slash_model_path(self):
        self.assertTrue(_match_lora_filter(
            "anima/Studio_Ghibli.safetensors", r"anima\**"
        ))

    def test_checkpoint_default_uses_comfyui_separator(self):
        node = {"inputs": {
            "name": "checkpoint",
            "ckpt_name": "sdxl/sd_xl_base_1.0.safetensors",
            "path_filter": "sdxl/**",
        }}
        object_info = {"CheckpointLoaderSimple": {"input": {"required": {
            "ckpt_name": ([r"sdxl\sd_xl_base_1.0.safetensors"],),
        }}}}

        parameter = _build_checkpoint_parameter(node, object_info)

        self.assertEqual(parameter.enum, [r"sdxl\sd_xl_base_1.0.safetensors"])
        self.assertEqual(parameter.default, r"sdxl\sd_xl_base_1.0.safetensors")

    def test_lora_default_uses_comfyui_separator(self):
        node = {"class_type": "StimmaLoraLoader", "inputs": {
            "path_filter": "anima/**",
            "lora_1": "anima/Studio_Ghibli.safetensors",
            "strength_1": 0.75,
        }}
        object_info = {"LoraLoader": {"input": {"required": {
            "lora_name": ([r"anima\Studio_Ghibli.safetensors"],),
        }}}}

        parameter = _build_lora_parameter(node, object_info)

        self.assertEqual(
            parameter.default,
            [{"name": r"anima\Studio_Ghibli.safetensors", "weight": 0.75}],
        )
        weight_schema = parameter.items["properties"]["weight"]
        self.assertEqual(weight_schema["minimum"], -10.0)
        self.assertEqual(weight_schema["maximum"], 10.0)

    def test_exact_match_is_preserved(self):
        resolved, ambiguous = _resolve_model_combo_value(
            "Anima/model.safetensors", ["Anima/model.safetensors"]
        )
        self.assertEqual(resolved, "Anima/model.safetensors")
        self.assertEqual(ambiguous, [])

    def test_unique_nested_basename_is_resolved(self):
        resolved, ambiguous = _resolve_model_combo_value(
            "model.safetensors",
            ["Flux/other.safetensors", "Anima/model.safetensors"],
        )
        self.assertEqual(resolved, "Anima/model.safetensors")
        self.assertEqual(ambiguous, [])

    def test_windows_separator_is_supported(self):
        resolved, ambiguous = _resolve_model_combo_value(
            "model.safetensors", [r"Anima\model.safetensors"]
        )
        self.assertEqual(resolved, r"Anima\model.safetensors")
        self.assertEqual(ambiguous, [])

    def test_duplicate_basename_is_ambiguous(self):
        resolved, ambiguous = _resolve_model_combo_value(
            "model.safetensors",
            ["Anima/model.safetensors", "Archive/model.safetensors"],
        )
        self.assertIsNone(resolved)
        self.assertEqual(len(ambiguous), 2)

    def test_validation_rewrites_prompt_to_comfyui_path(self):
        prompt = {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "anima-preview3-base.safetensors"},
            }
        }
        object_info = {
            "UNETLoader": {
                "input": {
                    "required": {
                        "unet_name": ([r"Anima\anima-preview3-base.safetensors"],)
                    }
                }
            }
        }

        warnings = _validate_workflow(prompt, object_info)

        self.assertEqual(warnings, [])
        self.assertEqual(
            prompt["1"]["inputs"]["unet_name"],
            r"Anima\anima-preview3-base.safetensors",
        )

    def test_anima_models_resolve_from_separate_nested_external_roots(self):
        prompt = {
            "clip": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "qwen_3_06b_base.safetensors"},
            },
            "vae": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "qwen_image_vae.safetensors"},
            },
            "unet": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "anima-preview3-base.safetensors"},
            },
        }
        object_info = {
            "CLIPLoader": {"input": {"required": {
                "clip_name": ([r"Anima\qwen_3_06b_base.safetensors"],),
            }}},
            "VAELoader": {"input": {"required": {
                "vae_name": ([r"Anima\qwen_image_vae.safetensors"],),
            }}},
            "UNETLoader": {"input": {"required": {
                "unet_name": ([r"Anima\anima-preview3-base.safetensors"],),
            }}},
        }

        warnings = _validate_workflow(prompt, object_info)

        self.assertEqual(warnings, [])
        self.assertEqual(
            prompt["clip"]["inputs"]["clip_name"],
            r"Anima\qwen_3_06b_base.safetensors",
        )
        self.assertEqual(
            prompt["vae"]["inputs"]["vae_name"],
            r"Anima\qwen_image_vae.safetensors",
        )
        self.assertEqual(
            prompt["unet"]["inputs"]["unet_name"],
            r"Anima\anima-preview3-base.safetensors",
        )

    def test_validation_blocks_ambiguous_basename(self):
        prompt = {
            "1": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "vae.safetensors"},
            }
        }
        object_info = {
            "VAELoader": {
                "input": {
                    "required": {
                        "vae_name": ([
                            "Anima/vae.safetensors",
                            "Qwen/vae.safetensors",
                        ],)
                    }
                }
            }
        }

        warnings = _validate_workflow(prompt, object_info)

        self.assertEqual(len(warnings), 1)
        self.assertIn("ambiguous matches", warnings[0])
        self.assertEqual(prompt["1"]["inputs"]["vae_name"], "vae.safetensors")

    def test_empty_fresh_install_model_combos_are_missing(self):
        prompt = {
            "unet": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "model.safetensors"},
            },
            "vae": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "vae.safetensors"},
            },
            "clip": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "clip.safetensors"},
            },
            "sampler": {
                "class_type": "KSampler",
                "inputs": {"sampler_name": "euler"},
            },
        }
        object_info = {
            "UNETLoader": {"input": {"required": {"unet_name": ([],)}}},
            "VAELoader": {"input": {"required": {"vae_name": ([],)}}},
            "CLIPLoader": {"input": {"required": {"clip_name": ([],)}}},
            "KSampler": {"input": {"required": {"sampler_name": ([],)}}},
        }
        issues = []

        warnings = _validate_workflow(prompt, object_info, issues)

        self.assertEqual(len(warnings), 3)
        self.assertEqual(
            {(issue["name"], issue["folder"]) for issue in issues},
            {
                ("model.safetensors", "diffusion_models"),
                ("vae.safetensors", "vae"),
                ("clip.safetensors", "text_encoders"),
            },
        )

    def test_inactive_default_model_variant_is_optional(self):
        prompt = {
            "int8": {"class_type": "UNETLoader", "inputs": {
                "unet_name": "model_int8.safetensors",
            }},
            "fp8": {"class_type": "UNETLoader", "inputs": {
                "unet_name": "model_fp8.safetensors",
            }},
            "precision": {"class_type": "StimmaStringParam", "inputs": {
                "value": "INT8 ConvRot",
            }},
            "compare": {"class_type": "StringCompare", "inputs": {
                "string_a": ["precision", 0], "string_b": "FP8",
                "mode": "Equal", "case_sensitive": True,
            }},
            "switch": {"class_type": "ComfySwitchNode", "inputs": {
                "switch": ["compare", 0], "on_false": ["int8", 0],
                "on_true": ["fp8", 0],
            }},
        }
        object_info = {
            "UNETLoader": {"input": {"required": {
                "unet_name": (["some_installed_model.safetensors"],),
            }}},
            "StringCompare": {"input": {}},
            "ComfySwitchNode": {"input": {}},
        }
        issues = []

        warnings = _validate_workflow(prompt, object_info, issues)

        self.assertEqual(_default_optional_switch_nodes(prompt), {"fp8"})
        self.assertEqual(len(warnings), 1)
        self.assertIn("model_int8.safetensors", warnings[0])
        by_name = {issue["name"]: issue for issue in issues}
        self.assertFalse(by_name["model_int8.safetensors"]["optional"])
        self.assertTrue(by_name["model_fp8.safetensors"]["optional"])


if __name__ == "__main__":
    unittest.main()
