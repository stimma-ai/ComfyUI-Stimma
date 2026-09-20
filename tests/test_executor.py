"""Tests for executor input injection and chain stripping."""

import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub stp_server.config to avoid ComfyUI imports
config_mod = types.ModuleType("stp_server.config")


class Config:
    def __init__(self):
        pass


config_mod.Config = Config
sys.modules["stp_server.config"] = config_mod

# Now we can import executor functions
from stp_server.executor import (
    _hydrate_missing_widget_defaults,
    _inject_fields,
    _is_input_required,
    _strip_unprovided_input_chains,
    _expand_stimma_images_reference_chains,
)


# Mock object_info for the Klein 9B i2i chain
MOCK_OBJECT_INFO = {
    "ImageScaleToTotalPixels": {
        "input": {
            "required": {
                "upscale_method": (
                    ["area", "nearest-exact", "bilinear", "bicubic", "lanczos"],
                ),
                "megapixels": ("FLOAT", {"default": 1.0}),
                "image": ("IMAGE",),
            },
            "optional": {},
        }
    },
    "VAEEncode": {
        "input": {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            },
            "optional": {},
        }
    },
    "ReferenceLatent": {
        "input": {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "latent": ("LATENT",),
            },
        }
    },
    "VAEDecode": {
        "input": {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
            },
            "optional": {},
        }
    },
    "BasicGuider": {
        "input": {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
            },
            "optional": {},
        }
    },
    "LoraLoaderModelOnly": {
        "input": {
            "required": {
                "model": ("MODEL",),
                "lora_name": (["a.safetensors"],),
                "strength_model": ("FLOAT", {"default": 1.0}),
            },
            "optional": {},
        }
    },
    "LTXVReferenceAudio": {
        "input": {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "reference_audio": ("AUDIO",),
                "audio_vae": ("VAE",),
            },
            "optional": {},
        }
    },
    "CFGGuider": {
        "input": {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 1.0}),
            },
            "optional": {},
        }
    },
}


class TestIsInputRequired(unittest.TestCase):
    def test_zero_minimum_autogrow_references_are_optional(self):
        info = {"TextEncodeQwenImage21": {"input": {"required": {
            "images": ["COMFY_AUTOGROW_V3", {"template": {
                "names": ["image_1", "image_2"], "min": 0,
            }}],
        }}}}
        self.assertFalse(_is_input_required("TextEncodeQwenImage21", "images.image_2", info))
        self.assertTrue(_is_input_required("TextEncodeQwenImage21", "images.unknown", info))
        info["TextEncodeQwenImage21"]["input"]["required"]["images"][1]["template"]["min"] = 1
        self.assertTrue(_is_input_required("TextEncodeQwenImage21", "images.image_1", info))

    def test_required_input(self):
        self.assertTrue(
            _is_input_required("ImageScaleToTotalPixels", "image", MOCK_OBJECT_INFO)
        )

    def test_optional_input(self):
        self.assertFalse(
            _is_input_required("ReferenceLatent", "latent", MOCK_OBJECT_INFO)
        )

    def test_unknown_node_defaults_to_required(self):
        self.assertTrue(
            _is_input_required("UnknownNode", "some_input", MOCK_OBJECT_INFO)
        )

    def test_unknown_input_defaults_to_required(self):
        self.assertTrue(
            _is_input_required("ReferenceLatent", "nonexistent", MOCK_OBJECT_INFO)
        )

    def test_h3_expanded_reference_sockets_are_optional(self):
        for name in (
            "ref_images.ref_image_0",
            "ref_videos.ref_video_1",
            "ref_video_audios.ref_video_audio_1",
            "ref_audios.ref_audio_2",
        ):
            self.assertFalse(
                _is_input_required("MiniMaxH3ReferenceToVideo", name, {})
            )


class TestHydrateMissingWidgetDefaults(unittest.TestCase):
    def test_legacy_video_param_gets_new_required_widget_defaults(self):
        prompt = {
            "26": {
                "class_type": "StimmaVideoParam",
                "inputs": {"video": "uploaded.mp4", "ui_control": "videoPicker", "ui_order": 1},
            }
        }
        object_info = {
            "StimmaVideoParam": {"input": {"required": {
                "video": (["example.mp4"],),
                "required": ("BOOLEAN", {"default": True}),
                "ui_control": (["video_picker"],),
                "ui_order": ("INT", {"default": 0}),
                "target_fps": ("INT", {"default": 0}),
                "frames": ("IMAGE",),
            }}}
        }

        _hydrate_missing_widget_defaults(prompt, object_info)

        self.assertEqual(prompt["26"]["inputs"]["video"], "uploaded.mp4")
        self.assertIs(prompt["26"]["inputs"]["required"], True)
        self.assertEqual(prompt["26"]["inputs"]["target_fps"], 0)
        self.assertNotIn("frames", prompt["26"]["inputs"])


class TestStripUnprovidedInputChains(unittest.TestCase):
    def _make_klein_prompt(self):
        """Create a mock prompt matching the Klein 9B i2i chain."""
        return {
            "21": {
                "class_type": "StimmaImageParam",
                "inputs": {
                    "image": "example.png",
                },
            },
            "41": {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {
                    "upscale_method": "area",
                    "megapixels": 1.0,
                    "image": ["21", 0],
                },
            },
            "43": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["41", 0], "vae": ["3", 0]},
            },
            "12": {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["11", 0], "latent": ["43", 0]},
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": "flux2-vae.safetensors"},
            },
            "11": {
                "class_type": "FluxGuidance",
                "inputs": {"guidance": 3.5, "conditioning": ["10", 0]},
            },
            "13": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["4", 0], "conditioning": ["12", 0]},
            },
        }

    def test_strip_i2i_chain_keeps_reference_latent(self):
        """When optional image input is unprovided, cascade removes the
        image processing chain but ReferenceLatent survives with latent removed."""
        prompt = self._make_klein_prompt()
        _strip_unprovided_input_chains(prompt, ["21"], MOCK_OBJECT_INFO)

        # Removed: StimmaImageParam, ImageScaleToTotalPixels, VAEEncode
        self.assertNotIn("21", prompt)
        self.assertNotIn("41", prompt)
        self.assertNotIn("43", prompt)

        # Survived: ReferenceLatent, FluxGuidance, VAELoader, BasicGuider
        self.assertIn("12", prompt)
        self.assertIn("11", prompt)
        self.assertIn("3", prompt)
        self.assertIn("13", prompt)

        # ReferenceLatent has conditioning but not latent
        self.assertEqual(prompt["12"]["inputs"]["conditioning"], ["11", 0])
        self.assertNotIn("latent", prompt["12"]["inputs"])

    def test_no_strip_when_no_unprovided(self):
        """Nothing changes when unprovided list is empty."""
        prompt = self._make_klein_prompt()
        original_keys = set(prompt.keys())
        _strip_unprovided_input_chains(prompt, [], MOCK_OBJECT_INFO)
        self.assertEqual(set(prompt.keys()), original_keys)

    def test_cascade_through_multiple_required(self):
        """Cascade continues through multiple required inputs."""
        prompt = {
            "a": {"class_type": "StimmaImageParam", "inputs": {}},
            "b": {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {"image": ["a", 0]},
            },
            "c": {"class_type": "VAEEncode", "inputs": {"pixels": ["b", 0], "vae": ["v", 0]}},
            "v": {"class_type": "VAELoader", "inputs": {}},
        }
        _strip_unprovided_input_chains(prompt, ["a"], MOCK_OBJECT_INFO)
        self.assertNotIn("a", prompt)
        self.assertNotIn("b", prompt)
        self.assertNotIn("c", prompt)
        self.assertIn("v", prompt)  # VAELoader has no refs to removed nodes

    def test_optional_ref_just_removed(self):
        """Optional inputs referencing removed nodes are deleted, not cascaded."""
        prompt = {
            "src": {"class_type": "StimmaImageParam", "inputs": {}},
            "ref": {
                "class_type": "ReferenceLatent",
                "inputs": {
                    "conditioning": ["other", 0],
                    "latent": ["src", 0],
                },
            },
        }
        _strip_unprovided_input_chains(prompt, ["src"], MOCK_OBJECT_INFO)
        self.assertNotIn("src", prompt)
        self.assertIn("ref", prompt)
        self.assertNotIn("latent", prompt["ref"]["inputs"])
        self.assertEqual(prompt["ref"]["inputs"]["conditioning"], ["other", 0])


class TestReferenceAudioBypass(unittest.TestCase):
    """No voice reference → the guide node AND its ID-LoRA leave the graph."""

    def _make_prompt(self):
        return {
            "aud": {"class_type": "StimmaAudioParam", "inputs": {}},
            "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {}},
            "distilled": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["ckpt", 0], "lora_name": "distilled", "strength_model": 0.5},
            },
            "idlora": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["distilled", 0], "lora_name": "id-lora", "strength_model": 1.0},
            },
            "ref": {
                "class_type": "LTXVReferenceAudio",
                "inputs": {
                    "model": ["idlora", 0],
                    "positive": ["pos", 0],
                    "negative": ["neg", 0],
                    "reference_audio": ["aud", 0],
                    "audio_vae": ["vae", 0],
                },
            },
            "pos": {"class_type": "CLIPTextEncode", "inputs": {}},
            "neg": {"class_type": "CLIPTextEncode", "inputs": {}},
            "vae": {"class_type": "VAELoader", "inputs": {}},
            "guider": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["ref", 0], "positive": ["ref", 1],
                    "negative": ["ref", 2], "cfg": 1.0,
                },
            },
            "guider2": {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": ["distilled", 0], "positive": ["pos", 0],
                    "negative": ["neg", 0], "cfg": 1.0,
                },
            },
        }

    def test_bypass_drops_reference_node_and_its_lora(self):
        prompt = self._make_prompt()
        _strip_unprovided_input_chains(prompt, ["aud"], MOCK_OBJECT_INFO)

        self.assertNotIn("aud", prompt)
        self.assertNotIn("ref", prompt)
        self.assertNotIn("idlora", prompt)  # the guide's LoRA goes with it
        self.assertIn("distilled", prompt)  # the shared one stays

        # The guider now reads the un-ID-LoRA'd model and the raw conditioning.
        self.assertEqual(prompt["guider"]["inputs"]["model"], ["distilled", 0])
        self.assertEqual(prompt["guider"]["inputs"]["positive"], ["pos", 0])
        self.assertEqual(prompt["guider"]["inputs"]["negative"], ["neg", 0])
        # The refine pass was never touched.
        self.assertEqual(prompt["guider2"]["inputs"]["model"], ["distilled", 0])

    def test_shared_lora_survives_the_chase(self):
        """A LoRA the rest of the graph also uses is rewired past, not deleted."""
        prompt = self._make_prompt()
        prompt["guider2"]["inputs"]["model"] = ["idlora", 0]

        _strip_unprovided_input_chains(prompt, ["aud"], MOCK_OBJECT_INFO)

        self.assertNotIn("ref", prompt)
        self.assertIn("idlora", prompt)
        self.assertEqual(prompt["guider"]["inputs"]["model"], ["distilled", 0])
        self.assertEqual(prompt["guider2"]["inputs"]["model"], ["idlora", 0])

    def test_provided_audio_leaves_the_chain_intact(self):
        prompt = self._make_prompt()
        _strip_unprovided_input_chains(prompt, [], MOCK_OBJECT_INFO)
        self.assertIn("ref", prompt)
        self.assertIn("idlora", prompt)
        self.assertEqual(prompt["guider"]["inputs"]["model"], ["ref", 0])


class TestInjectFieldsListHandling(unittest.TestCase):
    """Test that _inject_fields correctly handles list values for single-image fields."""

    def _make_workflow_with_image_input(self):
        """Create a minimal mock DiscoveredWorkflow."""
        wf = MagicMock()
        wf.field_nodes = [
            {
                "node_id": "21",
                "class_type": "StimmaImageParam",
                "name": "input_images",
                "inputs": {},
            }
        ]
        return wf

    def test_empty_list_raises_for_required_single_image(self):
        """Single image inputs are required and must fail when not provided."""
        import asyncio

        wf = self._make_workflow_with_image_input()
        prompt = {
            "21": {
                "class_type": "StimmaImageParam",
                "inputs": {"image": "example.png"},
            }
        }
        input_data = {"input_images": []}
        context = MagicMock()
        comfy = MagicMock()

        with self.assertRaises(RuntimeError):
            asyncio.get_event_loop().run_until_complete(
                _inject_fields(prompt, wf, input_data, context, comfy)
            )

    def test_none_value_raises_for_required_single_image(self):
        """Single image inputs are required and must fail when missing."""
        import asyncio

        wf = self._make_workflow_with_image_input()
        prompt = {
            "21": {
                "class_type": "StimmaImageParam",
                "inputs": {"image": "example.png"},
            }
        }
        input_data = {}
        context = MagicMock()
        comfy = MagicMock()

        with self.assertRaises(RuntimeError):
            asyncio.get_event_loop().run_until_complete(
                _inject_fields(prompt, wf, input_data, context, comfy)
            )

    def test_reference_to_video_requires_at_least_one_typed_reference(self):
        import asyncio

        wf = MagicMock()
        wf.tool_info = {"task_types": ["reference-to-video"]}
        wf.field_nodes = []

        with self.assertRaisesRegex(RuntimeError, "at least one reference"):
            asyncio.get_event_loop().run_until_complete(
                _inject_fields({}, wf, {}, MagicMock(), MagicMock())
            )

    def test_optional_image_and_video_slots_can_be_omitted(self):
        import asyncio

        wf = MagicMock()
        wf.tool_info = {"task_types": ["reference-to-video"]}
        wf.field_nodes = [
            {
                "node_id": "image",
                "class_type": "StimmaImageParam",
                "inputs": {"required": False},
            },
            {
                "node_id": "video",
                "class_type": "StimmaVideoParam",
                "inputs": {"required": False},
            },
        ]
        prompt = {
            "image": {"class_type": "StimmaImageParam", "inputs": {}},
            "video": {"class_type": "StimmaVideoParam", "inputs": {}},
        }

        omitted = asyncio.get_event_loop().run_until_complete(
            _inject_fields(
                prompt,
                wf,
                {"input_audios": ["reference.wav"]},
                MagicMock(),
                MagicMock(),
            )
        )

        self.assertEqual(omitted, ["image", "video"])


class TestReferenceChainExpansion(unittest.TestCase):
    def test_expand_stimma_images_reference_chain(self):
        prompt = {
            "101": {
                "class_type": "StimmaImagesParam",
                "inputs": {"image": "first.png", "min_images": 1, "max_images": 10},
            },
            "80": {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {"image": ["101", 0], "upscale_method": "nearest-exact", "megapixels": 1.0},
            },
            "78": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["80", 0], "vae": ["72", 0]},
            },
            "77": {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["74", 0], "latent": ["78", 0]},
            },
            "76": {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["82", 0], "latent": ["78", 0]},
            },
            "63": {
                "class_type": "CFGGuider",
                "inputs": {"positive": ["77", 0], "negative": ["76", 0], "model": ["7", 0], "cfg": 1.0},
            },
            "72": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
            "74": {"class_type": "CLIPTextEncode", "inputs": {"text": "x", "clip": ["3", 0]}},
            "82": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["74", 0]}},
        }

        expanded = _expand_stimma_images_reference_chains(
            prompt,
            "101",
            ["first.png", "second.png", "third.png"],
        )
        self.assertTrue(expanded)

        # Two additional refs should create two extra ReferenceLatent nodes per branch.
        ref_nodes = [
            nid for nid, nd in prompt.items()
            if nd.get("class_type") == "ReferenceLatent"
        ]
        self.assertEqual(len(ref_nodes), 6)

        # CFG inputs should be rewired away from original ref nodes to the expanded tail.
        self.assertNotEqual(prompt["63"]["inputs"]["positive"][0], "77")
        self.assertNotEqual(prompt["63"]["inputs"]["negative"][0], "76")

        # Cloned source nodes should exist with second/third image filenames.
        source_images = [
            nd["inputs"].get("image")
            for nd in prompt.values()
            if nd.get("class_type") == "StimmaImagesParam"
        ]
        self.assertIn("second.png", source_images)
        self.assertIn("third.png", source_images)


if __name__ == "__main__":
    unittest.main()
