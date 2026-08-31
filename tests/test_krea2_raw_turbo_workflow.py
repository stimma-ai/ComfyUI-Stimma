"""Structural regression tests for the bundled Krea 2 RAW-to-Turbo workflow."""

import fnmatch
import json
import unittest
from pathlib import Path

from stp_server.discovery import (
    ALL_STIMMA_TYPES,
    _convert_ui_to_api,
    _resolve_stimma_links,
    _validate_workflow,
)
from stp_server.manage.resolve import resolve_source


ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_NAME = "Stimma-Krea2-RAW-Turbo-T2I.json"
WORKFLOW_PATH = ROOT / "workflows" / WORKFLOW_NAME


class TestKrea2RawTurboWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = json.loads(WORKFLOW_PATH.read_text())
        cls.top_nodes = {node["id"]: node for node in cls.workflow["nodes"]}
        cls.top_links = {link[0]: link for link in cls.workflow["links"]}
        cls.subgraph = cls.workflow["definitions"]["subgraphs"][0]
        cls.inner_nodes = {node["id"]: node for node in cls.subgraph["nodes"]}
        cls.inner_links = {link["id"]: link for link in cls.subgraph["links"]}
        cls.manifest = json.loads((ROOT / "models.json").read_text())["models"]

    @classmethod
    def _object_info(cls):
        """Small current-ComfyUI schema sufficient to exercise subgraph expansion."""
        def widget(node_type):
            return ([node_type.lower()],)

        info = {}
        for node in cls.workflow["nodes"]:
            if node["type"] not in ALL_STIMMA_TYPES:
                continue
            required = {}
            for inp in node.get("inputs", []):
                required[inp["name"]] = (
                    (inp["type"],)
                    if inp["type"] in {"MODEL", "CLIP"}
                    else widget(inp["type"])
                )
            info[node["type"]] = {"input": {"required": required}}

        def core(**required):
            return {"input": {"required": required}}

        info.update({
            "UNETLoader": core(
                unet_name=(["krea2_raw_fp8_scaled.safetensors"],),
                weight_dtype=(["default"],),
            ),
            "CLIPLoader": core(
                clip_name=(["qwen3vl_4b_fp8_scaled.safetensors"],),
                type=(["krea2"],),
                device=(["default"],),
            ),
            "VAELoader": core(vae_name=(["qwen_image_vae.safetensors"],)),
            "LoraLoaderModelOnly": core(
                model=("MODEL",),
                lora_name=(["krea2-system/krea2_turbo_lora_rank_64_bf16.safetensors"],),
                strength_model=("FLOAT",),
            ),
            "EmptyLatentImage": core(width=("INT",), height=("INT",), batch_size=("INT",)),
            "CLIPTextEncode": core(clip=("CLIP",), text=("STRING",)),
            "VAEDecode": core(samples=("LATENT",), vae=("VAE",)),
            "RandomNoise": core(noise_seed=("INT", {"control_after_generate": True})),
            "BasicScheduler": core(
                model=("MODEL",), scheduler=(["simple"],), steps=("INT",), denoise=("FLOAT",),
            ),
            "SplitSigmas": core(sigmas=("SIGMAS",), step=("INT",)),
            "KSamplerSelect": core(sampler_name=(["euler"],)),
            "CFGGuider": core(
                model=("MODEL",), positive=("CONDITIONING",),
                negative=("CONDITIONING",), cfg=("FLOAT",),
            ),
            "BasicGuider": core(model=("MODEL",), conditioning=("CONDITIONING",)),
            "DisableNoise": core(),
            "SamplerCustomAdvanced": core(
                noise=("NOISE",), guider=("GUIDER",), sampler=("SAMPLER",),
                sigmas=("SIGMAS",), latent_image=("LATENT",),
            ),
        })
        return info

    def test_tool_identity_and_safe_defaults(self):
        self.assertEqual(self.top_nodes[200]["widgets_values"][0], "krea2-raw-turbo-t2i")

        steps = self.top_nodes[210]["widgets_values"]
        raw_steps = self.top_nodes[213]["widgets_values"]
        raw_guidance = self.top_nodes[214]["widgets_values"]
        self.assertEqual(steps[:5], ["steps", 8, 6, 16, 1])
        self.assertEqual(raw_steps[:5], ["raw_steps", 3, 1, 4, 1])
        self.assertEqual(raw_guidance[:5], ["raw_guidance", 2.5, 1.0, 5.0, 0.1])

    def test_one_schedule_is_split_between_two_samplers(self):
        scheduler = self.inner_nodes[31]
        split = self.inner_nodes[32]
        stage1 = self.inner_nodes[37]
        stage2 = self.inner_nodes[38]

        self.assertEqual(scheduler["type"], "BasicScheduler")
        self.assertEqual(split["type"], "SplitSigmas")
        self.assertEqual(stage1["type"], "SamplerCustomAdvanced")
        self.assertEqual(stage2["type"], "SamplerCustomAdvanced")

        self.assertEqual(self.inner_links[92]["origin_id"], 32)
        self.assertEqual(self.inner_links[92]["origin_slot"], 0)
        self.assertEqual(self.inner_links[92]["target_id"], 37)
        self.assertEqual(self.inner_links[97]["origin_id"], 32)
        self.assertEqual(self.inner_links[97]["origin_slot"], 1)
        self.assertEqual(self.inner_links[97]["target_id"], 38)

        self.assertEqual(self.inner_links[98]["origin_id"], 37)
        self.assertEqual(self.inner_links[98]["origin_slot"], 0)
        self.assertEqual(self.inner_links[98]["target_id"], 38)
        self.assertEqual(self.inner_links[94]["origin_id"], 36)
        self.assertEqual(self.inner_nodes[36]["type"], "DisableNoise")

    def test_raw_guidance_and_real_empty_negative_feed_only_stage_one(self):
        negative = self.inner_nodes[13]
        raw_guider = self.inner_nodes[34]
        turbo_guider = self.inner_nodes[35]

        self.assertEqual(negative["type"], "CLIPTextEncode")
        self.assertEqual(negative["widgets_values"], [""])
        self.assertEqual(raw_guider["type"], "CFGGuider")
        self.assertEqual(turbo_guider["type"], "BasicGuider")
        self.assertEqual(self.inner_links[82]["origin_id"], 13)
        self.assertEqual(self.inner_links[82]["target_id"], 34)
        self.assertEqual(self.inner_links[83]["origin_id"], -10)
        self.assertEqual(self.inner_links[83]["origin_slot"], 15)
        self.assertEqual(self.inner_links[83]["target_id"], 34)

    def test_user_loras_are_applied_before_fixed_turbo_adapter(self):
        style_loader = self.top_nodes[207]
        adapter = self.inner_nodes[15]
        adapter_name, adapter_strength = adapter["widgets_values"]

        self.assertEqual(style_loader["type"], "StimmaLoraLoader")
        self.assertEqual(style_loader["widgets_values"][0], "krea2/**")
        self.assertEqual(adapter["type"], "LoraLoaderModelOnly")
        self.assertEqual(adapter_strength, 1.0)
        self.assertEqual(
            adapter_name,
            "krea2-system/krea2_turbo_lora_rank_64_bf16.safetensors",
        )
        self.assertFalse(fnmatch.fnmatch(adapter_name, style_loader["widgets_values"][0]))
        self.assertEqual(self.inner_links[79]["origin_id"], -10)
        self.assertEqual(self.inner_links[79]["target_id"], 15)

    def test_required_models_have_download_sources_and_hashes(self):
        expected = {
            "krea2_raw_fp8_scaled.safetensors": (
                "diffusion_models",
                "diffusion_models/krea2_raw_fp8_scaled.safetensors",
            ),
            "krea2-system/krea2_turbo_lora_rank_64_bf16.safetensors": (
                "loras",
                "loras/krea2_turbo_lora_rank_64_bf16.safetensors",
            ),
            "qwen3vl_4b_fp8_scaled.safetensors": (
                "text_encoders",
                "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
            ),
            "qwen_image_vae.safetensors": (
                "vae",
                "split_files/vae/qwen_image_vae.safetensors",
            ),
        }
        for name, (directory, source_path) in expected.items():
            with self.subTest(name=name):
                entry = self.manifest[name]
                self.assertEqual(entry["directory"], directory)
                self.assertEqual(entry["source"]["type"], "huggingface")
                self.assertEqual(entry["source"]["path"], source_path)
                self.assertIn(WORKFLOW_NAME, entry["used_by"])
                self.assertGreater(entry["size"], 0)
                self.assertEqual(len(entry["sha256"]), 64)
                int(entry["sha256"], 16)

    def test_workflow_embeds_fresh_install_model_hints(self):
        raw_hint = self.inner_nodes[10]["properties"]["models"][0]
        adapter_hint = self.inner_nodes[15]["properties"]["models"][0]
        self.assertEqual(raw_hint["name"], "krea2_raw_fp8_scaled.safetensors")
        self.assertEqual(raw_hint["directory"], "diffusion_models")
        self.assertEqual(adapter_hint["name"], "krea2_turbo_lora_rank_64_bf16.safetensors")
        self.assertEqual(adapter_hint["directory"], "loras")

    def test_subgraph_expands_to_the_expected_api_graph(self):
        prompt = _convert_ui_to_api(self.workflow, self._object_info())
        self.assertIsNotNone(prompt)
        self.assertEqual(prompt["30:15"]["inputs"]["model"], ["207", 0])
        self.assertEqual(prompt["30:31"]["inputs"]["model"], ["207", 0])
        self.assertEqual(prompt["30:37"]["inputs"]["sigmas"], ["30:32", 0])
        self.assertEqual(prompt["30:38"]["inputs"]["sigmas"], ["30:32", 1])
        self.assertEqual(prompt["30:38"]["inputs"]["latent_image"], ["30:37", 0])
        self.assertEqual(prompt["30:38"]["inputs"]["noise"], ["30:36", 0])

        _resolve_stimma_links(prompt)
        self.assertEqual(prompt["30:31"]["inputs"]["steps"], 8)
        self.assertEqual(prompt["30:31"]["inputs"]["scheduler"], "simple")
        self.assertEqual(prompt["30:32"]["inputs"]["step"], 3)
        self.assertEqual(prompt["30:33"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(prompt["30:34"]["inputs"]["cfg"], 2.5)

        broken = []
        for node_id, node in prompt.items():
            for name, value in node.get("inputs", {}).items():
                if isinstance(value, list) and len(value) == 2 and str(value[0]) not in prompt:
                    broken.append(f"{node_id}.{name} -> {value[0]}")
        self.assertEqual(broken, [])

    def test_fresh_install_discovers_four_resolvable_model_downloads(self):
        object_info = self._object_info()
        for node in self.subgraph["nodes"]:
            object_info.setdefault(node["type"], {"input": {"required": {}}})
        object_info["UNETLoader"]["input"]["required"]["unet_name"] = ([],)
        object_info["CLIPLoader"]["input"]["required"]["clip_name"] = ([],)
        object_info["VAELoader"]["input"]["required"]["vae_name"] = ([],)
        object_info["LoraLoaderModelOnly"]["input"]["required"]["lora_name"] = ([],)

        prompt = _convert_ui_to_api(self.workflow, object_info)
        issues = []
        _validate_workflow(prompt, object_info, issues)
        missing = [
            issue for issue in issues
            if issue["kind"] == "missing_model" and not issue["optional"]
        ]
        self.assertEqual(
            {issue["name"] for issue in missing},
            {
                "krea2_raw_fp8_scaled.safetensors",
                "krea2-system/krea2_turbo_lora_rank_64_bf16.safetensors",
                "qwen3vl_4b_fp8_scaled.safetensors",
                "qwen_image_vae.safetensors",
            },
        )
        for issue in missing:
            source = resolve_source(issue["name"])
            self.assertIsNotNone(source, issue["name"])
            self.assertEqual(source["folder"], issue["folder"])
            self.assertTrue(source["url"].startswith("https://huggingface.co/"))


if __name__ == "__main__":
    unittest.main()
