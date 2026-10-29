"""Exercise MOSS discovery, layout prompt injection and model setup metadata."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stp_server.discovery import (
    DiscoveredWorkflow, _convert_ui_to_api, _extract_stimma_nodes,
    _resolve_stimma_links, _validate_workflow,
)
from stp_server.tool_builder import _build_single_tool
from stp_server.executor import _is_input_required, _strip_unprovided_input_chains


def object_info():
    # Ordered native widget contracts; model enums contain both released checkpoints.
    contracts = {
        'CLIPLoader': dict(clip_name=(['qwen3.5_9b_int8_convrot.safetensors'],), type=(['moss_image'],), device=(['default'],)),
        'VAELoader': dict(vae_name=(['flux2-vae_small.safetensors'],)),
        'UNETLoader': dict(unet_name=([f'moss_image_{v}_int8_convrot.safetensors' for v in ('sft', 'base')],), weight_dtype=(['default'],)),
        'EmptyFlux2LatentImage': dict(width=('INT',), height=('INT',), batch_size=('INT',)),
        'CLIPTextEncode': dict(clip=('CLIP',), text=('STRING',)),
        'KSampler': dict(model=('MODEL',), seed=('INT', {'control_after_generate': True}), steps=('INT',), cfg=('FLOAT',), sampler_name=(['euler'],), scheduler=(['sgm_uniform', 'simple'],), positive=('CONDITIONING',), negative=('CONDITIONING',), latent_image=('LATENT',), denoise=('FLOAT',)),
        'ModelSamplingAuraFlow': dict(model=('MODEL',), shift=('FLOAT',)),
        'SkipLayerGuidanceDiTSimple': dict(model=('MODEL',), double_layers=('STRING',), single_layers=('STRING',), start_percent=('FLOAT',), end_percent=('FLOAT',)),
        'VAEDecode': dict(samples=('LATENT',), vae=('VAE',)),
        'StimmaToolInfo': dict.fromkeys(['slug', 'display_name', 'task_types', 'badges', 'description', 'model_vendor', 'model', 'instructions'], ('STRING',)),
        'StimmaPromptParam': dict(name=('STRING',), default_text=('STRING',), required=('BOOLEAN',), ui_order=('INT',), ui_description=('STRING',)),
        'StimmaStringParam': dict(name=('STRING',), value=('STRING',), enum_values=('STRING',), ui_control=('STRING',), ui_order=('INT',), ui_description=('STRING',)),
        'StimmaResolutionParam': dict(width=('INT',), height=('INT',), min_size=('INT',), max_size=('INT',), step=('INT',), supported_resolutions=('STRING',), ui_order=('INT',), slider_max_pixels=('INT',)),
        'StimmaSeedParam': dict(name=('STRING',), value=('INT',), ui_order=('INT',)),
        'StimmaDropdownParam': dict(name=('STRING',), value=('STRING',), ui_order=('INT',), ui_description=('STRING',)),
        'StimmaImageOutput': dict(images=('IMAGE',), filename_prefix=('STRING',), _stimma_output_dir=('STRING',)),
        'StimmaLayoutGroup': dict(group_label=('STRING',), param_names=('STRING',), collapsed=('BOOLEAN',), ui_order=('INT',)),
        'StimmaLoraLoader': dict(model=('MODEL',), clip=('CLIP',), path_filter=('STRING',), ui_order=('INT',)),
        'LoraLoaderModelOnly': dict(model=('MODEL',), lora_name=([
            'moss-image-presets/moss_image_base_lodestone_8step_dmd.safetensors',
            'moss-image-presets/moss_image_sft_lodestone_8step_dmd.safetensors',
            'moss-image-presets/moss_image_edit_hq_armV5.safetensors',
        ],), strength_model=('FLOAT',)),
        'StimmaImageParam': dict(image=(['example.png'],), required=('BOOLEAN',), controlnet_types=('STRING',), ui_control=(['image_picker'],), ui_order=('INT',), allow_prep=('BOOLEAN',), instructions=('STRING',)),
        'ImageScaleToTotalPixels': dict(image=('IMAGE',), upscale_method=(['area'],), megapixels=('FLOAT',), resolution_steps=('INT',)),
        'GetImageSize': dict(image=('IMAGE',)),
        'StimmaImageSeed': dict(seed=('INT',), image=('IMAGE',)),
        'TextEncodeMOSSImageEdit': dict(clip=('CLIP',), prompt=('STRING',), edit_format=(['ai-toolkit'],), images=('COMFY_AUTOGROW_V3', {'template': {'prefix': 'image_', 'min': 1, 'max': 10}}), image_strength=('FLOAT',), vae=('VAE',)),
    }
    for typ, scalar in [('StimmaIntParam', 'INT'), ('StimmaFloatParam', 'FLOAT')]:
        contracts[typ] = dict(name=('STRING',), value=(scalar,), minimum=(scalar,), maximum=(scalar,), step=(scalar,), ui_control=('STRING',), ui_order=('INT',), ui_description=('STRING',))
    for i in range(1, 11):
        contracts['StimmaLoraLoader'][f'lora_{i}'] = (['None'],)
        contracts['StimmaLoraLoader'][f'strength_{i}'] = ('FLOAT',)
    return {k: {'input': {'required': v}} for k, v in contracts.items()}


class MossImageTests(unittest.TestCase):
    def test_turbo_and_edit_presets(self):
        info = object_info()
        manifest = json.loads((ROOT / 'models.json').read_text())['models']
        for variant in ('Base-Turbo', 'SFT-Turbo', 'Edit-SFT'):
            with self.subTest(variant=variant):
                filename = f'Stimma-MOSS-Image-{variant}.json'
                workflow = json.loads((ROOT / 'workflows' / filename).read_text())
                nodes = {n['id']: n for n in workflow['nodes']}
                for lid, src, slot, dst, inp, _ in workflow['links']:
                    self.assertIn(lid, nodes[src]['outputs'][slot]['links'])
                    self.assertEqual(nodes[dst]['inputs'][inp]['link'], lid)
                api = _convert_ui_to_api(workflow, info)
                self.assertEqual(_validate_workflow(api, info), [])
                discovered = DiscoveredWorkflow(file_path=filename, api_prompt=api, **_extract_stimma_nodes(api))
                descriptor = _build_single_tool(discovered, object_info=info, config=object(), provider=object()).to_descriptor()
                props = descriptor.parameter_schema['properties']
                edit = variant == 'Edit-SFT'
                self.assertEqual(props['steps']['default'], 30 if edit else 8)
                self.assertEqual(props['cfg']['default'], 3 if edit else 1)
                self.assertEqual(props['scheduler']['default'], 'simple')
                self.assertEqual(api['115']['inputs']['strength_model'], 1)
                self.assertEqual(api['107']['inputs']['model'], ['115', 0])
                self.assertEqual(api['107']['inputs']['lora_1'], 'None')
                for n in workflow['nodes']:
                    for m in n.get('properties', {}).get('models', []):
                        entry = manifest[m['name']]
                        self.assertIn(filename, entry['used_by'])
                        self.assertIn(entry['source']['revision'], m['url'])
                        self.assertEqual(len(entry['sha256']), 64)
                if edit:
                    self.assertEqual(props['input_images']['x-min-items'], 1)
                    self.assertEqual(props['input_images']['x-max-items'], 4)
                    self.assertNotIn('width', props)
                    self.assertEqual(api['19']['inputs']['seed'], ['144', 0])
                    self.assertEqual(api['28']['inputs']['width'], ['141', 0])
                    # One image must leave both conditioning branches and the
                    # sampler intact; only absent extra reference chains go.
                    _strip_unprovided_input_chains(api, ['121', '122', '123'], info)
                    for encoder in ('11', '12'):
                        self.assertEqual(api[encoder]['inputs']['edit_format'], 'ai-toolkit')
                        self.assertIn('images.image_1', api[encoder]['inputs'])
                        self.assertNotIn('images.image_2', api[encoder]['inputs'])
                    self.assertIn('19', api)
                    self.assertTrue(_is_input_required('TextEncodeMOSSImageEdit', 'images.image_1', info))
                    self.assertFalse(_is_input_required('TextEncodeMOSSImageEdit', 'images.image_2', info))
                    self.assertTrue(_is_input_required('TextEncodeMOSSImageEdit', 'images.image_11', info))
                else:
                    self.assertNotIn('input_images', props)
                    expected = 'base' if variant.startswith('Base') else 'sft'
                    self.assertIn(f'moss_image_{expected}_', api['115']['inputs']['lora_name'])
                    self.assertIn(f'moss_image_{expected}_', api['44']['inputs']['unet_name'])

    def test_discovery_and_execution_graph(self):
        info = object_info()
        layout = json.dumps({'summary': {'caption': 'A cream poster'}, 'items': [
            {'label': 'text', 'text': 'OPEN STORIES', 'caption': 'Large green title', 'bbox': [-400, -400, -200, 400]},
        ], 'bbox_coord_space': 'abs_centralized', 'bbox_format': 'yxyx'})
        for variant in ('SFT', 'Base'):
            with self.subTest(variant=variant):
                workflow = json.loads((ROOT / f'workflows/Stimma-MOSS-Image-{variant}.json').read_text())
                nodes = {n['id']: n for n in workflow['nodes']}
                for lid, src, slot, dst, inp, _ in workflow['links']:
                    self.assertIn(lid, nodes[src]['outputs'][slot]['links'])
                    self.assertEqual(nodes[dst]['inputs'][inp]['link'], lid)
                api = _convert_ui_to_api(workflow, info)
                self.assertEqual(_validate_workflow(api, info), [])
                extracted = _extract_stimma_nodes(api)
                discovered = DiscoveredWorkflow(file_path='moss.json', api_prompt=api, **extracted)
                descriptor = _build_single_tool(discovered, object_info=info, config=object(), provider=object()).to_descriptor()
                self.assertEqual(descriptor.model_vendor, 'openmoss')
                self.assertEqual(descriptor.model, f'moss-image-{variant.lower()}')
                props = descriptor.parameter_schema['properties']
                self.assertEqual(props['width']['x-step'], 16)
                self.assertEqual(props['steps']['default'], 50)
                self.assertEqual(props['cfg']['default'], 3.0)
                self.assertEqual(props['shift']['default'], 5.0)
                self.assertNotIn('input_images', props)
                # The executor injects user text before resolving Stimma links.
                api = copy.deepcopy(api)
                api['101']['inputs']['default_text'] = layout
                _resolve_stimma_links(api)
                self.assertEqual(api['11']['inputs']['text'], layout)
                # MOSS uses timesteps in [0, 1]. SD3's shift patch changes the
                # multiplier to 1000 and produces noise despite a successful job.
                self.assertEqual(api['112']['class_type'], 'ModelSamplingAuraFlow')
                self.assertEqual(api['112']['inputs']['shift'], 5.0)
                self.assertEqual(api['113']['inputs']['single_layers'], '22')
                self.assertEqual(api['19']['inputs']['model'], ['113', 0])
                self.assertEqual(api['19']['inputs']['steps'], 50)
                self.assertEqual(api['19']['inputs']['scheduler'], 'sgm_uniform')
                manifest = json.loads((ROOT / 'models.json').read_text())['models']
                for node in nodes.values():
                    for model in node.get('properties', {}).get('models', []):
                        entry = manifest[model['name']]
                        self.assertIn(f'Stimma-MOSS-Image-{variant}.json', entry['used_by'])
                        self.assertIn(entry['source']['revision'], model['url'])
                        self.assertEqual(len(entry['sha256']), 64)
                        self.assertGreater(entry['size'], 0)


if __name__ == '__main__':
    unittest.main()
