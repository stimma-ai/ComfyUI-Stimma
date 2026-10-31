"""Masked edit tools must remain discoverable and preserve the crop/mask path."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_moss_image import object_info as moss_object_info
from stp_server.discovery import (
    DiscoveredWorkflow, _convert_ui_to_api, _extract_stimma_nodes,
    _resolve_stimma_links, _validate_workflow,
)
from stp_server.executor import _strip_unprovided_input_chains
from stp_server.tool_builder import _build_single_tool

NAMES = (
    'Stimma-Qwen-Image-2.1-Inpaint.json',
    'Stimma-Qwen-Image-2.1-Turbo-Inpaint.json',
    'Stimma-MOSS-Image-Edit-SFT-Inpaint.json',
)


def object_info():
    info = moss_object_info()
    contracts = {
        'TextEncodeQwenImage21': dict(
            clip=('CLIP',), prompt=('STRING',), negative_prompt=('STRING',),
            resolution=('INT',), images=('COMFY_AUTOGROW_V3', {'template': {
                'names': [f'image_{i}' for i in range(1, 17)], 'min': 0,
            }}), vae=('VAE',)),
        'StimmaMaskParam': dict(name=('STRING',), image=(['example.png'],), source_image_field=('STRING',), ui_order=('INT',)),
        'SplitImageWithAlpha': dict(image=('IMAGE',)),
        'VAEEncode': dict(pixels=('IMAGE',), vae=('VAE',)),
        'SetLatentNoiseMask': dict(samples=('LATENT',), mask=('MASK',)),
        'ResizeMask': dict(mask=('MASK',), width=('INT',), height=('INT',), keep_proportions=('BOOLEAN',), upscale_method=(['nearest-exact'],), crop=(['disabled'],)),
        'DrawMaskOnImage': dict(image=('IMAGE',), mask=('MASK',), color=('STRING',), device=(['cpu'],)),
        'StringConcatenate': dict(string_a=('STRING',), string_b=('STRING',), delimiter=('STRING',)),
        'InpaintStitchImproved': dict(stitcher=('STITCHER',), inpainted_image=('IMAGE',)),
    }
    # Native CropAndStitch widget order: linked widgets still occupy positions.
    crop_types = [
        ('image', 'IMAGE'), ('downscale_algorithm', ['bilinear']), ('upscale_algorithm', ['bicubic']),
        ('preresize', 'BOOLEAN'), ('preresize_mode', ['ensure minimum resolution']),
        ('preresize_min_width', 'INT'), ('preresize_min_height', 'INT'),
        ('preresize_max_width', 'INT'), ('preresize_max_height', 'INT'),
        ('mask_fill_holes', 'BOOLEAN'), ('mask_expand_pixels', 'INT'), ('mask_invert', 'BOOLEAN'),
        ('mask_blend_pixels', 'INT'), ('mask_hipass_filter', 'FLOAT'),
        ('extend_for_outpainting', 'BOOLEAN'), ('extend_up_factor', 'FLOAT'), ('extend_down_factor', 'FLOAT'),
        ('extend_left_factor', 'FLOAT'), ('extend_right_factor', 'FLOAT'),
        ('context_from_mask_extend_factor', 'FLOAT'), ('output_resize_to_target_size', 'BOOLEAN'),
        ('output_target_width', 'INT'), ('output_target_height', 'INT'),
        ('output_padding', ['32']), ('device_mode', ['gpu (much faster)']),
    ]
    contracts['InpaintCropImproved'] = {k: (v,) for k, v in crop_types}
    info.update({k: {'input': {'required': v}} for k, v in contracts.items()})
    for loader, prop, values in [
        ('UNETLoader', 'unet_name', ['qwen_image_2.1_int8_convrot.safetensors', 'qwen_image_2.1_turbo_int8_convrot.safetensors']),
        ('CLIPLoader', 'clip_name', ['qwen3vl_8b_int8_convrot.safetensors']),
        ('VAELoader', 'vae_name', ['qwen_image_2.1_vae_bf16.safetensors']),
    ]:
        info[loader]['input']['required'][prop][0].extend(values)
    return info


class InpaintWorkflowTests(unittest.TestCase):
    def test_ui_links_and_model_setup(self):
        manifest = json.loads((ROOT / 'models.json').read_text())['models']
        for name in NAMES:
            with self.subTest(name=name):
                workflow = json.loads((ROOT / 'workflows' / name).read_text())
                nodes = {n['id']: n for n in workflow['nodes']}
                for lid, src, slot, dst, inp, _ in workflow['links']:
                    self.assertIn(lid, nodes[src]['outputs'][slot]['links'])
                    self.assertEqual(nodes[dst]['inputs'][inp]['link'], lid)
                for node in nodes.values():
                    for model in node.get('properties', {}).get('models', []):
                        entry = manifest[model['name']]
                        self.assertIn(name, entry['used_by'])
                        self.assertEqual(entry['directory'], model['directory'])
                        self.assertEqual(len(entry['sha256']), 64)

    def test_editor_contract_and_execution_with_one_or_all_references(self):
        info = object_info()
        for name in NAMES:
            with self.subTest(name=name):
                workflow = json.loads((ROOT / 'workflows' / name).read_text())
                api = _convert_ui_to_api(workflow, info)
                self.assertEqual(_validate_workflow(api, info), [])
                discovered = DiscoveredWorkflow(file_path=name, api_prompt=api, **_extract_stimma_nodes(api))
                descriptor = _build_single_tool(discovered, object_info=info, config=object(), provider=object()).to_descriptor()
                self.assertEqual(descriptor.task_types, ['inpaint-image'])
                props = descriptor.parameter_schema['properties']
                self.assertEqual(props['input_images']['x-min-items'], 1)
                self.assertEqual(props['input_images']['x-max-items'], 4 if 'MOSS' in name else 10)
                self.assertEqual(props['mask']['x-control'], 'mask_editor')
                self.assertEqual(props['mask']['x-source_image_field'], 'input_images')
                self.assertIn('mask', descriptor.parameter_schema['required'])
                for unused in ('width', 'height', 'output_megapixels', 'reference_resolution'):
                    self.assertNotIn(unused, props)
                for key in ('mask_grow', 'mask_feather', 'context', 'seed', 'loras'):
                    self.assertIn(key, props)
                refs = [n['node_id'] for n in discovered.field_nodes if n['class_type'] == 'StimmaImageParam']
                for missing in ([], refs[1:]):
                    prompt = copy.deepcopy(api)
                    _strip_unprovided_input_chains(prompt, missing, info)
                    _resolve_stimma_links(prompt)
                    by_type = {}
                    for nid, n in prompt.items():
                        by_type.setdefault(n['class_type'], []).append((nid, n['inputs']))
                    sampler = by_type['KSampler'][0][1]
                    mask_id, noise = by_type['SetLatentNoiseMask'][0]
                    crop_id, crop = by_type['InpaintCropImproved'][0]
                    encode_id, encode = by_type['VAEEncode'][0]
                    stitch_id, stitch = by_type['InpaintStitchImproved'][0]
                    self.assertEqual(sampler['latent_image'], [mask_id, 0])
                    self.assertEqual(noise['samples'], [encode_id, 0])
                    self.assertEqual(noise['mask'], [crop_id, 2])
                    self.assertEqual(encode['pixels'], [crop_id, 1])
                    self.assertEqual(stitch['stitcher'], [crop_id, 0])
                    self.assertEqual(by_type['StimmaImageOutput'][0][1]['images'], [stitch_id, 0])
                    self.assertEqual(crop['output_padding'], '32')
                    marked_id = by_type['DrawMaskOnImage'][0][0]
                    encoders = by_type.get('TextEncodeMOSSImageEdit', by_type.get('TextEncodeQwenImage21'))
                    for _, encoder in encoders:
                        self.assertEqual(encoder['images.image_0' if 'MOSS' in name else 'images.image_1'], [marked_id, 0])
                        if 'Qwen' in name:
                            self.assertEqual(encoder['resolution'], 0)
                    positive_id = sampler['positive'][0]
                    self.assertEqual(prompt[positive_id]['inputs']['prompt'], ['212', 0])
                    self.assertIn('green area', prompt['211']['inputs']['string_a'])
                    if 'Qwen' in name:
                        rgb_id, rgb = by_type['SplitImageWithAlpha'][0]
                        self.assertEqual(stitch['inpainted_image'], [rgb_id, 0])
                        self.assertEqual(rgb['image'], [by_type['VAEDecode'][0][0], 0])
                    self.assertEqual(sampler['steps'], 8 if 'Turbo' in name else 30 if 'MOSS' in name else 25)


if __name__ == '__main__':
    unittest.main()
