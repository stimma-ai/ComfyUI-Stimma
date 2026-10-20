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


def object_info():
    # Ordered native widget contracts; model enums contain both released checkpoints.
    contracts = {
        'CLIPLoader': dict(clip_name=(['qwen3.5_9b_int8_convrot.safetensors'],), type=(['moss_image'],), device=(['default'],)),
        'VAELoader': dict(vae_name=(['flux2-vae_small.safetensors'],)),
        'UNETLoader': dict(unet_name=([f'moss_image_{v}_int8_convrot.safetensors' for v in ('sft', 'base')],), weight_dtype=(['default'],)),
        'EmptyFlux2LatentImage': dict(width=('INT',), height=('INT',), batch_size=('INT',)),
        'CLIPTextEncode': dict(clip=('CLIP',), text=('STRING',)),
        'KSampler': dict(model=('MODEL',), seed=('INT', {'control_after_generate': True}), steps=('INT',), cfg=('FLOAT',), sampler_name=(['euler'],), scheduler=(['sgm_uniform'],), positive=('CONDITIONING',), negative=('CONDITIONING',), latent_image=('LATENT',), denoise=('FLOAT',)),
        'ModelSamplingSD3': dict(model=('MODEL',), shift=('FLOAT',)),
        'SkipLayerGuidanceDiTSimple': dict(model=('MODEL',), double_layers=('STRING',), single_layers=('STRING',), start_percent=('FLOAT',), end_percent=('FLOAT',)),
        'VAEDecode': dict(samples=('LATENT',), vae=('VAE',)),
        'StimmaToolInfo': dict.fromkeys(['slug', 'display_name', 'task_types', 'badges', 'description', 'model_vendor', 'model'], ('STRING',)),
        'StimmaPromptParam': dict(name=('STRING',), default_text=('STRING',), required=('BOOLEAN',), ui_order=('INT',), ui_description=('STRING',)),
        'StimmaStringParam': dict(name=('STRING',), value=('STRING',), enum_values=('STRING',), ui_control=('STRING',), ui_order=('INT',), ui_description=('STRING',)),
        'StimmaResolutionParam': dict(width=('INT',), height=('INT',), min_size=('INT',), max_size=('INT',), step=('INT',), supported_resolutions=('STRING',), ui_order=('INT',), slider_max_pixels=('INT',)),
        'StimmaSeedParam': dict(name=('STRING',), value=('INT',), ui_order=('INT',)),
        'StimmaDropdownParam': dict(name=('STRING',), value=('STRING',), ui_order=('INT',), ui_description=('STRING',)),
        'StimmaImageOutput': dict(images=('IMAGE',), filename_prefix=('STRING',), _stimma_output_dir=('STRING',)),
        'StimmaLayoutGroup': dict(group_label=('STRING',), param_names=('STRING',), collapsed=('BOOLEAN',), ui_order=('INT',)),
        'StimmaLoraLoader': dict(model=('MODEL',), clip=('CLIP',), path_filter=('STRING',), ui_order=('INT',)),
    }
    for typ, scalar in [('StimmaIntParam', 'INT'), ('StimmaFloatParam', 'FLOAT')]:
        contracts[typ] = dict(name=('STRING',), value=(scalar,), minimum=(scalar,), maximum=(scalar,), step=(scalar,), ui_control=('STRING',), ui_order=('INT',), ui_description=('STRING',))
    for i in range(1, 11):
        contracts['StimmaLoraLoader'][f'lora_{i}'] = (['None'],)
        contracts['StimmaLoraLoader'][f'strength_{i}'] = ('FLOAT',)
    return {k: {'input': {'required': v}} for k, v in contracts.items()}


class MossImageTests(unittest.TestCase):
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
