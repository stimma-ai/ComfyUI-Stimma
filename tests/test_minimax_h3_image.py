"""The fused H3 Image graph must survive omitted references in every mode."""
import copy
import json
import unittest
from pathlib import Path

from stp_server.discovery import _convert_ui_to_api, _extract_stimma_nodes, DiscoveredWorkflow
from stp_server.tool_builder import _build_single_tool
from stp_server.executor import _strip_unprovided_input_chains

ROOT = Path(__file__).resolve().parents[1]


class MiniMaxH3ImageTests(unittest.TestCase):
    def test_optional_references_keep_generation_and_edit_graph(self):
        workflow = json.loads((ROOT / 'workflows/Stimma-MiniMax-H3-Image.json').read_text())
        # Real V3 optional Autogrow schema: concrete sockets are dotted names.
        info = {'MiniMaxH3ImageConditioning': {'input': {'optional': {
            'images': ['COMFY_AUTOGROW_V3', {'template': {'prefix': 'image_', 'min': 0, 'max': 100}}]
        }}}}
        for count in (0, 1, 2, 16):
            with self.subTest(references=count):
                prompt = _convert_ui_to_api(copy.deepcopy(workflow))
                _strip_unprovided_input_chains(prompt, [str(100+i) for i in range(count+1, 17)], info)
                self.assertIn('32', prompt)
                self.assertIn('38', prompt)
                self.assertIn('8', prompt)
                inputs = prompt['32']['inputs']
                self.assertEqual(sum(k.startswith('images.') for k in inputs), count)
                for i in range(1, count+1):
                    self.assertEqual(inputs[f'images.image_{i}'], [str(100+i), 0])

    def test_editor_tools_advertise_required_task_contracts(self):
        for suffix, task in [('Inpaint', 'inpaint-image'), ('Outpaint', 'outpaint-image')]:
            name = f'Stimma-MiniMax-H3-Image-{suffix}.json'
            w = json.loads((ROOT/'workflows'/name).read_text())
            info = json.loads((ROOT/'tests/fixtures/minimax_h3_image_stimma_inputs.json').read_text())
            api = _convert_ui_to_api(w, info)
            discovered = DiscoveredWorkflow(file_path=name, api_prompt=api, **_extract_stimma_nodes(api))
            tool = _build_single_tool(discovered, object_info=info, config=object(), provider=object()).to_descriptor()
            self.assertEqual(tool.task_types, [task])
            props = tool.parameter_schema['properties']
            self.assertIn('input_images', tool.parameter_schema['required'])
            self.assertEqual(props['input_images']['x-min-items'], 1)
            self.assertNotIn('width', props)
            self.assertNotIn('height', props)
            self.assertEqual(discovered.tool_info['model'], 'minimax-h3-image')
            if suffix == 'Inpaint':
                self.assertIn('mask', tool.parameter_schema['required'])
                self.assertEqual(props['mask']['x-source_image_field'], 'input_images')
                self.assertEqual(props['input_images']['x-max-items'], 16)
                # Fresh image latent, then crop/stitch: no incompatible video latent.
                self.assertEqual(api['38']['inputs']['latent_image'], ['32', 1])
                self.assertEqual(api['209']['inputs']['stitcher'], ['204', 0])
                self.assertEqual(api['8']['inputs']['images'], ['217', 0])
                self.assertEqual(api['217']['inputs']['destination'], ['101', 0])
                self.assertEqual(api['217']['inputs']['source'], ['209', 0])
                self.assertEqual(api['217']['inputs']['mask'], ['216', 0])
                self.assertEqual(api['32']['inputs']['images.image_1'], ['210', 0])
                self.assertEqual(api['32']['inputs']['width'], ['215', 0])
            else:
                self.assertEqual(props['input_images']['x-max-items'], 1)
                for edge in ('top', 'bottom', 'left', 'right'):
                    self.assertIn(f'expand_{edge}_pct', props)
                self.assertEqual(api['42']['inputs']['destination'], ['22', 0])
                self.assertEqual(api['42']['inputs']['mask'], ['23', 1])
                self.assertEqual(api['43']['inputs']['images'], ['42', 0])
                self.assertEqual(api['70']['inputs']['images.image_1'], ['80', 0])
                self.assertEqual(api['60']['inputs']['latent_image'], ['85', 0])
                self.assertEqual(api['85']['inputs']['samples'], ['84', 0])
                self.assertEqual(api['85']['inputs']['mask'], ['86', 0])
                self.assertEqual(api['83']['inputs']['image'], ['22', 0])

    def test_sampling_controls_and_clean_input_copy(self):
        info = json.loads((ROOT/'tests/fixtures/minimax_h3_image_stimma_inputs.json').read_text())
        for suffix in ('', '-Inpaint', '-Outpaint'):
            name = f'Stimma-MiniMax-H3-Image{suffix}.json'
            w = json.loads((ROOT/'workflows'/name).read_text())
            api = _convert_ui_to_api(w, info)
            discovered = DiscoveredWorkflow(file_path=name, api_prompt=api, **_extract_stimma_nodes(api))
            tool = _build_single_tool(discovered, object_info=info, config=object(), provider=object()).to_descriptor()
            props = tool.parameter_schema['properties']
            for key, default in [('steps', 6), ('cfg', 1.0), ('sampler', 'euler'), ('scheduler', 'simple'), ('negative_prompt', '')]:
                self.assertEqual(props[key]['default'], default)
            self.assertIn('heun', props['sampler']['enum'])
            self.assertIn('loras', props)
            self.assertFalse(props['input_images'].get('description'))
            self.assertFalse(props.get('prompt', {}).get('description'))
            sampler = next(n['inputs'] for n in api.values() if n['class_type'] == 'KSampler')
            for key, node in [('steps', '300'), ('cfg', '301'), ('sampler_name', '302'), ('scheduler', '303')]:
                self.assertEqual(sampler[key], [node, 0])
            self.assertEqual(sampler['negative'], ['307', 0])
            self.assertEqual(api['307']['inputs']['text'], ['304', 0])
            self.assertEqual(sampler['model'], ['306', 0])

    def test_manifest_and_link_integrity(self):
        manifest = json.loads((ROOT/'models.json').read_text())['models']
        for suffix in ('', '-Inpaint', '-Outpaint'):
            name = f'Stimma-MiniMax-H3-Image{suffix}.json'
            w = json.loads((ROOT/'workflows'/name).read_text())
            nodes = {n['id']: n for n in w['nodes']}
            for lid, src, slot, dst, inp, typ in w['links']:
                self.assertIn(lid, nodes[src]['outputs'][slot]['links'])
                self.assertEqual(nodes[dst]['inputs'][inp]['link'], lid)
            for n in nodes.values():
                for model in n.get('properties', {}).get('models', []):
                    entry = manifest[model['name']]
                    self.assertIn(name, entry['used_by'])
                    self.assertEqual(len(entry['sha256']), 64)
            sampler = next(n for n in nodes.values() if n['type'] == 'KSampler')
            self.assertEqual(sampler['widgets_values'][2:], [6, 1, 'euler', 'simple', 1])



if __name__ == '__main__':
    unittest.main()
