"""Check the FLUX.3 Klein tools share one unified graph and downloadable models."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('Stimma-Flux3-Klein.json', 'Stimma-Flux3-Klein-Base.json')
ALL = NAMES + ('Stimma-Flux3-Klein-Inpaint.json', 'Stimma-Flux3-Klein-Outpaint.json')


def load(name):
    return json.loads((ROOT / 'workflows' / name).read_text())


def edges(workflow):
    return {tuple(link[1:5]) for link in workflow['links']}


class Flux3KleinTests(unittest.TestCase):
    def test_tools_share_reference_graph(self):
        distilled, base = (load(name) for name in NAMES)
        # Base adds only its CFG and negative prompt controls.
        self.assertEqual(edges(base) - edges(distilled), {(46, 0, 6, 3), (47, 0, 9, 1)})
        self.assertEqual(edges(distilled) - edges(base), set())
        nodes = {n['id']: n for n in distilled['nodes']}
        self.assertEqual(nodes[1]['widgets_values'][0], 'flux3-klein')
        self.assertEqual(nodes[1]['widgets_values'][2], 'text-to-image,image-to-image')
        self.assertEqual(nodes[3]['widgets_values'][1], 'flux3')
        self.assertEqual(nodes[6]['widgets_values'][2:6], [16, 1.0, 'euler', 'simple'])
        self.assertEqual(nodes[51]['type'], 'StimmaFlux3References')
        refs = [i['name'] for i in nodes[51]['inputs'] if i['name'].startswith('image_')]
        self.assertEqual(refs, [f'image_{i}' for i in range(1, 11)])
        # Decode -> crop padding -> restore unchanged pixels -> output.
        self.assertIn((7, 0, 52, 0), edges(distilled))
        self.assertIn((52, 0, 53, 0), edges(distilled))
        self.assertIn((53, 0, 8, 0), edges(distilled))
        base_nodes = {n['id']: n for n in base['nodes']}
        self.assertEqual(base_nodes[1]['widgets_values'][0], 'flux3-klein-base')
        self.assertEqual(base_nodes[46]['widgets_values'][1], 4.0)
        self.assertEqual(base_nodes[12]['widgets_values'][1], 50)

    def test_updated_bfl_sampling_and_prompt_route(self):
        for name in ALL:
            workflow = load(name)
            nodes = {n['id']: n for n in workflow['nodes']}
            patch = next(n for n in nodes.values() if n['type'] == 'StimmaFlux3Sampling')
            sampler = next(n for n in nodes.values() if n['type'] == 'KSampler')
            self.assertIn((patch['id'], 0, sampler['id'], 0), edges(workflow))
            alpha = next(n for n in nodes.values() if n['type'] == 'StimmaFloatParam' and n['widgets_values'][0] == 'timestep_alpha')
            self.assertEqual(alpha['widgets_values'][1], 16.0)
            self.assertIn((alpha['id'], 0, patch['id'], 1), edges(workflow))
            pup = next(n for n in nodes.values() if n['type'] == 'StimmaFlux3PromptUpsample')
            switch = next(n for n in nodes.values() if n['type'] == 'StimmaBoolParam' and n['widgets_values'][0] == 'prompt_upsampling')
            self.assertTrue(switch['widgets_values'][1])
            self.assertIn((switch['id'], 0, pup['id'], 1), edges(workflow))
            self.assertTrue(any(l[1] == pup['id'] and l[5] == 'STRING' for l in workflow['links']))
            if 'Outpaint' in name:
                self.assertIn((24, 0, pup['id'], 4), edges(workflow))
            for n in nodes.values():
                if n['type'] == 'StimmaResolutionParam':
                    self.assertEqual(n['widgets_values'][:2], [2048, 2048])
            # Every link must agree with both endpoint socket records.
            for lid, src, slot, dst, inp, typ in workflow['links']:
                self.assertIn(lid, nodes[src]['outputs'][slot]['links'])
                self.assertEqual(nodes[dst]['inputs'][inp]['link'], lid)

    def test_required_models_are_downloadable(self):
        manifest = json.loads((ROOT / 'models.json').read_text())['models']
        for name in ALL:
            for node in load(name)['nodes']:
                for model in node.get('properties', {}).get('models', []):
                    entry = manifest[model['name']]
                    self.assertIn(name, entry['used_by'])
                    self.assertEqual(entry['directory'], model['directory'])
                    self.assertTrue(entry['size'] > 0)
                    self.assertEqual(len(entry['sha256']), 64)
                    self.assertIn(entry['source']['path'], model['url'])


if __name__ == '__main__':
    unittest.main()
