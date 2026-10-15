"""Check the FLUX.3 Klein tools share one unified graph and downloadable models."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('Stimma-Flux3-Klein.json', 'Stimma-Flux3-Klein-Base.json')


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
        refs = [i['name'] for i in nodes[51]['inputs'] if i['name'].startswith('images.')]
        self.assertEqual(refs, [f'images.image{i}' for i in range(10)])
        base_nodes = {n['id']: n for n in base['nodes']}
        self.assertEqual(base_nodes[1]['widgets_values'][0], 'flux3-klein-base')
        self.assertEqual(base_nodes[46]['widgets_values'][1], 3.5)
        self.assertEqual(base_nodes[12]['widgets_values'][1], 50)

    def test_required_models_are_downloadable(self):
        manifest = json.loads((ROOT / 'models.json').read_text())['models']
        for name in NAMES:
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
