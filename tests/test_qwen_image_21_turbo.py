"""Check Turbo keeps the base tool's unified generation/editing graph."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class QwenTurboTests(unittest.TestCase):
    def test_turbo_preserves_unified_graph(self):
        base = json.loads((ROOT / 'workflows/Stimma-Qwen-Image-2.1.json').read_text())
        turbo = json.loads((ROOT / 'workflows/Stimma-Qwen-Image-2.1-Turbo.json').read_text())
        self.assertEqual(base['links'], turbo['links'])
        nodes = {n['id']: n for n in turbo['nodes']}
        self.assertEqual(nodes[1]['widgets_values'][0], 'qwen-image-2.1-turbo')
        self.assertEqual(nodes[1]['widgets_values'][2], 'text-to-image,image-to-image')
        self.assertEqual(nodes[6]['widgets_values'][2:6], [8, 1.0, 'euler', 'simple'])
        self.assertEqual(nodes[12]['widgets_values'][1], 8)
        # All reference, seed, latent, resolution, and conditioning nodes stay identical.
        for node in base['nodes']:
            if node['id'] not in {1, 2, 6, 8, 12}:
                self.assertEqual(node, nodes[node['id']])

    def test_required_models_are_downloadable(self):
        name = 'Stimma-Qwen-Image-2.1-Turbo.json'
        workflow = json.loads((ROOT / 'workflows' / name).read_text())
        manifest = json.loads((ROOT / 'models.json').read_text())['models']
        for node in workflow['nodes']:
            for model in node.get('properties', {}).get('models', []):
                entry = manifest[model['name']]
                self.assertIn(name, entry['used_by'])
                self.assertEqual(entry['directory'], model['directory'])
                self.assertTrue(entry['size'] > 0)
                self.assertEqual(len(entry['sha256']), 64)
                self.assertIn(entry['source']['path'], model['url'])

if __name__ == '__main__':
    unittest.main()
