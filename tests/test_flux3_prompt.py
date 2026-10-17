"""Check BFL sampling and upsampler routing without loading GPU models."""
import importlib.util
from pathlib import Path
import math
import sys
import types
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('flux3_prompt', Path(__file__).resolve().parents[1] / 'nodes/flux3_prompt.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PromptTests(unittest.TestCase):
    def test_disabled_does_not_load_a_model(self):
        self.assertEqual(module.StimmaFlux3PromptUpsample().execute('unchanged', False), ('unchanged',))

    def test_generation_and_ordered_different_size_edit_references(self):
        clip = Mock()
        clip.decode.return_value = ' Expanded caption '
        loader = Mock(return_value=types.SimpleNamespace(load_clip=Mock(return_value=(clip,))))
        with patch.dict(sys.modules, {'nodes': types.SimpleNamespace(CLIPLoader=loader)}):
            node = module.StimmaFlux3PromptUpsample()
            self.assertEqual(node.execute('a fox', seed=42), ('Expanded caption',))
            self.assertEqual(clip.tokenize.call_args.kwargs['images'], [])
            self.assertIn('text-to-image model', clip.tokenize.call_args.kwargs['system_prompt'])
            self.assertEqual(clip.generate.call_args.kwargs['presence_penalty'], 1.5)
            frame1, frame2 = Mock(), Mock()
            frame1.__getitem__ = Mock(return_value='first')
            frame2.__getitem__ = Mock(return_value='second')
            node.execute('change the cup', seed=42, image_1=[frame1], image_3=[frame2])
            self.assertEqual(clip.tokenize.call_args.kwargs['images'], ['first', 'second'])
            self.assertIn('image-editing model', clip.tokenize.call_args.kwargs['system_prompt'])
            self.assertEqual(clip.generate.call_args.kwargs['presence_penalty'], 0)
            self.assertEqual(clip.generate.call_args.kwargs['seed'], 42)
            clip.decode.return_value = ''
            with self.assertRaisesRegex(RuntimeError, 'empty caption'):
                node.execute('a fox')

    def test_alpha_uses_logarithmic_constant_shift(self):
        patcher = Mock(return_value=('patched',))
        with patch.dict(sys.modules, {'comfy_extras.nodes_model_advanced': types.SimpleNamespace(ModelSamplingFlux=lambda: types.SimpleNamespace(patch=patcher))}):
            self.assertEqual(module.StimmaFlux3Sampling().execute('model', 16), ('patched',))
        patcher.assert_called_once_with('model', math.log(16), math.log(16), 2048, 2048)


if __name__ == '__main__':
    unittest.main()
