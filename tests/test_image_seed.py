"""Regression coverage for independent, repeatable image-edit noise."""

import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "image_seed", Path(__file__).resolve().parents[1] / "nodes" / "image_seed.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

latent_spec = importlib.util.spec_from_file_location(
    "image_latent", Path(__file__).resolve().parents[1] / "nodes" / "image_latent.py"
)
latent_module = importlib.util.module_from_spec(latent_spec)
latent_spec.loader.exec_module(latent_module)


class ImageSeedTests(unittest.TestCase):
    def setUp(self):
        self.node = module.StimmaImageSeed()
        self.image = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3) / 32

    def test_repeatable_without_reusing_generation_seed(self):
        actual = self.node.execute(2142, self.image)
        self.assertEqual(actual, self.node.execute(2142, self.image.clone()))
        self.assertNotEqual(actual, (2142,))
        self.assertTrue(0 <= actual[0] <= 0xFFFFFFFFFFFFFFFF)

    def test_generation_without_image_preserves_requested_seed(self):
        self.assertEqual(self.node.execute(2142), (2142,))
        self.assertIn("image", self.node.INPUT_TYPES()["optional"])

    def test_new_reference_or_requested_seed_changes_noise(self):
        original = self.node.execute(2142, self.image)
        edited = self.image.clone()
        edited[0, 0, 0, 0] += 1 / 255
        self.assertNotEqual(original, self.node.execute(2142, edited))
        self.assertNotEqual(original, self.node.execute(2143, self.image))

    def test_equivalent_tensor_storage_has_identical_seed(self):
        original = self.node.execute(2142, self.image)
        self.assertEqual(original, self.node.execute(2142, self.image.double()))
        noncontiguous = self.image.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertEqual(original, self.node.execute(2142, noncontiguous))

    def test_input_and_random_state_are_unchanged(self):
        before = self.image.clone()
        state = torch.random.get_rng_state().clone()
        self.node.execute(0xFFFFFFFFFFFFFFFF, self.image)
        self.assertTrue(torch.equal(before, self.image))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


class ImageModeLatentTests(unittest.TestCase):
    def test_generation_uses_requested_canvas(self):
        generation = {"samples": torch.zeros(1, 4, 64, 96)}
        edit = {"samples": torch.zeros(1, 64, 64, 64)}
        actual = latent_module.StimmaImageModeLatent().execute(generation, edit)
        self.assertIs(actual[0], generation)

    def test_even_black_reference_uses_reference_canvas(self):
        generation = {"samples": torch.zeros(1, 4, 64, 96)}
        edit = {"samples": torch.zeros(1, 64, 64, 64)}
        actual = latent_module.StimmaImageModeLatent().execute(
            generation, edit, torch.zeros(1, 1024, 1024, 3)
        )
        self.assertIs(actual[0], edit)


if __name__ == "__main__":
    unittest.main()
