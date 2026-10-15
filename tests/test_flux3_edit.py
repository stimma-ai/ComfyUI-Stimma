"""FLUX.3 Klein references follow Black Forest Labs' preparation geometry."""

import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    "flux3_edit", Path(__file__).resolve().parents[1] / "nodes" / "flux3_edit.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def image(width, height):
    return torch.rand(1, height, width, 3)


class Flux3ReferenceTests(unittest.TestCase):
    def test_single_reference_pads_without_resampling_and_crops_back(self):
        source = image(1000, 750)
        prepared, box = module.prepare_single(source)
        self.assertEqual(tuple(prepared.shape), (1, 752, 1008, 3))
        self.assertEqual(box, (4, 1, 1004, 751))
        (cropped,) = module.StimmaCropToReference().execute(prepared, box)
        self.assertTrue(torch.equal(cropped, source))

    def test_grid_aligned_reference_is_untouched(self):
        source = image(1024, 768)
        prepared, box = module.prepare_single(source)
        self.assertIsNone(box)
        self.assertTrue(torch.equal(prepared, source))
        self.assertEqual(module.StimmaCropToReference().execute(source, None)[0].shape, source.shape)

    def test_large_single_reference_fits_four_megapixels(self):
        (fit_w, fit_h), (w, h) = module.single_canvas(4000, 3000)
        self.assertLessEqual(w * h, 4 * 1024 * 1024)
        self.assertEqual((w % 16, h % 16), (0, 0))
        self.assertLessEqual(w - fit_w, 15)
        self.assertLessEqual(h - fit_h, 15)

    def test_multi_reference_fits_one_megapixel_on_the_grid(self):
        prepared = module.prepare_multi(image(1000, 750))
        self.assertEqual(tuple(prepared.shape), (1, 736, 992, 3))
        self.assertEqual(tuple(module.prepare_multi(image(2000, 1500)).shape)[1:3], (880, 1168))


class PreserveUnchangedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.source = torch.rand(1, 200, 300, 3)

    def test_local_edit_keeps_untouched_pixels_exact(self):
        edited = self.source + torch.randn_like(self.source) * 0.01  # re-rendered everywhere
        edited[:, 50:100, 50:120] = torch.tensor([1.0, 0.0, 0.0])  # the actual edit
        out = module.preserve_unchanged(self.source, edited)
        self.assertTrue(torch.equal(out[:, 150:, 200:], self.source[:, 150:, 200:]))
        self.assertTrue(torch.allclose(out[:, 60:90, 60:110], edited[:, 60:90, 60:110]))

    def test_isolated_specks_are_not_grown_into_patches(self):
        edited = self.source.clone()
        edited[:, 100:103, 100:103] = 1 - edited[:, 100:103, 100:103]
        self.assertTrue(torch.equal(module.preserve_unchanged(self.source, edited), self.source))

    def test_global_edits_and_size_mismatches_pass_through(self):
        restyled = 1 - self.source
        self.assertTrue(torch.equal(module.preserve_unchanged(self.source, restyled), restyled))
        other = torch.rand(1, 208, 304, 3)
        self.assertTrue(torch.equal(module.preserve_unchanged(self.source, other), other))
        node = module.StimmaPreserveUnchanged()
        self.assertIs(node.execute(restyled, True, None)[0], restyled)
        self.assertIs(node.execute(restyled, False, self.source)[0], restyled)


if __name__ == "__main__":
    unittest.main()
