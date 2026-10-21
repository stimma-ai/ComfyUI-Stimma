"""Keyframe loading and the H3 and LTX adapters' frame placement.

Needs ComfyUI's Python (torch, numpy, cv2, PIL): run with ComfyUI's venv,
`python tests/test_keyframes.py`.
"""

import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_module(tmpdir):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_annotated_filepath = lambda name: os.path.join(tmpdir, name)
    sys.modules["folder_paths"] = folder_paths
    pkg = types.ModuleType("stimma_nodes")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["stimma_nodes"] = pkg
    for name in ("fields", "keyframes"):
        spec = importlib.util.spec_from_file_location(f"stimma_nodes.{name}", ROOT / "nodes" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["stimma_nodes.keyframes"]


def _write_video(path, frames=48, fps=24, size=(64, 48)):
    import cv2

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        # Brightness encodes the frame index so tests can tell which frame they got.
        writer.write(np.full((size[1], size[0], 3), i * 5, dtype=np.uint8))
    writer.release()


class TestKeyframes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.kf = _load_module(cls.tmp)
        from PIL import Image

        Image.new("RGB", (32, 24), (200, 10, 10)).save(os.path.join(cls.tmp, "still.png"))
        _write_video(os.path.join(cls.tmp, "clip.mp4"))

    def _fake_audio(self, *_args, **_kwargs):
        return {"waveform": torch.arange(48000, dtype=torch.float32).reshape(1, 1, -1), "sample_rate": 16000}

    def test_still(self):
        kf = self.kf.load_keyframe({"file": "still.png", "at": 1.5}, 24)
        self.assertEqual(tuple(kf["image"].shape), (1, 24, 32, 3))
        self.assertEqual(kf["at"], 1.5)
        self.assertIsNone(kf["audio"])

    def test_video_frame_at_from(self):
        kf = self.kf.load_keyframe({"file": "clip.mp4", "at": 0, "from": 1.0}, 24)
        self.assertEqual(kf["image"].shape[0], 1)
        # Frame 24 was written with brightness 120; codecs may drift a little.
        self.assertAlmostEqual(float(kf["image"].mean()) * 255, 120, delta=8)

    def test_video_clip_with_sound(self):
        with patch.object(sys.modules["stimma_nodes.fields"], "_load_video_audio", self._fake_audio):
            kf = self.kf.load_keyframe(
                {"file": "clip.mp4", "at": -1, "from": 0.5, "length": 22 / 24, "sound": True}, 24)
        self.assertEqual(kf["image"].shape[0], 22)
        wave = kf["audio"]["waveform"]
        self.assertEqual(wave.shape[-1], round(22 / 24 * 16000))
        self.assertEqual(int(wave[..., 0]), 8000)

    def test_param_node_round_trip(self):
        node = self.kf.StimmaKeyframesParam()
        wire = json.dumps([{"file": "still.png", "at": 0}, {"file": "clip.mp4", "at": 2, "from": 0.2}])
        (out,) = node.execute("keyframes", 6, "any", 24, 5, 17, True, "duration", 15, wire, "")
        self.assertEqual([k["at"] for k in out], [0.0, 2.0])
        self.assertEqual(node.execute("keyframes", 6, "any", 24, 5, 17, True, "duration", 15, "", "")[0], [])

    def test_h3_guide_frames(self):
        self.assertEqual([self.kf._h3_guide_frames(n) for n in (1, 4, 5, 21, 22, 38, 39)], [1, 1, 5, 5, 22, 22, 39])

    def test_h3_adapter_places_and_clamps(self):
        calls = []

        class FakeAddGuide:
            @staticmethod
            def execute(**kw):
                calls.append((kw["frame_idx"], kw["image"].shape[0], kw["audio"] is not None, kw["audio_vae"]))
                return types.SimpleNamespace(result=(kw["positive"] + [kw["frame_idx"]],))

        comfy_minimax = types.ModuleType("comfy.ldm.minimax.model")
        comfy_minimax.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
        nodes_h3 = types.ModuleType("comfy_extras.nodes_minimax_h3")
        nodes_h3.MiniMaxH3AddGuide = FakeAddGuide
        # latent_t 37 -> 124 frames (5 s at 24 fps)
        latent = {"samples": types.SimpleNamespace(tensors=(torch.zeros(1, 24, 37, 2, 2), None))}
        still = torch.zeros(1, 4, 4, 3)
        clip = torch.zeros(22, 4, 4, 3)
        frames = [
            {"at": 0.0, "image": still, "audio": None},
            {"at": 2.0, "image": still, "audio": None},
            {"at": -1.0, "image": still, "audio": None},
            {"at": 5.0, "image": still, "audio": None},
            {"at": 5.0, "image": clip, "audio": {"waveform": None}},
        ]
        with patch.dict(sys.modules, {
            "comfy": types.ModuleType("comfy"), "comfy.ldm": types.ModuleType("comfy.ldm"),
            "comfy.ldm.minimax": types.ModuleType("comfy.ldm.minimax"), "comfy.ldm.minimax.model": comfy_minimax,
            "comfy_extras": types.ModuleType("comfy_extras"), "comfy_extras.nodes_minimax_h3": nodes_h3,
        }):
            (positive,) = self.kf.StimmaApplyKeyframesH3().execute([], latent, "vae", "avae", frames, 5.0)
            calls_no_duration = len(calls)
            self.kf.StimmaApplyKeyframesH3().execute([], latent, "vae", "avae", frames[3:4])
        # at >= duration pins the last frame (H3 rounds 5 s up to 124 frames); clips end there.
        self.assertEqual([c[0] for c in calls[:calls_no_duration]], [0, 48, 100, 123, 102])
        self.assertEqual(positive, [0, 48, 100, 123, 102])
        # Without the duration input, 5.0 s is simply frame 120.
        self.assertEqual(calls[-1][0], 120)
        calls = calls[:calls_no_duration]
        self.assertEqual([c[3] for c in calls], [None, None, None, None, "avae"])
        self.assertEqual(self.kf.StimmaApplyKeyframesH3().execute("pos", latent, "v", "a", None), ("pos",))

    def test_ltx_clip_start(self):
        # 5 s at 24 fps -> 121 frames. Starts are 0 or 8j + 1; a 17-frame clip
        # mid-video keeps 16 frames, so the last start is 105.
        start = self.kf._ltx_clip_start
        self.assertEqual(start(0, 17, 121), 0)
        self.assertEqual(start(48, 17, 121), 49)
        self.assertEqual(start(120 - 17, 17, 121), 105)
        self.assertEqual(start(500, 17, 121), 105)
        self.assertEqual([self.kf._ltx_clip_frames(n) for n in (1, 8, 9, 22, 25)], [1, 1, 9, 17, 25])

    def test_ltx_adapter_places_resamples_and_clamps(self):
        calls = []

        class FakeAddGuide:
            @staticmethod
            def execute(**kw):
                calls.append((kw["frame_idx"], kw["image"].shape[0], tuple(kw["image"].shape[1:3]), kw["strength"]))
                samples = torch.cat([kw["latent"]["samples"], torch.zeros(1, 4, 1, 3, 5)], dim=2)
                return types.SimpleNamespace(result=(kw["positive"] + [kw["frame_idx"]], kw["negative"], {"samples": samples}))

        def get_keyframe_idxs(cond, shape=None):
            return None, len(cond)

        nodes_lt = types.ModuleType("comfy_extras.nodes_lt")
        nodes_lt.LTXVAddGuide = FakeAddGuide
        nodes_lt.get_keyframe_idxs = get_keyframe_idxs
        nodes_lt.preprocess = lambda frame, crf: frame
        comfy_utils = types.ModuleType("comfy.utils")
        comfy_utils.common_upscale = lambda s, w, h, method, crop: torch.nn.functional.interpolate(s, size=(h, w))
        comfy = types.ModuleType("comfy")
        comfy.utils = comfy_utils
        vae = types.SimpleNamespace(downscale_index_formula=(8, 32, 32))
        # 16 latent frames -> 121 pixel frames (5 s at 24 fps), 96x160 pixels.
        latent = {"samples": torch.zeros(1, 4, 16, 3, 5)}
        still = torch.zeros(1, 40, 40, 3)
        clip = torch.zeros(22, 40, 40, 3)  # 22 frames loaded at 24 fps
        frames = [
            {"at": 0.0, "image": still, "audio": None, "length": 0.0},
            {"at": 2.0, "image": clip, "audio": None, "length": 22 / 24},
            {"at": -1.0, "image": still, "audio": None, "length": 0.0},
            {"at": 5.0, "image": still, "audio": None, "length": 0.0},
            {"at": 9.0, "image": clip, "audio": None, "length": 22 / 24},
        ]
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.utils": comfy_utils,
                                      "comfy_extras": types.ModuleType("comfy_extras"),
                                      "comfy_extras.nodes_lt": nodes_lt}):
            positive, negative, out = self.kf.StimmaApplyKeyframesLTX().execute(
                [], "neg", vae, latent, 24.0, 0.9, 18, frames)
            self.kf.StimmaApplyKeyframesLTX().execute([], "neg", vae, latent, 48.0, 1.0, 0, frames[1:2])
        # Appended guide frames don't shift later placements: each keeps 121 frames.
        self.assertEqual([c[0] for c in calls[:5]], [0, 49, 97, 120, 105])
        self.assertEqual([c[1] for c in calls[:5]], [1, 17, 1, 1, 17])
        self.assertEqual(calls[0][2], (96, 160))
        self.assertEqual(calls[0][3], 0.9)
        self.assertEqual(out["samples"].shape[2], 21)
        # At 48 fps a 22/24 s clip is 44 frames, kept as 41 (8k + 1).
        self.assertEqual(calls[5][1], 41)
        self.assertEqual(self.kf.StimmaApplyKeyframesLTX().execute("p", "n", vae, latent, 24.0, 1.0, 18, None),
                         ("p", "n", latent))


if __name__ == "__main__":
    unittest.main()
