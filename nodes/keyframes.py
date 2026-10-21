"""Keyframes: stills or short clips pinned at times on the output video.

`StimmaKeyframesParam` exposes the tool's `keyframes` parameter and loads each
pinned item. Model adapters (`StimmaApplyKeyframesH3`) turn the list into the
model's own guide conditioning, so a new video model only needs an adapter.
"""

import json
import os

import folder_paths
import numpy as np
import torch
from PIL import Image, ImageOps

KEYFRAMES_TYPE = "STIMMA_KEYFRAMES"


def _load_still(path):
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode == "I":
        img = img.point(lambda i: i * (1 / 255))
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def _is_still(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _read_video_frames(path, start_s, length_s, fps):
    """Frames of `path` from `start_s`: one frame when `length_s` is falsy, else
    `round(length_s * fps)` frames resampled to `fps`."""
    import cv2

    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(fps)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_idx = max(0, int(round(start_s * src_fps)))
    if total:
        start_idx = min(start_idx, total - 1)
    want = max(1, int(round(length_s * fps))) if length_s else 1
    src_needed = max(1, int(round(want * src_fps / fps))) if length_s else 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    frames = []
    while len(frames) < src_needed:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Could not read frames from {os.path.basename(path)} at {start_s:.2f}s")
    batch = torch.from_numpy(np.stack(frames, axis=0).astype(np.float32) / 255.0)
    if length_s and batch.shape[0] != want:
        idx = torch.linspace(0, batch.shape[0] - 1, want).round().long()
        batch = batch[idx]
    return batch


def _slice_audio(path, start_s, length_s):
    from .fields import _load_video_audio

    audio = _load_video_audio(path, fallback_duration_s=length_s)
    sr = int(audio["sample_rate"])
    a = int(round(start_s * sr))
    b = a + max(1, int(round(length_s * sr)))
    return {"waveform": audio["waveform"][..., a:b].contiguous(), "sample_rate": sr}


def load_keyframe(item, fps):
    """Load one wire item: {file, at, from?, length?, sound?} → loaded keyframe."""
    path = folder_paths.get_annotated_filepath(item["file"])
    start = float(item.get("from") or 0.0)
    length = float(item.get("length") or 0.0)
    if _is_still(path):
        image, length = _load_still(path), 0.0
    else:
        image = _read_video_frames(path, start, length, fps)
    audio = _slice_audio(path, start, length) if (length and item.get("sound")) else None
    return {"at": float(item.get("at") or 0.0), "image": image, "audio": audio, "length": length}


class StimmaKeyframesParam:
    """Stills or short clips pinned at times on the output video."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "keyframes"}),
                "max_items": ("INT", {"default": 6, "min": 1, "max": 32}),
                "positions": (["any", "ends"],),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "clip_min_frames": ("INT", {"default": 0, "min": 0, "max": 1000,
                                            "tooltip": "Shortest clip in frames; 0 means stills only."}),
                "clip_step_frames": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "allow_sound": ("BOOLEAN", {"default": False}),
                "timeline_param": ("STRING", {"default": "duration"}),
                "ui_order": ("INT", {"default": 15, "min": 0, "max": 100}),
            },
            "optional": {
                "_stimma_keyframes": ("STRING", {"default": "", "multiline": False}),
                # Agent-facing guidance. Keep LAST: widgets are positional.
                "instructions": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = (KEYFRAMES_TYPE,)
    RETURN_NAMES = ("keyframes",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, name, max_items, positions, fps, clip_min_frames, clip_step_frames,
                allow_sound, timeline_param, ui_order, _stimma_keyframes="", instructions=""):
        items = json.loads(_stimma_keyframes) if _stimma_keyframes else []
        return ([load_keyframe(item, fps) for item in items[:max_items]],)


def _h3_guide_frames(n):
    """Frames H3 keeps from an n-frame guide: one still, or 17k + 5 for a clip."""
    if n < 5:
        return 1
    while n % 17 != 5:
        n -= 1
    return n


class StimmaApplyKeyframesH3:
    """Pin Stimma keyframes onto a MiniMax H3 video via Add Guide for MiniMax H3."""

    FPS = 24

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
            },
            "optional": {
                "keyframes": (KEYFRAMES_TYPE,),
                # The requested duration. H3 rounds the length up to its frame grid,
                # so a keyframe at `at >= duration` means the last frame.
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Keyframes"

    def execute(self, positive, latent, vae, audio_vae, keyframes=None, duration=0.0):
        if not keyframes:
            return (positive,)
        from comfy.ldm.minimax.model import FRAME_PER_TOKEN
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

        latent_t = latent["samples"].tensors[0].shape[2]
        frame_count = sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))
        for kf in keyframes:
            span = _h3_guide_frames(kf["image"].shape[0])
            if duration and kf["at"] >= duration:
                idx = frame_count - span
            elif kf["at"] < 0:
                idx = frame_count + int(round(kf["at"] * self.FPS))
            else:
                idx = int(round(kf["at"] * self.FPS))
            idx = max(0, min(idx, frame_count - span))
            out = MiniMaxH3AddGuide.execute(
                positive=positive, latent=latent, frame_idx=idx, vae=vae,
                audio_vae=audio_vae if kf["audio"] is not None else None,
                image=kf["image"], audio=kf["audio"],
            )
            positive = out.result[0]
        return (positive,)
