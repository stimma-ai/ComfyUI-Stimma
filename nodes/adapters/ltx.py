"""LTX keyframe placement adapter."""
import torch
from ..keyframes import KEYFRAMES_TYPE

def _ltx_clip_frames(n, step=8):
    """Frames LTX keeps from an n-frame guide: 8k + 1."""
    return (max(1, n) - 1) // step * step + 1


def _ltx_clip_start(target, span, frame_count, step=8):
    """Nearest start LTX allows for a span-frame guide (span = 8k + 1) inside the video.

    A clip can start at frame 0, or at 8j + 1. Mid-video, LTXVAddGuide encodes
    the clip behind a throwaway frame and keeps its first span - 1 frames.
    """
    starts = [0] if span <= frame_count else []
    starts += range(1, frame_count - (span - 1) + 1, step)
    return min(starts, key=lambda s: abs(s - target)) if starts else 0


class StimmaApplyKeyframesLTX:
    """Pin Stimma keyframes onto an LTX-2 video via LTXVAddGuide.

    Apply it after each sampling stage's own guides, and crop with
    LTXVCropGuides (fed this node's conditioning) before upscaling or decoding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "img_compression": ("INT", {"default": 18, "min": 0, "max": 100,
                                            "tooltip": "LTXVPreprocess compression; 0 skips it."}),
            },
            "optional": {
                "keyframes": (KEYFRAMES_TYPE,),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Keyframes"

    def execute(self, positive, negative, vae, latent, fps, strength, img_compression, keyframes=None):
        if not keyframes:
            return (positive, negative, latent)
        import comfy.utils
        from comfy_extras.nodes_lt import LTXVAddGuide, get_keyframe_idxs, preprocess

        time_scale, width_scale, height_scale = vae.downscale_index_formula
        for kf in keyframes:
            samples = latent["samples"]
            _, _, latent_t, latent_h, latent_w = samples.shape
            _, guides = get_keyframe_idxs(positive, samples.shape)
            frame_count = (latent_t - guides - 1) * time_scale + 1

            image = kf["image"]
            if image.shape[0] > 1:
                want = _ltx_clip_frames(max(1, int(round(kf["length"] * fps))), time_scale)
                want = min(want, _ltx_clip_frames(frame_count, time_scale))
                idx = torch.linspace(0, image.shape[0] - 1, want).round().long()
                image = image[idx]
            span = image.shape[0]

            target = frame_count + int(round(kf["at"] * fps)) if kf["at"] < 0 else int(round(kf["at"] * fps))
            if span == 1:
                frame_idx = max(0, min(target, frame_count - 1))
            else:
                frame_idx = _ltx_clip_start(target, span, frame_count, time_scale)

            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1), latent_w * width_scale, latent_h * height_scale, "lanczos", "center",
            ).movedim(1, -1).clamp(0, 1)
            if img_compression:
                pixels = torch.stack([preprocess(frame, img_compression) for frame in pixels])
            positive, negative, latent = LTXVAddGuide.execute(
                positive=positive, negative=negative, vae=vae, latent=latent,
                image=pixels, frame_idx=frame_idx, strength=strength,
            ).result
        return (positive, negative, latent)
