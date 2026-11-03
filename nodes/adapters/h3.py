"""MiniMax H3 state encoding and timeline conditioning; all H3 rules live here."""
from ..keyframes import KEYFRAMES_TYPE
from ..sidecars import SIDECARS_TYPE
from stimma_tools_protocol.sidecars import identity_mapping, map_interval
import hashlib
import os
from pathlib import Path
import uuid

FORMAT = "minimax_h3_latent"
FORMAT_VERSION = 1
LAYOUT = "comfy_h3_sampler_av_v1"


def _vae_id(vae):
    """Encoding identity belongs to this adapter, never to the host."""
    cached = getattr(vae, "_stimma_h3_encoding_id", None)
    if cached:
        return cached
    import torch
    model = vae.first_stage_model
    digest = hashlib.sha256(type(model).__qualname__.encode())
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    cached = digest.hexdigest()
    vae._stimma_h3_encoding_id = cached
    return cached


class StimmaH3GenerationState:
    """Export the original H3 sample as prepared files for a generic output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",), "vae": ("VAE",), "audio_vae": ("VAE",),
        }, "optional": {
            "keyframes": (KEYFRAMES_TYPE,),
            "_stimma_output_dir": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = (SIDECARS_TYPE,)
    FUNCTION = "execute"
    CATEGORY = "Stimma/State/MiniMax H3"

    def execute(self, samples, vae, audio_vae, keyframes=None, _stimma_output_dir=""):
        try:
            return self._export(samples, vae, audio_vae, keyframes, _stimma_output_dir)
        except (OSError, ValueError, RuntimeError):
            return ({"sidecars": [{"format": FORMAT, "format_version": FORMAT_VERSION,
                                   "display_label": "MiniMax H3 · video + audio", "descriptor": {},
                                   "status": "unavailable", "unavailable_reason": "capture_failed"}],
                     "input_usage": [dict(k.usage) for k in keyframes or [] if hasattr(k, "usage")]},)

    def _export(self, samples, vae, audio_vae, keyframes=None, _stimma_output_dir=""):
        import folder_paths
        from safetensors.torch import save_file
        from comfy.ldm.minimax.model import FRAME_PER_TOKEN

        video, audio = samples["samples"].tensors
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24 or audio.ndim != 4:
            raise ValueError("Unsupported H3 sample layout")
        frames = sum(FRAME_PER_TOKEN[k % 5] for k in range(video.shape[2]))
        root = Path(_stimma_output_dir or Path(folder_paths.get_temp_directory()) / "stimma_state" / uuid.uuid4().hex)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "sample.safetensors"
        tensors = {"video": video.detach().cpu().contiguous(), "audio": audio.detach().cpu().contiguous()}
        temp = path.with_suffix(".tmp")
        save_file(tensors, str(temp))
        os.replace(temp, path)
        descriptor = {
            "latent_layout": LAYOUT, "width": video.shape[-1] * 16,
            "height": video.shape[-2] * 16, "frame_count": frames,
            "fps": {"num": 24, "den": 1}, "audio_time_base": {"num": 1, "den": 40},
            "streams": ["video", "audio"], "video_vae": _vae_id(vae), "audio_vae": _vae_id(audio_vae),
            "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
        }
        companion = {
            "file": str(path), "format": FORMAT, "format_version": FORMAT_VERSION,
            "mime_type": "application/x-safetensors", "display_label": "MiniMax H3 · video + audio",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "byte_size": path.stat().st_size,
            "descriptor": descriptor, "media_mapping": identity_mapping(frames, 24, ["video", "audio"]),
            "render_facts": {"width": descriptor["width"], "height": descriptor["height"], "frame_count": frames, "fps": 24},
        }
        return ({"sidecars": [companion], "input_usage": [dict(k.usage) for k in keyframes or [] if hasattr(k, "usage")]},)


def _native_guide(kf, target, vae, audio_vae):
    """The experimentally verified 39-frame tail -> target-frame-zero path."""
    import folder_paths
    from safetensors import SafetensorError
    from safetensors.torch import load_file
    source = getattr(kf, "source", {})
    entries = source.get("sidecars", [])
    reason = "no_stored_state" if not entries else "no_compatible_sidecar"
    for entry in entries:
        if entry.get("format") != FORMAT or entry.get("format_version") != FORMAT_VERSION:
            continue
        if entry.get("status") == "unavailable":
            reason = entry.get("unavailable_reason", "state_unavailable")
            continue
        d = entry.get("descriptor") or {}
        if not isinstance(d, dict):
            reason = "invalid_descriptor"
            continue
        reason = "unsupported_interval"
        if kf["at"] != 0 or abs(source.get("length", 0) * 24 - 39) > .02:
            continue
        streams = ["video", "audio"] if source.get("sound") else ["video"]
        mapped = map_interval(entry.get("media_mapping"), source.get("from", 0), source.get("length", 0), streams)
        if mapped is None:
            reason = "coverage_unavailable"
            continue
        start, length = (float(v) * 24 for v in mapped)
        if abs(start + 39 - d.get("frame_count", 0)) > .02 or abs(length - 39) > .02:
            continue
        tv = target["samples"].tensors[0]
        if (d.get("width"), d.get("height"), d.get("fps"), d.get("latent_layout")) != (
                tv.shape[-1] * 16, tv.shape[-2] * 16, {"num": 24, "den": 1}, LAYOUT):
            reason = "encoding_mismatch"
            continue
        if tv.shape[2] < 12 or d.get("video_vae") != _vae_id(vae) or (source.get("sound") and d.get("audio_vae") != _vae_id(audio_vae)):
            reason = "encoding_mismatch"
            continue
        try:
            path = Path(folder_paths.get_annotated_filepath(entry["file"]))
            if entry.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
                reason = "integrity_failed"
                continue
            tensors = load_file(str(path))
            video, audio = tensors["video"], tensors["audio"]
            for name, tensor in tensors.items():
                if d["tensors"][name] != {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}:
                    raise ValueError("tensor_schema_mismatch")
            if video.shape[2] % 5 != 2 or video.shape[2] < 12 or tuple(video.shape[:2]) != (1, 24) or tuple(video.shape[-2:]) != tuple(tv.shape[-2:]):
                raise ValueError("tensor_schema_mismatch")
            expected_frames = (video.shape[2] // 5) * 17 + 5
            if d["frame_count"] != expected_frames or audio.shape[-1] != round(expected_frames / 24 * 40):
                raise ValueError("tensor_timing_mismatch")
            if tuple(audio.shape[:3]) != (1, 32, 2) or audio.shape[-1] < 65:
                raise ValueError("tensor_schema_mismatch")
            guide = {"resolved_frame_index": 0, "latent": video[:, :, -12:].clone()}
            if source.get("sound"):
                guide["audio_latent"] = audio[..., -65:].clone()
            kf.usage = {"source": "native_latent", "format": FORMAT, "format_version": FORMAT_VERSION,
                        "from": source.get("from", 0), "length": source.get("length"), "sound": bool(source.get("sound"))}
            return guide
        except (OSError, ValueError, KeyError, RuntimeError, SafetensorError):
            reason = "invalid_state"
    if hasattr(kf, "usage"):
        kf.usage = {"source": "rendered_video", "reason": reason, "from": source.get("from", 0), "length": source.get("length")}
    return None

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
                "keyframes": (KEYFRAMES_TYPE, {"x-accept-sidecars": [{"format": FORMAT, "format_version": FORMAT_VERSION}]}),
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
            native = _native_guide(kf, latent, vae, audio_vae)
            if native is not None:
                import node_helpers
                guides = list(positive[0][1].get("minimax_keyframes", [])) + [native]
                positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": guides})
                continue
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

