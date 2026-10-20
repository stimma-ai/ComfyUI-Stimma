"""Stimma field nodes — capture user-provided data (images, prompts, etc.)."""

import os
import json
import time
import torch
import folder_paths
import numpy as np
from PIL import Image, ImageOps


def _safe_mtime_from_annotated(image_name: str) -> float:
    """Return mtime for annotated input file, falling back when missing."""
    try:
        image_path = folder_paths.get_annotated_filepath(image_name)
        return os.path.getmtime(image_path)
    except Exception:
        # Keep node execution resilient when a previous uploaded filename no longer exists.
        return time.time()


class StimmaPromptParam:
    """Text prompt input for Stimma tools."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "prompt"}),
                "default_text": ("STRING", {"default": "", "multiline": True}),
                "required": ("BOOLEAN", {"default": True}),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 100}),
                "ui_description": ("STRING", {"default": "Text prompt", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, name, default_text, required, ui_order, ui_description):
        return (default_text,)


class StimmaImageParam:
    """Single image input — works like ComfyUI's LoadImage."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        )
        return {
            "required": {
                "image": (files, {"image_upload": True}),
                "required": ("BOOLEAN", {"default": True}),
                "controlnet_types": ("STRING", {"default": "", "multiline": False}),
                "ui_control": (["image_picker", "video_frame_picker"],),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
            "optional": {
                # Show the Stimma prep controls (Scale / Extend Canvas / Paint)
                # on this input. Independent of controlnet support. Optional so
                # workflows saved before this widget existed still validate.
                "allow_prep": ("BOOLEAN", {"default": True}),
                # Agent-facing guidance for this input (role, order). Becomes the
                # parameter's schema description. Keep LAST: widgets are positional.
                "instructions": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, image, required=True, controlnet_types="", ui_control="image_picker", ui_order=0,
                allow_prep=True, instructions=""):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))
        image_out = img.convert("RGB")
        image_np = np.array(image_out).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        if "A" in img.getbands():
            mask_np = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask_np)[None,]
        else:
            mask_tensor = torch.zeros(
                (1, image_np.shape[0], image_np.shape[1]), dtype=torch.float32
            )

        return (image_tensor, mask_tensor)

    @classmethod
    def IS_CHANGED(cls, image, controlnet_types="", ui_control="image_picker", ui_order=0, **kwargs):
        return _safe_mtime_from_annotated(image)

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        image,
        controlnet_types="",
        ui_control="image_picker",
        ui_order=0,
        **kwargs,
    ):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


class StimmaMaskParam:
    """Mask input tied to a source image field — for inpainting workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        )
        return {
            "required": {
                "name": ("STRING", {"default": "mask"}),
                "image": (files, {"image_upload": True}),
                "source_image_field": ("STRING", {"default": "input_image"}),
                "ui_order": ("INT", {"default": 1, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, name, image, source_image_field="input_image", ui_order=1):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        if img.mode == "I":
            img = img.point(lambda i: i * (1 / 255))

        # Extract mask from alpha channel
        if "A" in img.getbands():
            mask_np = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask_np)[None,]
        else:
            # No alpha — treat as full mask
            mask_np = np.ones((img.height, img.width), dtype=np.float32)
            mask_tensor = torch.from_numpy(mask_np)[None,]

        # Also output as IMAGE
        image_out = img.convert("RGB")
        image_np = np.array(image_out).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]

        return (mask_tensor, image_tensor)

    @classmethod
    def IS_CHANGED(cls, name, image, source_image_field="input_image", ui_order=1, **kwargs):
        return _safe_mtime_from_annotated(image)

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        name,
        image,
        source_image_field="input_image",
        ui_order=1,
        **kwargs,
    ):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


class StimmaImagesParam:
    """Multiple image input — for batch workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        )
        return {
            "required": {
                "image": (files, {"image_upload": True}),
                "min_images": ("INT", {"default": 1, "min": 0, "max": 20}),
                "max_images": ("INT", {"default": 3, "min": 1, "max": 20}),
                "controlnet_types": ("STRING", {"default": "", "multiline": False}),
                "ui_control": (["image_picker", "video_frame_picker"],),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
            "optional": {
                # Show the Stimma prep controls (Scale / Extend Canvas / Paint).
                # Optional so pre-existing workflows still validate.
                "allow_prep": ("BOOLEAN", {"default": True}),
                "_stimma_images": ("STRING", {"default": "", "multiline": False}),
                # Agent-facing guidance for this input (role, order). Becomes the
                # parameter's schema description. Keep LAST: widgets are positional.
                "instructions": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(
        self,
        image,
        min_images=1,
        max_images=3,
        controlnet_types="",
        ui_control="image_picker",
        ui_order=0,
        allow_prep=True,
        _stimma_images="",
        instructions="",
    ):
        filenames = [image]
        if _stimma_images:
            try:
                parsed = json.loads(_stimma_images)
                if isinstance(parsed, list) and parsed:
                    filenames = [str(x) for x in parsed if str(x).strip()]
            except Exception:
                pass

        tensors = []
        for fname in filenames:
            image_path = folder_paths.get_annotated_filepath(fname)
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            if img.mode == "I":
                img = img.point(lambda i: i * (1 / 255))
            image_out = img.convert("RGB")
            image_np = np.array(image_out).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(image_np))

        if len(tensors) == 1:
            image_tensor = tensors[0][None,]
        else:
            # Normalize all refs to first image size so multi-ref batches are preserved.
            h0, w0 = tensors[0].shape[0], tensors[0].shape[1]
            norm = [tensors[0]]
            for t in tensors[1:]:
                if t.shape[0] == h0 and t.shape[1] == w0:
                    norm.append(t)
                    continue

                # PIL resize expects uint8 RGB.
                arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                pil = pil.resize((w0, h0), Image.Resampling.LANCZOS)
                arr2 = np.array(pil).astype(np.float32) / 255.0
                norm.append(torch.from_numpy(arr2))

            image_tensor = torch.stack(norm, dim=0)

        return (image_tensor,)

    @classmethod
    def IS_CHANGED(
        cls,
        image,
        min_images=1,
        max_images=3,
        controlnet_types="",
        ui_control="image_picker",
        ui_order=0,
        _stimma_images="",
        **kwargs,
    ):
        return _safe_mtime_from_annotated(image)

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        image,
        min_images=1,
        max_images=3,
        controlnet_types="",
        ui_control="image_picker",
        ui_order=0,
        _stimma_images="",
        **kwargs,
    ):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


def _load_video_audio(video_path, fallback_duration_s=0.0):
    """Extract the source audio track from a video as a ComfyUI AUDIO dict.

    Reuses ComfyUI's own av-based loader (the same one LoadAudio uses), which
    decodes the audio stream from any container including mp4. When the file has
    no audio stream (or decoding fails), returns silence spanning
    ``fallback_duration_s`` so downstream audio nodes (TrimAudioDuration,
    AudioConcat) see a track on the same timeline as the frames — a 1-sample
    placeholder makes any trim window fall outside the audio and error out.
    AudioConcat downstream reconciles sample rates, so we keep the source rate
    as-is.
    """
    import torch

    try:
        from comfy_extras.nodes_audio import load as _load_audio
        waveform, sample_rate = _load_audio(video_path)
        if waveform.shape[-1] > 0:
            return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}
    except Exception:
        pass
    # No audio stream (or no torchaudio/av) — emit video-length silence.
    n_samples = max(1, int(round(fallback_duration_s * 44100)))
    return {"waveform": torch.zeros((1, 2, n_samples), dtype=torch.float32), "sample_rate": 44100}


class StimmaVideoParam:
    """Video input — loads video frames as an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        )
        return {
            "required": {
                "video": (files,),
                "required": ("BOOLEAN", {"default": True}),
                "ui_control": (["video_picker"],),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 100}),
                "target_fps": ("INT", {"default": 0, "min": 0, "max": 240}),
            },
            "optional": {
                # Agent-facing guidance for this input (role, order). Becomes the
                # parameter's schema description. Keep LAST: widgets are positional.
                "instructions": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO")
    RETURN_NAMES = ("frames", "fps", "audio")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, video, required=True, ui_control="video_picker", ui_order=0,
                target_fps=0, instructions=""):
        import torch

        video_path = folder_paths.get_annotated_filepath(video)

        # Decode video frames with OpenCV first — the frame count sizes the
        # silent-audio fallback for videos without an audio track.
        frames_tensor = None
        source_fps = 30
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            source_fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or 30
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR → RGB, uint8 → float32 [0,1]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            cap.release()
            if frames:
                arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
                frames_tensor = torch.from_numpy(arr)
        except Exception:
            pass

        if frames_tensor is None:
            # Fallback: load as single image frame
            img = Image.open(video_path)
            img = img.convert("RGB")
            image_np = np.array(img).astype(np.float32) / 255.0
            frames_tensor = torch.from_numpy(image_np)[None,]
            source_fps = 30

        # Some models assign timestamps from a fixed frame rate rather than
        # accepting FPS as a separate input (MiniMax H3 reference video uses
        # 24 fps). Opt in per workflow; zero preserves legacy behavior.
        target_fps = int(target_fps or 0)
        if target_fps > 0 and source_fps > 0 and target_fps != source_fps:
            source_count = frames_tensor.shape[0]
            target_count = max(1, int(round(source_count * target_fps / source_fps)))
            indices = torch.linspace(0, source_count - 1, target_count).round().long()
            frames_tensor = frames_tensor[indices]
            source_fps = target_fps

        duration_s = frames_tensor.shape[0] / float(source_fps)
        audio = _load_video_audio(video_path, fallback_duration_s=duration_s)
        return (frames_tensor, source_fps, audio)

    @classmethod
    def IS_CHANGED(cls, video, required=True, ui_control="video_picker", ui_order=0,
                   target_fps=0, **kwargs):
        return _safe_mtime_from_annotated(video)

    @classmethod
    def VALIDATE_INPUTS(cls, video, required=True, ui_control="video_picker", ui_order=0,
                        target_fps=0, **kwargs):
        # Mirror StimmaImageParam: validate the uploaded file exists rather than
        # relying on ComfyUI's COMBO "value in list" check, which snapshots the
        # input directory listing and rejects freshly-uploaded videos.
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


class StimmaVideosParam:
    """Multiple video input — for workflows that can accept several videos."""

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        )
        return {
            "required": {
                "video": (files,),
                "min_videos": ("INT", {"default": 1, "min": 0, "max": 20}),
                "max_videos": ("INT", {"default": 3, "min": 1, "max": 20}),
                "ui_control": (["video_picker"],),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
            "optional": {
                # Agent-facing guidance for this input (role, order). Becomes the
                # parameter's schema description. Keep LAST: widgets are positional.
                "instructions": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, video, min_videos, max_videos, ui_control, ui_order, instructions=""):
        # Same placeholder behavior as StimmaVideoParam.
        return StimmaVideoParam().execute(video, ui_control=ui_control, ui_order=ui_order)

    @classmethod
    def IS_CHANGED(cls, video, min_videos=1, max_videos=3, ui_control="video_picker", ui_order=0, **kwargs):
        return _safe_mtime_from_annotated(video)

    @classmethod
    def VALIDATE_INPUTS(cls, video, min_videos=1, max_videos=3, ui_control="video_picker", ui_order=0, **kwargs):
        if not folder_paths.exists_annotated_filepath(video):
            return f"Invalid video file: {video}"
        return True


class StimmaAudioParam:
    """Audio input — loads an audio file as a ComfyUI AUDIO dict.

    For audio-conditioned tools (LTX image+audio-to-video, lip-sync, voice
    reference). Also reports the clip's duration so a workflow can size the
    video to the audio instead of asking the user to keep the two in sync.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = sorted(
            [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        )
        return {
            "required": {
                "audio": (files, {"audio_upload": True}),
                "required": ("BOOLEAN", {"default": True}),
                "ui_control": (["audio_picker"],),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 100}),
                "ui_label": ("STRING", {"default": "Audio"}),
                # driving  = the output reproduces this track (lip-sync, ia2v)
                # reference = the clip only steers generated audio (voice identity)
                "audio_role": (["driving", "reference"],),
            },
            "optional": {
                # Agent-facing guidance for this input (role, order). Becomes the
                # parameter's schema description. Keep LAST: widgets are positional.
                "instructions": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, audio, required=True, ui_control="audio_picker", ui_order=0,
                ui_label="Audio", audio_role="driving", instructions=""):
        audio_path = folder_paths.get_annotated_filepath(audio)

        # ComfyUI's own av-based loader — the same one LoadAudio uses, so we
        # accept every container it does (mp3/wav/m4a/flac/ogg).
        from comfy_extras.nodes_audio import load as _load_audio
        waveform, sample_rate = _load_audio(audio_path)
        sample_rate = int(sample_rate)
        duration_s = waveform.shape[-1] / float(sample_rate) if sample_rate else 0.0
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}, duration_s)

    @classmethod
    def IS_CHANGED(cls, audio, **kwargs):
        return _safe_mtime_from_annotated(audio)

    @classmethod
    def VALIDATE_INPUTS(cls, audio, **kwargs):
        # Mirrors StimmaImageParam/StimmaVideoParam: validate the file exists
        # rather than relying on the COMBO "value in list" check, which
        # snapshots the input directory and rejects freshly-uploaded files.
        if not folder_paths.exists_annotated_filepath(audio):
            return f"Invalid audio file: {audio}"
        return True


class StimmaSeedParam:
    """Seed input — provides a random or fixed seed value."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "seed"}),
                "value": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                }),
                "ui_order": ("INT", {"default": 99, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, name, value, ui_order):
        return (value,)
