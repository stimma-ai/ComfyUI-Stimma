"""Stimma input nodes — capture user-provided data (images, prompts, etc.)."""

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
                "controlnet_types": ("STRING", {"default": "", "multiline": False}),
                "ui_control": (["image_picker", "video_frame_picker"],),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, image, controlnet_types="", ui_control="image_picker", ui_order=0):
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
                "_stimma_images": ("STRING", {"default": "", "multiline": False}),
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
        _stimma_images="",
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
                "ui_control": (["video_picker"],),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, video, ui_control, ui_order):
        import torch

        video_path = folder_paths.get_annotated_filepath(video)

        # Decode video frames with OpenCV
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
                return (torch.from_numpy(arr), source_fps)
        except Exception:
            pass

        # Fallback: load as single image frame
        img = Image.open(video_path)
        img = img.convert("RGB")
        image_np = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        return (image_tensor, 30)


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
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "fps")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, video, min_videos, max_videos, ui_control, ui_order):
        # Same placeholder behavior as StimmaVideoParam.
        return StimmaVideoParam().execute(video, ui_control, ui_order)


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
