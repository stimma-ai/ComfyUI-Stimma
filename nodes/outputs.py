"""Stimma output nodes — capture generated images/videos."""

import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class StimmaImageOutput:
    """Image output node.

    In normal mode (no _stimma_output_dir): saves to ComfyUI output directory.
    In STP mode (_stimma_output_dir set by executor): saves to specified temp dir.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Stimma"}),
            },
            "optional": {
                "_stimma_output_dir": ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Stimma/Outputs"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute so output files are written to the current temp dir
        import time
        return time.time()

    def execute(self, images, filename_prefix="Stimma", _stimma_output_dir="",
                prompt=None, extra_pnginfo=None):
        import folder_paths

        results = []
        for i, image in enumerate(images):
            img_np = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key in extra_pnginfo:
                    metadata.add_text(key, json.dumps(extra_pnginfo[key]))

            if _stimma_output_dir:
                # STP mode — save to specified directory
                os.makedirs(_stimma_output_dir, exist_ok=True)
                filename = f"stimma_output_{i:04d}.png"
                filepath = os.path.join(_stimma_output_dir, filename)
                img.save(filepath, pnginfo=metadata, compress_level=4)
                results.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "output",
                })
            else:
                # Normal mode — save to ComfyUI output directory
                full_output_folder, filename, counter, subfolder, filename_prefix_out = (
                    folder_paths.get_save_image_path(
                        filename_prefix, folder_paths.get_output_directory(),
                        images[0].shape[1], images[0].shape[0]
                    )
                )
                filename_with_counter = f"{filename_prefix_out}_{counter:05d}.png"
                filepath = os.path.join(full_output_folder, filename_with_counter)
                img.save(filepath, pnginfo=metadata, compress_level=4)
                results.append({
                    "filename": filename_with_counter,
                    "subfolder": subfolder,
                    "type": "output",
                })

        return {"ui": {"images": results}}


class StimmaVideoOutput:
    """Video output node.

    In normal mode: saves to ComfyUI output directory as mp4.
    In STP mode: saves to specified temp dir.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("INT", {"default": 16, "min": 1, "max": 120}),
                "filename_prefix": ("STRING", {"default": "Stimma"}),
            },
            "optional": {
                "_stimma_output_dir": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Stimma/Outputs"

    def execute(self, frames, fps=16, filename_prefix="Stimma", _stimma_output_dir=""):
        import folder_paths
        import tempfile
        import subprocess

        # Convert frames to individual PNGs in a temp dir, then encode with ffmpeg
        temp_dir = tempfile.mkdtemp(prefix="stimma_video_")

        try:
            for i, frame in enumerate(frames):
                img_np = 255.0 * frame.cpu().numpy()
                img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                img.save(os.path.join(temp_dir, f"frame_{i:06d}.png"))

            if _stimma_output_dir:
                # STP mode
                os.makedirs(_stimma_output_dir, exist_ok=True)
                output_path = os.path.join(_stimma_output_dir, "stimma_output_0000.mp4")
            else:
                # Normal mode
                output_folder = folder_paths.get_output_directory()
                counter = 1
                while True:
                    output_path = os.path.join(
                        output_folder, f"{filename_prefix}_{counter:05d}.mp4"
                    )
                    if not os.path.exists(output_path):
                        break
                    counter += 1

            # Encode with ffmpeg
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg encoding failed: {result.stderr.decode()}")

            filename = os.path.basename(output_path)
            if _stimma_output_dir:
                return {"ui": {"videos": [{"filename": filename, "subfolder": "", "type": "output"}]}}
            else:
                return {"ui": {"videos": [{"filename": filename, "subfolder": "", "type": "output"}]}}

        finally:
            # Clean up temp frame PNGs
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
