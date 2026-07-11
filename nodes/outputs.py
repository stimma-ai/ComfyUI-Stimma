"""Stimma output nodes — capture generated images/videos."""

import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def _write_audio_wav(audio, wav_path):
    """Write a ComfyUI AUDIO dict ({waveform, sample_rate}) to a 16-bit PCM WAV.

    Uses only the stdlib `wave` module (torchaudio isn't available in the ComfyUI
    env). waveform is a tensor of shape [batch, channels, samples]; we take the
    first batch item. Returns the path on success, or None if there's no usable
    audio (so the caller falls back to a silent video).
    """
    import wave

    if not isinstance(audio, dict):
        return None
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate") or 0)
    if waveform is None or sample_rate <= 0:
        return None

    arr = waveform
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:  # [batch, channels, samples] -> first batch item
        arr = arr[0]
    if arr.ndim == 1:  # [samples] -> [1, samples]
        arr = arr[None, :]
    # arr is now [channels, samples]; WAV wants interleaved [samples, channels]
    channels = arr.shape[0]
    if channels == 0 or arr.shape[1] == 0:
        return None
    interleaved = arr.T  # [samples, channels]
    pcm = np.clip(interleaved, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return wav_path


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
                "audio": ("AUDIO",),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "_stimma_output_dir": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Stimma/Outputs"

    def execute(self, frames, fps=16, filename_prefix="Stimma", audio=None,
                generate_audio=True, _stimma_output_dir=""):
        import folder_paths
        import tempfile
        import subprocess
        import shutil

        try:
            import torch
        except ImportError:
            torch = None

        num_frames = int(frames.shape[0])
        height = int(frames.shape[1])
        width = int(frames.shape[2])

        temp_dir = None
        try:
            # Optionally write the audio track to a WAV for muxing. Disabled when
            # generate_audio is False or no audio is connected.
            audio_path = None
            if generate_audio and audio is not None:
                temp_dir = tempfile.mkdtemp(prefix="stimma_video_")
                audio_path = _write_audio_wav(audio, os.path.join(temp_dir, "audio.wav"))

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

            # Stream raw RGB frames straight into ffmpeg over stdin — no
            # intermediate PNG round-trip through the filesystem.
            cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "pipe:0",
            ]
            if audio_path:
                cmd += ["-i", audio_path]
            cmd += [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
            if audio_path:
                # Map the video stream + audio stream; end at the shorter input.
                cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
            cmd += [output_path]

            # stderr goes to a temp file (not a pipe) so ffmpeg can never block
            # on a full stderr buffer while we're blocked writing frames.
            with tempfile.TemporaryFile() as stderr_file:
                proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=stderr_file,
                )
                try:
                    chunk = 64
                    for start in range(0, num_frames, chunk):
                        batch = frames[start:start + chunk]
                        if batch.shape[-1] > 3:
                            batch = batch[..., :3]
                        if torch is not None and isinstance(batch, torch.Tensor):
                            arr = batch.mul(255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
                        else:
                            arr = np.clip(np.asarray(batch) * 255.0, 0, 255).astype(np.uint8)
                        proc.stdin.write(arr.tobytes())
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                finally:
                    returncode = proc.wait()
                if returncode != 0:
                    stderr_file.seek(0)
                    err = stderr_file.read().decode(errors="replace")
                    raise RuntimeError(f"ffmpeg encoding failed: {err[-4000:]}")

            filename = os.path.basename(output_path)
            if _stimma_output_dir:
                return {"ui": {"videos": [{"filename": filename, "subfolder": "", "type": "output"}]}}
            else:
                return {"ui": {"videos": [{"filename": filename, "subfolder": "", "type": "output"}]}}

        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
