"""MOSS SoundEffect v2 with dependencies isolated from the ComfyUI process."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from audio_runtime import PLUGIN_ROOT, missing_moss_files, model_directory, runtime_python, runtime_ready


class StimmaMossSoundEffect:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "seconds": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 30.0, "step": 0.1}),
            "steps": ("INT", {"default": 100, "min": 1, "max": 300}),
            "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            "sigma_shift": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 20.0, "step": 0.1}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "append_duration_suffix": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Audio"

    def execute(self, prompt, seconds, steps, cfg, sigma_shift, seed,
                negative_prompt="", append_duration_suffix=True):
        import numpy as np
        import torch
        import folder_paths
        import comfy.model_management as mm
        from comfy.utils import ProgressBar

        if not prompt.strip():
            raise ValueError("Enter a sound description.")
        python = runtime_python()
        if not runtime_ready() or missing_moss_files(folder_paths.models_dir):
            raise RuntimeError("MOSS SoundEffect needs setup. Run tools/stimma-comfy setup-audio --model moss --comfyui <ComfyUI directory> from the plugin.")
        device = mm.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("MOSS SoundEffect currently requires an NVIDIA GPU.")
        # The child inherits CUDA_VISIBLE_DEVICES and uses the same logical device
        # as this ComfyUI worker. Never allocate on another worker's GPU.
        mm.unload_all_models()
        mm.soft_empty_cache()
        request = dict(
            model_directory=str(model_directory(folder_paths.models_dir)), device=str(device),
            prompt=prompt, seconds=seconds, num_inference_steps=steps, cfg_scale=cfg,
            sigma_shift=sigma_shift, seed=seed, negative_prompt=negative_prompt,
            append_duration_suffix=append_duration_suffix,
        )
        progress = ProgressBar(steps)
        with tempfile.TemporaryDirectory(prefix="stimma-moss-") as directory:
            job = Path(directory)
            (job / "request.json").write_text(json.dumps(request))
            env = dict(os.environ, PYTHONUNBUFFERED="1", TORCHDYNAMO_DISABLE="1")
            # Do not let a host PYTHONPATH inject ComfyUI's packages into the runtime.
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
            with (job / "worker.log").open("w+") as log:
                process = subprocess.Popen([str(python), str(PLUGIN_ROOT / "tools/moss_worker.py"), str(job)],
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    last = None
                    while process.poll() is None:
                        mm.throw_exception_if_processing_interrupted()
                        try:
                            current = json.loads((job / "progress.json").read_text())
                        except (OSError, ValueError):
                            current = None
                        if current and current != last:
                            progress.update_absolute(*current)
                            last = current
                        time.sleep(0.2)
                    if process.returncode:
                        log.seek(0)
                        raise RuntimeError("MOSS SoundEffect failed: " + log.read()[-5000:])
                finally:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
            waveform = torch.from_numpy(np.load(job / "audio.npy", allow_pickle=False))
            sample_rate = json.loads((job / "result.json").read_text())["sample_rate"]
            if waveform.ndim != 3 or not torch.isfinite(waveform).all():
                raise RuntimeError("MOSS returned invalid audio.")
            progress.update_absolute(steps, steps)
            return ({"waveform": waveform, "sample_rate": sample_rate},)
