"""Stimma checkpoint loader node."""

import folder_paths
import comfy.sd


class StimmaCheckpointLoader:
    """Checkpoint loader with path_filter for Stimma integration.

    Wraps CheckpointLoaderSimple with a path_filter field that controls
    which checkpoints are exposed in the Stimma UI. The path_filter uses
    fnmatch glob patterns (e.g. "sdxl/**") to filter the checkpoint list.
    """

    @classmethod
    def INPUT_TYPES(cls):
        ckpt_list = folder_paths.get_filename_list("checkpoints")
        return {
            "required": {
                "ckpt_name": (ckpt_list,),
                "path_filter": ("STRING", {"default": ""}),
                "ui_order": ("INT", {"default": 50, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Loaders"

    def execute(self, ckpt_name, path_filter, ui_order):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, output_vae=True, output_clip=True,
        )
        return out[:3]  # model, clip, vae
