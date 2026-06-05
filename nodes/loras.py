"""Stimma LoRA loader nodes."""

import folder_paths
import comfy.sd
import comfy.utils


class StimmaPairedLoraLoader:
    """Paired LoRA loader for Wan 2.2's dual-model architecture.

    Takes two model inputs (high_noise_model, low_noise_model), applies the
    correct LoRA variant to each model chain. Each slot has a high and low
    LoRA filename plus a shared strength.
    """

    MAX_LORAS = 10

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "high_noise_model": ("MODEL",),
                "low_noise_model": ("MODEL",),
                "path_filter": ("STRING", {"default": ""}),
                "ui_order": ("INT", {"default": 50, "min": 0, "max": 100}),
            },
            "optional": {},
        }
        for i in range(1, cls.MAX_LORAS + 1):
            inputs["optional"][f"lora_{i}"] = (lora_list,)
            inputs["optional"][f"lora_low_{i}"] = (lora_list,)
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
            })
        return inputs

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("high_noise_model", "low_noise_model")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Loaders"

    def execute(self, high_noise_model, low_noise_model, path_filter, ui_order, **kwargs):
        for i in range(1, self.MAX_LORAS + 1):
            hi_name = kwargs.get(f"lora_{i}", "None")
            lo_name = kwargs.get(f"lora_low_{i}", "None")
            strength = kwargs.get(f"strength_{i}", 1.0)
            if hi_name != "None" and strength != 0:
                lora_path = folder_paths.get_full_path_or_raise("loras", hi_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                high_noise_model, _ = comfy.sd.load_lora_for_models(
                    high_noise_model, None, lora, strength, 0,
                )
            if lo_name != "None" and strength != 0:
                lora_path = folder_paths.get_full_path_or_raise("loras", lo_name)
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                low_noise_model, _ = comfy.sd.load_lora_for_models(
                    low_noise_model, None, lora, strength, 0,
                )
        return (high_noise_model, low_noise_model)


class StimmaLoraLoader:
    """LoRA loader with up to 10 slots and path_filter for Stimma integration.

    Actually loads LoRAs in ComfyUI (works standalone without executor magic).
    The path_filter field uses fnmatch glob patterns to filter the LoRA list
    exposed to Stimma.
    """

    MAX_LORAS = 10

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "path_filter": ("STRING", {"default": ""}),
                "ui_order": ("INT", {"default": 50, "min": 0, "max": 100}),
            },
            "optional": {},
        }
        for i in range(1, cls.MAX_LORAS + 1):
            inputs["optional"][f"lora_{i}"] = (lora_list,)
            inputs["optional"][f"strength_{i}"] = ("FLOAT", {
                "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,
            })
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Loaders"

    def execute(self, model, clip, path_filter, ui_order, **kwargs):
        for i in range(1, self.MAX_LORAS + 1):
            lora_name = kwargs.get(f"lora_{i}", "None")
            strength = kwargs.get(f"strength_{i}", 1.0)
            if lora_name == "None" or strength == 0:
                continue
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength)
        return (model, clip)
