"""Stimma parameter nodes — configurable parameters for tool users."""


class StimmaIntParam:
    """Integer parameter (steps, width, height, etc.)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "steps"}),
                "value": ("INT", {"default": 20, "min": -2147483648, "max": 2147483647, "display": "number"}),
                "minimum": ("INT", {"default": 1, "min": -2147483648, "max": 2147483647, "display": "number"}),
                "maximum": ("INT", {"default": 150, "min": -2147483648, "max": 2147483647, "display": "number"}),
                "step": ("INT", {"default": 1, "min": 1, "max": 1000, "display": "number"}),
                "ui_control": (["input", "slider", "upscale_resolution"],),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 2147483647, "display": "number"}),
                "ui_description": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, name, value, minimum, maximum, step, ui_control, ui_order, ui_description):
        return (value,)


class StimmaFloatParam:
    """Float parameter (cfg, guidance, denoise, etc.)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "cfg"}),
                "value": ("FLOAT", {"default": 4.0, "min": -1e9, "max": 1e9, "step": 0.01, "display": "number"}),
                "minimum": ("FLOAT", {"default": 0.0, "min": -1e9, "max": 1e9, "step": 0.01, "display": "number"}),
                "maximum": ("FLOAT", {"default": 20.0, "min": -1e9, "max": 1e9, "step": 0.01, "display": "number"}),
                "step": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 100.0, "step": 0.001, "display": "number"}),
                "ui_control": (["input", "slider", "upscale_resolution"],),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 2147483647, "display": "number"}),
                "ui_description": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, name, value, minimum, maximum, step, ui_control, ui_order, ui_description):
        return (value,)


class StimmaStringParam:
    """String parameter — dropdown (with enum_values) or free text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "sampler"}),
                "value": ("STRING", {"default": "euler"}),
                "enum_values": ("STRING", {"default": "", "multiline": True}),
                "ui_control": (["dropdown", "textarea"],),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 2147483647, "display": "number"}),
                "ui_description": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, name, value, enum_values, ui_control, ui_order, ui_description):
        return (value,)


class StimmaDropdownParam:
    """Dropdown parameter — enum auto-resolved from connected ComfyUI node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "sampler"}),
                "value": ("STRING", {"default": "euler"}),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 2147483647, "display": "number"}),
                "ui_description": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, name, value, ui_order, ui_description):
        return (value,)


class StimmaResolutionParam:
    """Resolution input — paired width/height with optional preset resolutions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 64, "display": "number"}),
                "min_size": ("INT", {"default": 512, "min": 64, "max": 16384, "display": "number"}),
                "max_size": ("INT", {"default": 4096, "min": 64, "max": 16384, "display": "number"}),
                "step": ("INT", {"default": 64, "min": 1, "max": 512, "display": "number"}),
                "supported_resolutions": ("STRING", {"default": "", "multiline": True}),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 100, "display": "number"}),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, width, height, min_size, max_size, step, supported_resolutions, ui_order):
        return (width, height)


class StimmaDurationToFrames:
    """Compute frame count from duration and fps, snapped to valid steps (e.g. 4n+1 for Wan)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 1000.0, "step": 0.1}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 120}),
                "frame_step": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Utils"

    def execute(self, duration, fps, frame_step):
        raw = duration * fps
        if frame_step > 1:
            n = max(0, round((raw - 1) / frame_step))
            frames = frame_step * n + 1
        else:
            frames = max(1, round(raw))
        return (frames,)


class StimmaBoolParam:
    """Boolean parameter — checkbox."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "enabled"}),
                "value": ("BOOLEAN", {"default": True}),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 2147483647, "display": "number"}),
                "ui_description": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("value",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, name, value, ui_order, ui_description):
        return (value,)
