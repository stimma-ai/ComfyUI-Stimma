"""Stimma custom nodes for ComfyUI."""

from .tool_info import StimmaToolInfo
from .fields import (
    StimmaPromptParam,
    StimmaImageParam,
    StimmaMaskParam,
    StimmaImagesParam,
    StimmaVideoParam,
    StimmaVideosParam,
    StimmaSeedParam,
)
from .params import (
    StimmaIntParam,
    StimmaFloatParam,
    StimmaStringParam,
    StimmaDropdownParam,
    StimmaResolutionParam,
    StimmaBoolParam,
    StimmaDurationToFrames,
)
from .loras import StimmaLoraLoader, StimmaPairedLoraLoader
from .checkpoints import StimmaCheckpointLoader
from .outputs import StimmaImageOutput, StimmaVideoOutput
from .layout import StimmaLayoutGroup

NODE_CLASS_MAPPINGS = {
    "StimmaToolInfo": StimmaToolInfo,
    "StimmaPromptParam": StimmaPromptParam,
    "StimmaImageParam": StimmaImageParam,
    "StimmaMaskParam": StimmaMaskParam,
    "StimmaImagesParam": StimmaImagesParam,
    "StimmaVideoParam": StimmaVideoParam,
    "StimmaVideosParam": StimmaVideosParam,
    "StimmaSeedParam": StimmaSeedParam,
    "StimmaIntParam": StimmaIntParam,
    "StimmaFloatParam": StimmaFloatParam,
    "StimmaStringParam": StimmaStringParam,
    "StimmaDropdownParam": StimmaDropdownParam,
    "StimmaResolutionParam": StimmaResolutionParam,
    "StimmaBoolParam": StimmaBoolParam,
    "StimmaDurationToFrames": StimmaDurationToFrames,
    "StimmaLoraLoader": StimmaLoraLoader,
    "StimmaPairedLoraLoader": StimmaPairedLoraLoader,
    "StimmaCheckpointLoader": StimmaCheckpointLoader,
    "StimmaImageOutput": StimmaImageOutput,
    "StimmaVideoOutput": StimmaVideoOutput,
    "StimmaLayoutGroup": StimmaLayoutGroup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StimmaToolInfo": "Stimma Tool Info",
    "StimmaPromptParam": "Stimma Prompt",
    "StimmaImageParam": "Stimma Image",
    "StimmaMaskParam": "Stimma Mask",
    "StimmaImagesParam": "Stimma Images",
    "StimmaVideoParam": "Stimma Video",
    "StimmaVideosParam": "Stimma Videos",
    "StimmaSeedParam": "Stimma Seed",
    "StimmaIntParam": "Stimma Int",
    "StimmaFloatParam": "Stimma Float",
    "StimmaStringParam": "Stimma String",
    "StimmaDropdownParam": "Stimma Dropdown",
    "StimmaResolutionParam": "Stimma Resolution",
    "StimmaBoolParam": "Stimma Bool",
    "StimmaDurationToFrames": "Stimma Duration to Frames",
    "StimmaLoraLoader": "Stimma LoRA Loader",
    "StimmaPairedLoraLoader": "Stimma Paired LoRA Loader",
    "StimmaCheckpointLoader": "Stimma Checkpoint Loader",
    "StimmaImageOutput": "Stimma Image Output",
    "StimmaVideoOutput": "Stimma Video Output",
    "StimmaLayoutGroup": "Stimma Layout Group",
}
