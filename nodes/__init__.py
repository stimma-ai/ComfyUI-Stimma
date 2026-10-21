"""Stimma custom nodes for ComfyUI."""

from .tool_info import StimmaToolInfo
from .fields import (
    StimmaPromptParam,
    StimmaImageParam,
    StimmaMaskParam,
    StimmaImagesParam,
    StimmaVideoParam,
    StimmaVideosParam,
    StimmaAudioParam,
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
from .outputs import StimmaImageOutput, StimmaVideoOutput, StimmaAudioOutput
from .moss_audio import StimmaMossSoundEffect
from .layout import StimmaLayoutGroup
from .stitch_assembler import StimmaVideoStitchAssembler
from .outpaint import StimmaOutpaintPadding
from .image_seed import StimmaImageSeed
from .image_latent import StimmaImageModeLatent
from .flux3_prompt import StimmaFlux3PromptUpsample, StimmaFlux3Sampling
from .flux3_edit import StimmaFlux3References, StimmaCropToReference, StimmaPreserveUnchanged
from .keyframes import StimmaKeyframesParam, StimmaApplyKeyframesH3

NODE_CLASS_MAPPINGS = {
    "StimmaToolInfo": StimmaToolInfo,
    "StimmaPromptParam": StimmaPromptParam,
    "StimmaImageParam": StimmaImageParam,
    "StimmaMaskParam": StimmaMaskParam,
    "StimmaImagesParam": StimmaImagesParam,
    "StimmaVideoParam": StimmaVideoParam,
    "StimmaVideosParam": StimmaVideosParam,
    "StimmaAudioParam": StimmaAudioParam,
    "StimmaSeedParam": StimmaSeedParam,
    "StimmaImageSeed": StimmaImageSeed,
    "StimmaImageModeLatent": StimmaImageModeLatent,
    "StimmaFlux3Sampling": StimmaFlux3Sampling,
    "StimmaFlux3PromptUpsample": StimmaFlux3PromptUpsample,
    "StimmaFlux3References": StimmaFlux3References,
    "StimmaKeyframesParam": StimmaKeyframesParam,
    "StimmaApplyKeyframesH3": StimmaApplyKeyframesH3,
    "StimmaCropToReference": StimmaCropToReference,
    "StimmaPreserveUnchanged": StimmaPreserveUnchanged,
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
    "StimmaAudioOutput": StimmaAudioOutput,
    "StimmaMossSoundEffect": StimmaMossSoundEffect,
    "StimmaVideoOutput": StimmaVideoOutput,
    "StimmaLayoutGroup": StimmaLayoutGroup,
    "StimmaVideoStitchAssembler": StimmaVideoStitchAssembler,
    "StimmaOutpaintPadding": StimmaOutpaintPadding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StimmaToolInfo": "Stimma Tool Info",
    "StimmaPromptParam": "Stimma Prompt",
    "StimmaImageParam": "Stimma Image",
    "StimmaMaskParam": "Stimma Mask",
    "StimmaImagesParam": "Stimma Images",
    "StimmaVideoParam": "Stimma Video",
    "StimmaVideosParam": "Stimma Videos",
    "StimmaAudioParam": "Stimma Audio",
    "StimmaSeedParam": "Stimma Seed",
    "StimmaImageSeed": "Stimma Image Edit Seed",
    "StimmaImageModeLatent": "Stimma Generation / Edit Canvas",
    "StimmaFlux3Sampling": "Stimma FLUX.3 Sampling",
    "StimmaFlux3PromptUpsample": "Stimma FLUX.3 Prompt Upsampler",
    "StimmaFlux3References": "Stimma FLUX.3 References",
    "StimmaKeyframesParam": "Stimma Keyframes",
    "StimmaApplyKeyframesH3": "Stimma Apply Keyframes (MiniMax H3)",
    "StimmaCropToReference": "Stimma Crop To Reference",
    "StimmaPreserveUnchanged": "Stimma Preserve Unchanged Pixels",
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
    "StimmaAudioOutput": "Stimma Audio Output",
    "StimmaMossSoundEffect": "Stimma MOSS SoundEffect v2",
    "StimmaVideoOutput": "Stimma Video Output",
    "StimmaLayoutGroup": "Stimma Layout Group",
    "StimmaVideoStitchAssembler": "Stimma Video Stitch Assembler",
    "StimmaOutpaintPadding": "Stimma Outpaint Padding",
}
