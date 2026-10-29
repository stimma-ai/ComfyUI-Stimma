<p align="center">
  <a href="https://www.comfy.org"><img src="https://raw.githubusercontent.com/Comfy-Org/ComfyUI_frontend/main/public/assets/images/comfy-logo-single.svg" alt="ComfyUI" height="72"></a><img src="assets/times.svg" alt="×" height="72"><a href="https://stimma.ai"><img src="https://stimma.ai/logo.png" alt="Stimma" height="72"></a>
</p>
<h1 align="center">ComfyUI-Stimma</h1>

A ComfyUI plugin that exposes saved workflows as [Stimma](https://stimma.ai) tools via the [Stimma Tools Protocol (STP)](https://github.com/stimma-ai/stimma-tools-protocol). Stimma is built for the ComfyUI community — we know custom workflows are the heart of what makes ComfyUI powerful, and this plugin is designed to bring those workflows into the Stimma environment without compromise. Drop Stimma nodes into any ComfyUI workflow, save it, and it becomes a remotely callable tool.

## How It Works

1. Build a workflow in ComfyUI and add Stimma nodes (a `StimmaToolInfo` for metadata, inputs, parameters, outputs).
2. Save the workflow. The plugin scans workflow directories, discovers files containing Stimma nodes, and registers them as tools.
3. Stimma connects over WebSocket (`/stp-v1`) and can list, execute, and cancel tools.
4. On execution, the plugin injects user-provided values into the workflow, queues it to ComfyUI, monitors progress, and returns the output as an asset.
5. A small **manager** (served by the plugin at `/stp-v1/manage/`, and embedded in Stimma behind the ComfyUI icon in the top bar) shows status, the queue, GPU load, and every discovered workflow — and downloads the models a workflow is missing, installs missing node packs through ComfyUI-Manager, and restarts ComfyUI, so you never have to open ComfyUI again once the plugin is installed.

## Installation

Clone into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/stimma-ai/ComfyUI-Stimma.git
pip install -r ComfyUI-Stimma/requirements.txt
```

Restart ComfyUI. The plugin registers its nodes and starts the STP server automatically.

## Qwen3-TTS

Three bundled workflows use the 1.7B Qwen3-TTS models locally:

- **Qwen3-TTS Custom Voice** (`qwen3-tts-custom-voice`): choose a preset speaker
  and give delivery instructions. Aiden and Ryan are native English voices.
- **Qwen3-TTS Voice Design** (`qwen3-tts-voice-design`): describe a voice and
  speak sample text. Reuse an accepted sample with Voice Clone for consistent
  identity across subsequent jobs.
- **Qwen3-TTS Voice Clone** (`qwen3-tts-voice-clone`): supply one clean speech
  reference and new text. Qwen3-ASR transcribes the reference automatically.

Workflow setup installs
[ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS) and downloads
the selected model and its shared speech tokenizer into `models/qwen-tts`.
The clone workflow also needs ComfyUI-Drama3's Qwen3-ASR nodes and 0.6B ASR
checkpoint. Model files are pinned and SHA-256 verified by the manager.

The node pack was tested at revision
`374c2e05dc705231258237a16258d2f8fae8e32a`, which supports Transformers 5 as
well as 4.57.3. Do not downgrade an existing ComfyUI environment to install it.
Install missing dependencies from the pack's requirements using the same Python
environment as ComfyUI. The bundled workflows use PyTorch SDPA attention,
BF16 where supported, and release their TTS model after generation.

Put the words to speak in `prompt` and voice/delivery directions in `instruct`.
Choose the target `language` explicitly. For long narration, render sentence
or paragraph sections with the same speaker or reference, then assemble and
check the audio before producing video. The speech-token limit is a cap, not
a target duration. Output is a 24 kHz mono WAV.

## Fish Audio Drama 3

Three bundled workflows run speech generation locally:

- **Fish Audio Drama 3** (`fish-audio-drama-3-tts`): text to speech with a
  seed-generated voice. Use voice cloning to keep a chosen voice across jobs.
- **Fish Audio Drama 3 Voice Clone** (`fish-audio-drama-3-voice-clone`): supply
  one voice reference and the text to speak. The reference is transcribed
  automatically with Qwen3-ASR.
- **Fish Audio Drama 3 Dialogue** (`fish-audio-drama-3-dialogue`): supply two
  references, in speaker 0 / speaker 1 order, and start each script line with
  `<|speaker:0|>` or `<|speaker:1|>`.

Use the manager's workflow setup to install
[ComfyUI-Drama3](https://github.com/PedroMarinhoDev/ComfyUI-Drama3) and download
the models, then restart ComfyUI. The workflows were tested with node-pack
revision `67a733a96890637dc27b1914974ad00c569405fb`. The pack uses ComfyUI's
existing dependencies; no additional Python environment is required.

Model downloads are pinned and SHA-256 verified: the original FP8 expert
weights (`drama3_fp8experts_bf16.safetensors`, 38.6 GB) go in
`models/diffusion_models`, the codec (1.6 GB) in `models/vae`, and Qwen3-ASR
0.6B (1.6 GB, for the reference workflows) in `models/asr`. The default keeps
experts in system RAM and caches active experts in VRAM. Upstream recommends
an NVIDIA GPU with at least 8 GB VRAM and about 48 GB system RAM for FP8.

The prompt contains the words to speak. Use `[laughs]`, `[sighs]`, `[whispers]`,
or natural-language directions for delivery; parentheses are spoken. Voice
references should contain a few seconds of clean speech from one speaker,
cut between sentences so automatic transcription covers the whole clip.

Long scripts are split automatically with earlier speech as context. The
advanced `max_frames` setting limits **each chunk**, not the whole recording
(600 frames is about 28 seconds). Increase it for slow delivery; reaching it
can truncate speech. The defaults are temperature 1.0, top-p 0.9, and top-k 0
(no candidate cap). If a take becomes silent or skips words, try a new seed
or another reference. Output is a 44.1 kHz mono WAV. Built with Fish Audio.

## MOSS-Image

`MOSS-Image SFT` (`moss-image-sft`) is the recommended everyday generation
checkpoint. `MOSS-Image Base` (`moss-image-base`) preserves the pretraining
checkpoint for compatible LoRAs and fine-tuning. Both support natural-language
prompts, tag sequences, English/Chinese text rendering, and MOSS Layout JSON
in the prompt field (up to 4096 tokens).

Update ComfyUI to a build with native `moss_image` support in `CLIPLoader`
and `SkipLayerGuidanceDiTSimple`. The manager downloads the pinned Comfy-Org
INT8 ConvRot diffusion/text-encoder weights and the FLUX.2 small VAE with
SHA-256 verification. No extra custom nodes or isolated runtime are needed.

Defaults follow the [upstream guide](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md):
50 steps, CFG 3, shift 5, Euler / sgm_uniform, and negative-branch skipping
at layer 22. These checkpoints are not step-distilled. Advanced exposes the
negative prompt, steps, CFG, shift, sampler, scheduler, seed, and compatible
LoRAs under `moss-image/`.

Start at 1024×1024; 1–2 megapixels are recommended and 0.25–4 megapixels
are supported by the model. Dimensions use a 16-pixel grid. For layout control,
paste the model's JSON directly into Prompt and use the same canvas dimensions
used to calculate its boxes. For example, at 1024×1024:

```json
{"summary":{"caption":"A minimalist cream poster with green typography"},"items":[{"label":"text","caption":"Large green headline near the top","text":"OPEN STORIES","bbox":[-400,-400,-200,400]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
```

The released Base/SFT checkpoints are text-to-image models. They do not provide
reference-image instruction editing, dedicated inpainting, or outpainting;
separately trained community adapters have different requirements.

`MOSS-Image Base Turbo` (`moss-image-base-turbo`) and `MOSS-Image SFT Turbo`
(`moss-image-sft-turbo`) add Lodestone's eight-step DMD adapter trained on the
corresponding checkpoint. Defaults are 8 steps, CFG 1, shift 5, Euler/simple.
The speed adapters are fixed parts of these workflows; the optional LoRA
selector remains available for additional models under `moss-image/`.

`MOSS-Image SFT Edit` (`moss-image-edit-sft`) uses the community
[Edit-HQ armV5 adapter](https://huggingface.co/akhaliq/MOSS-Image-Edit-HQ)
on SFT. Update ComfyUI to include native `TextEncodeMOSSImageEdit` support
(merged October 22, 2026). The existing Qwen3.5 encoder includes the vision tower
this adapter needs. The workflow uses native ComfyUI nodes without an additional
plugin or runtime. Defaults are 30 steps, CFG 3, shift 5, Euler/simple and
negative-branch skipping at layer 22.

Supply one image to edit, optionally followed by up to three reference images.
Describe each image's role in the instruction. Output defaults to about one
megapixel at the first image's aspect ratio, aligned to 16 pixels. Advanced
exposes a separate reference pixel budget: keep the default one megapixel for
preservation, or try 0.25 for background replacement if the original scene
persists. Smaller references may change faces, text or other untouched details.
Pose changes and precise text edits remain limited. The requested seed is
combined with the source pixels to avoid reusing its generation noise.

The manager downloads all three adapters from pinned revisions with SHA-256
verification into `loras/moss-image-presets/`. Base editing and turbo editing
are not bundled.

## Qwen Image 2.1

`Qwen Image 2.1` (`qwen-image-2.1`) is one unified generation/editing tool.
With no images it generates using width/height (default 2048 × 2048).
With 1–10 images it edits the first image using the others as references.
Mention references as `<image1>`, `<image2>`, and so on. Edits follow the first
image's aspect ratio; `reference_resolution` controls the pixel budget
(default 1024, approximately one megapixel; 0 preserves input size).
Width/height apply only when generating without an image.

The workflow uses the official INT8 weights and defaults to 25 Euler/simple
steps, matching the current ComfyUI templates. Steps, CFG, negative prompt,
sampler, scheduler, reference resolution, optional LoRAs from `qwen-2.1/`,
and seed are exposed under Advanced. CFG defaults to 1, as Qwen Image 2.1 is
intended to sample without classifier-free guidance. For a LoRA or prompt that
benefits from guidance, set CFG above 1 and optionally enter a negative prompt;
this increases work per sampling step. At CFG 1 the negative prompt has no
effect. Denoise remains fixed at 1. Euler/simple is the default path.

When images are supplied, the workflow derives its sampler seed from the
requested seed and the first reference image's pixels. Reusing the exact
generation noise to edit a Qwen-generated image can cause severe sharpening
and changes to untouched backgrounds. Image-dependent edit noise avoids that reuse, including across
successive edits, while keeping the same inputs and seed reproducible. The
requested seed therefore differs from the raw KSampler seed during editing.
Without an image, the requested seed passes through unchanged.

This replaces `qwen-image-2.1-t2i` and `qwen-image-2.1-edit`. On startup, the
installer updates the generation workflow and removes the obsolete edit
workflow only if its installed copy is unchanged. User-modified copies are
preserved and can be migrated manually.

Update ComfyUI to a build containing `TextEncodeQwenImage21`. The manager can
download the three required files from
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1).
For transparent output, request an RGBA image with an alpha channel and a
transparent background; the output PNG preserves alpha.

`Qwen Image 2.1 Turbo` (`qwen-image-2.1-turbo`) offers the same generation,
editing, reference images, and RGBA controls using the official Turbo INT8
checkpoint. It defaults to 8 Euler/simple steps at CFG 1 and shares the base
model's text encoder and VAE. Install its separate diffusion checkpoint through
the manager; the Turbo tool is bundled alongside the base tool.

## FLUX.3 Klein

`FLUX.3 Klein` (`flux3-klein`) is one unified generation/editing tool built on
the official distilled checkpoint. With no images it generates using
width/height (default 2048 × 2048). With 1–10 images it edits the first image
using the others as references. Write generation prompts as detailed captions
and edits as instructions, naming inputs by ordinal ("the first image", "the
second image"); Stimma's prompt enhancement rewrites short requests into both
forms, following Black Forest Labs' guidance for external language models.
Enable **Enhance Prompt** in Stimma for short requests, or supply a detailed
caption yourself. The bundled workflows pass the resulting prompt through
without a second rewrite and do not require BFL's local upsampler checkpoint.

References are prepared the way Black Forest Labs' reference pipeline does it.
A single image keeps its own size (fitted to 4 MP only when larger) and is
reflect-padded to the 16 px latent grid; the padding is cropped off the result,
so the edit lines up with the input pixel for pixel. With several images, each
is fitted under 1 MP and the output follows the first. ComfyUI's generic
`ReferenceMultiLatent` path is not used because it resamples to a megapixel
budget and crops to the grid, shifting scale and framing.

`preserve_unchanged` (on by default, under Advanced) then copies the input's
exact pixels back wherever the edit left the image alone. Every output pixel
passes through the autoencoder and sampler, so untouched areas otherwise come
back slightly re-rendered. Changed regions are detected from the difference,
cleared of isolated specks, grown and feathered; whole-image edits such as a
restyle, and multi-reference edits whose size differs from the first image, are
returned as generated.

It defaults to 16 Euler/simple steps without guidance, matching the model card.
The timestep shift defaults to alpha 16, matching BFL's updated workflows.
4–8 steps work for drafts. Steps, sampler, scheduler, `timestep_alpha`,
`preserve_unchanged`, optional LoRAs from `flux3-klein/`, and seed are exposed under Advanced. Edit
noise is derived from the seed and first reference, as for Qwen Image 2.1.

`FLUX.3 Klein Base` (`flux3-klein-base`) uses the undistilled checkpoint with
the same graph, defaulting to 50 steps at CFG 4 with an optional negative
prompt. It is slower and intended for LoRAs and research.

`FLUX.3 Klein Inpaint` (`flux3-klein-inpaint`) and `FLUX.3 Klein Outpaint`
(`flux3-klein-outpaint`) match the Flux.2 Klein 9B inpaint and outpaint tools
parameter for parameter, so the image editor's Remove, Repaint and Expand use
them. Inpaint crops around the mask to 1024 × 1024, paints the masked area green
in the reference and asks for it to be repainted (FLUX.3 otherwise reproduces an
unmarked reference faithfully), locks everything outside the mask with a latent
noise mask, and stitches the result back. Outpaint marks the new canvas green,
asks for it to be filled, and composites the original back over it. In both,
pixels outside the mask or new canvas are kept exactly. Reference images on all
FLUX.3 tools accept canny, depth, lineart and pose preprocessing.

Update ComfyUI to v0.41.3 or later for the corrected FLUX.3 workflows. All four tools use the
official INT8 files from
[black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy),
which is gated: accept the FLUX Kommunity License on Hugging Face and give the
manager a token before downloading. The VAE is shared with FLUX.2.

## MiniMax H3 workflows

All H3 tools use conservative native Sol attention. I2V retains optional last-frame
conditioning; T2V generates without an image; R2V retains image, video and audio
references. Precision selection, user LoRA slots, resolution, duration, audio,
seed and sampler/scheduler controls are shared across the standard and ⚡ tools.
A collapsed Sol Tuning group exposes sparsity and its start/end window, with the
benchmarked defaults. Spectrum is not part of the bundled H3 graphs or their UI.

The separate H3 ⚡ tools add LightX2V eight-step 768p adapters. Their generation
defaults are eight steps, Euler/simple and shifts 6/3.

I2V and T2V use `minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`;
R2V uses the dedicated `minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors`.
Both are available in the [LightX2V model repository](https://huggingface.co/lightx2v/Minimax-h3-Turbo).
The manager's model manifest includes sizes and SHA256 hashes. These replace the
earlier four-step H3 turbo recipes and no longer require Larry's Turbo node pack.

Use a current ComfyUI build with native `ModelAttentionBackend`,
`MiniMaxH3SigmaShift` and `BlockSparseAttention` support, including the Comfy Kitchen
Sol kernels. The presets use tau 1.0, sparse attention from 20–90%, dense first/last
blocks, 256 extra tokens and dense conditioning/audio rows.
The recipe was evaluated at 1344×768; other sizes and extra LoRAs remain available
for experimentation. Ref2v uses its own adapter and needs separate quality judgment
from the i2v comparison.

## Nodes

### Metadata

| Node | Purpose |
|------|---------|
| **StimmaToolInfo** | Marks a workflow as a Stimma tool. Set the `slug` (unique ID), `display_name`, `task_types`, and `description`. |

Common STP task types include `text-to-image`, `image-to-image`, `text-to-video`,
`image-to-video`, `video-to-video`, `upscale-image`, and `upscale-video`.

### Fields

| Node | Purpose |
|------|---------|
| **StimmaPromptParam** | Text prompt input. Outputs `STRING`. |
| **StimmaImageParam** | Single image upload. Outputs `IMAGE` + `MASK`. |
| **StimmaMaskParam** | Mask input tied to a source image (inpainting). Outputs `MASK` + `IMAGE`. |
| **StimmaImagesParam** | Batch image upload. Outputs `IMAGE` batch. |
| **StimmaVideoParam** | Video upload — loads frames. Outputs `IMAGE` batch. |
| **StimmaVideosParam** | Batch video upload. Outputs `IMAGE` batch + `INT` fps. |
| **StimmaSeedParam** | Seed value. Auto-randomized if not provided. Outputs `INT`. |
| **StimmaResolutionParam** | Width/height pair with optional supported resolutions list. Outputs two `INT`s. |

### Parameters

| Node | Purpose |
|------|---------|
| **StimmaIntParam** | Integer parameter with min/max/step. |
| **StimmaFloatParam** | Float parameter with min/max/step. |
| **StimmaStringParam** | String parameter — free text or dropdown with explicit enum values. |
| **StimmaDropdownParam** | Enum auto-resolved from the connected ComfyUI node's spec (e.g., connect to KSampler's `sampler_name` and it picks up the valid values automatically). |
| **StimmaBoolParam** | Boolean checkbox. |
| **StimmaDurationToFrames** | Converts a duration (seconds) and fps to a frame count. Outputs `INT`. |

### LoRAs

| Node | Purpose |
|------|---------|
| **StimmaLoraLoader** | Up to 10 LoRA slots with strength control. Filters available LoRAs by `path_filter` (fnmatch glob, `;`-delimited). Wire its `MODEL`/`CLIP` outputs into your workflow. |
| **StimmaPairedLoraLoader** | Paired LoRA loader for high/low noise pipelines. Filters by `path_filter`. |

### Checkpoints

| Node | Purpose |
|------|---------|
| **StimmaCheckpointLoader** | Checkpoint selection with `path_filter` filtering. Outputs `MODEL`, `CLIP`, `VAE`. |

### Outputs

| Node | Purpose |
|------|---------|
| **StimmaImageOutput** | Captures generated images. Embeds ComfyUI metadata into PNGs. |
| **StimmaVideoOutput** | Captures generated video (encodes frames to MP4 via ffmpeg). |
| **StimmaAudioOutput** | Captures generated audio as 16-bit PCM WAV, preserving sample rate and channels. |

### Layout

| Node | Purpose |
|------|---------|
| **StimmaLayoutGroup** | Groups parameters into collapsible sections in the Stimma UI. |

## Configuration

Copy `config.yaml.default` to `config.yaml` and edit as needed:

```yaml
provider:
  id: comfyui                  # Unique provider identifier
  name: ComfyUI Workflows      # Display name

comfyui:
  addresses: []                # ComfyUI instance addresses (auto-detected if empty)
                               # Supports list, comma-separated, or port ranges (e.g. "localhost:8188-8191")

discovery:
  extra_workflow_dirs: []      # Additional directories to scan for workflows
  watch_interval: 2.0          # Seconds between filesystem polls (0 to disable)

credentials:                   # Written by the manager's Settings tab; optional
  huggingface_token: hf_...    # For gated model downloads
  civitai_api_key: ...
```

### Multi-GPU

List multiple ComfyUI instances to load-balance across GPUs:

```yaml
comfyui:
  addresses:
    - "localhost:8188-8191"     # Port range expands to 4 instances
```

All instances must have the same models and custom nodes installed — the plugin treats them as interchangeable. If you want different models on different GPUs, run a separate plugin instance on each ComfyUI with a distinct `provider.id`.

The instance Stimma connects to (the first address) hosts the manager and drives the others: it monitors their liveness (jobs stop being routed to an unreachable instance), fans downloads out to instances on *other* machines (instances on the same machine share the model directory and download once), and restarts all of them together.

## Exposing ComfyUI workflows

Stimma automatically scans the default user's workflows directory in ComfyUI looking for workflows that contain `StimmaToolInfo` nodes. These are automatically turned into tools for Stimma.

## Smoke-testing the bundled workflows

The [`stp`](https://github.com/stimma-ai/stimma-tools-protocol-cli) CLI can sweep every workflow this plugin exposes — running the cheapest valid generation of each and checking the output is a plausible asset. It's a fast way to verify that the bundled workflows still work end-to-end after a ComfyUI/model update. Point it at a ComfyUI instance with this plugin loaded:

```
stp --url ws://<comfyui-host>:8188/stp-v1 sweep --report sweep.json -o sweep-out/
```

Each workflow is reported PASS / FAIL / SKIP (SKIP = a referenced model isn't installed, or a required image/video input has no fixture). Fixtures for image-to-image / upscale / video workflows are fetched automatically from Stimma Cloud. Use `stp ... sweep --list` to preview the plan without running anything, and `--only <slug>` to test a single workflow. The run takes a while (one real generation per workflow), so it's a manual/periodic check rather than something to run constantly.

## Building Stimma workflows in ComfyUI

We highly recommend checking out the workflows in workflows/ to get an idea of how it is done. The general pattern is:

- Identify which inputs and parameters you wish to expose to Stimma users and add the corresponding Stimma nodes.
- Identify your output media, and wire that to a Stimma output node.
- Add a StimmaToolInfo node with the appropriate metadata to identify the tool.
- Add StimmaLayoutGroup as needed to organize properties into groups
- Add a StimmaLoraLoader to facilitate Lora Loading and configure path filters so that Stimma knows where you keep LoRAs relevant to that workflow
- Test in Stimma
- Iterate

The ComfyUI-Stimma plugin auto-reloads + updates tools as files change, so you should be able to change properties around and see results in Stimma as soon as you save the workflow on the ComfyUI side.

## Migrating Workflows with Claude Code

The easiest way to adapt an existing ComfyUI workflow into a Stimma tool is with [Claude Code](https://claude.ai/code) using the bundled `stimmafy` skill.

From the plugin directory:

```bash
claude
```

Then

```
> /stimmafy [tell it the workflow and what you want]
```

Tell it the path to the workflow file. It will create a new Stimmafy'd one in workflows/. 

The skill knows the Stimma nodes, tool building practices, and wiring patterns. It will analyze the workflow, prepare it for stimma and test it against your ComfyUI. If you want changes, discuss with Claude and you can iterate. All of the samples in workflows/ were built this way.

We are happy to take pull requests for further reference workflows if they are broadly of interest to the community. The bundled workflows combine stock and community ComfyUI workflows with workflows developed or adapted by the Stimma team.
