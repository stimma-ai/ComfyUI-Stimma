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

## Audio generation

Three bundled STP tools generate downloadable WAV audio:

| Tool | Output | Default sampling | Controls |
|------|--------|------------------|----------|
| **Stable Audio 3 Medium** (`stable-audio-3-medium`) | 44.1 kHz stereo, 1–380 seconds | 8 steps, LCM/simple, CFG 1 | Prompt, duration, seed, steps, sampler, scheduler |
| **Stable Audio 3 Medium Base** (`stable-audio-3-medium-base`) | 44.1 kHz stereo, 1–380 seconds | 50 steps, LCM/simple, CFG 7 | Above, plus guidance and negative prompt |
| **MOSS SoundEffect v2** (`moss-soundeffect-v2`) | 48 kHz mono, 0.1–30 seconds | 100 steps, guidance 4, shift 5 | Prompt, duration, seed, steps, guidance, negative prompt, schedule shift, duration suffix |

Stable Audio follows the [official ComfyUI recipes](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/audio_stable_audio_3_medium.json).
Use Medium for quick music, instruments, ambience, and sound effects; use Base
when you want classifier-free guidance and negative prompts. Guidance is fixed
at 1 on the distilled Medium tool because its recommended sampling does not
use a negative prompt. Describe instruments, style, tempo, and sound events in
the prompt. Duration defaults to 30 seconds. No automatic prompt rewriting is
applied. These workflows use ComfyUI's native model loading and memory manager.

[MOSS SoundEffect v2](https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_soundeffect_v2)
accepts English and Chinese descriptions of foley, sound effects, and ambient
environments. Duration defaults to 10 seconds. Keep the duration suffix enabled
to match its training convention. The model denoises a 30-second latent and
crops to the requested duration, so shorter clips do not proportionally reduce
generation time.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the plugin checkout:

```bash
tools/stimma-comfy setup-audio --comfyui /path/to/ComfyUI
tools/stimma-comfy audio-status --comfyui /path/to/ComfyUI
```

Use `--model stable-audio-3` or `--model moss` to install one family. The setup
command downloads pinned model revisions (approximately 20 GB for Stable Audio
and 11 GB for MOSS), plus MOSS's Python runtime. Stable Audio files are also
available through the manager's model download UI. Restart ComfyUI after setup.

MOSS requires an NVIDIA GPU and uses a separate Python 3.12 environment under
the plugin's ignored `.runtimes/` directory, with upstream's pinned dependencies.
It does not change ComfyUI's torch, transformers, or numpy packages. Each MOSS
job unloads that worker's ComfyUI models, loads MOSS in a child process on the
same GPU, and releases its GPU memory on completion or cancellation. This adds
model-loading time to each generation but lets image/video jobs reclaim the
GPU afterward. Progress and cancellation are forwarded through ComfyUI/STP.

Example STP calls (replace the provider alias with yours):

```bash
stp -s comfyui run stable-audio-3-medium "Warm jazz trio with brushed drums, upright bass and piano, 85 BPM" --duration 30 -o jazz.wav
stp -s comfyui run moss-soundeffect-v2 "Rain falling on a metal roof, distant rolling thunder, no voices" --duration 10 -o rain.wav
```

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
