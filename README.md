<p align="center">
  <a href="https://www.comfy.org"><img src="https://raw.githubusercontent.com/Comfy-Org/ComfyUI_frontend/main/public/assets/images/comfy-logo-single.svg" alt="ComfyUI" height="72"></a><img src="assets/times.svg" alt="×" height="72"><a href="https://stimma.ai"><img src="https://stimma.ai/logo.png" alt="Stimma" height="72"></a>
</p>
<h1 align="center">ComfyUI-Stimma</h1>

A ComfyUI plugin that exposes saved workflows as [Stimma](https://stimma.ai) tools via the Stimma Tools Protocol (STP). Stimma is built for the ComfyUI community — we know custom workflows are the heart of what makes ComfyUI powerful, and this plugin is designed to bring those workflows into the Stimma environment without compromise. Drop Stimma nodes into any ComfyUI workflow, save it, and it becomes a remotely callable tool.

## How It Works

1. Build a workflow in ComfyUI and add Stimma nodes (a `StimmaToolInfo` for metadata, inputs, parameters, outputs).
2. Save the workflow. The plugin scans workflow directories, discovers files containing Stimma nodes, and registers them as tools.
3. Stimma connects over WebSocket (`/stp-v1`) and can list, execute, and cancel tools.
4. On execution, the plugin injects user-provided values into the workflow, queues it to ComfyUI, monitors progress, and returns the output as an asset.

## Installation

Clone into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/stimma/ComfyUI-Stimma.git
pip install -r ComfyUI-Stimma/requirements.txt
```

Restart ComfyUI. The plugin registers its nodes and starts the STP server automatically.

## Nodes

### Metadata

| Node | Purpose |
|------|---------|
| **StimmaToolInfo** | Marks a workflow as a Stimma tool. Set the `slug` (unique ID), `display_name`, `task_type`, and `description`. |

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
```

### Multi-GPU

List multiple ComfyUI instances to load-balance across GPUs:

```yaml
comfyui:
  addresses:
    - "localhost:8188-8191"     # Port range expands to 4 instances
```

All instances must have the same models and custom nodes installed — the plugin treats them as interchangeable. If you want different models on different GPUs, run a separate plugin instance on each ComfyUI with a distinct `provider.id`.

## Exposing ComfyUI workflows

Stimma automatically scans the default user's workflows directory in ComfyUI looking for workflows that contain `StimmaToolInfo` nodes. These are automatically turned into tools for Stimma.

## Smoke-testing the bundled workflows

The [`stp`](https://github.com/stimma-ai/stimma-tools-protocol-cli) CLI can sweep every workflow this plugin exposes — running the cheapest valid generation of each and checking the output is a plausible asset. It's a fast way to surveil that the bundled workflows still work end-to-end after a ComfyUI/model update. Point it at a ComfyUI instance with this plugin loaded:

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

We are happy to take pull requests for further reference workflows if they are broadly of interest to the community. Most of the workflows so far are based on ComfyUI's stock workflows and nodes.
