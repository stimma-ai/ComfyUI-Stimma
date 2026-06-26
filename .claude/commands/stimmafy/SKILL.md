---
name: stimmafy
description: >
  Convert ComfyUI workflows into Stimma-compatible tools or tweak existing ones.
  Use for stimmafying raw workflows, adjusting parameters/layout on existing
  Stimma workflows, or any workflow preparation for the Stimma platform. Also
  trigger when the user mentions converting, wrapping, or adapting workflows
  for Stimma, even if they don't use the exact word "stimmafy."
---

# Stimmafy

Convert a working ComfyUI workflow into a Stimma tool by adding an interface
layer of Stimma nodes. The core workflow logic stays untouched — you're adding
inputs, parameters, outputs, and metadata around it.

The user you are helping is bringing **their own** ComfyUI workflow and wants to
expose it as a Stimma tool. Don't assume anything about a particular model,
filename, or prior workflow — derive every decision from the workflow in front of
you and the heuristics in `references/`.

## Prerequisites

Confirm (or help the user set up) before testing:
- **ComfyUI is running locally** — default `localhost:8188`. If it's on another
  host/port, pass `--comfy-url HOST:PORT` to `test_workflow.py`.
- **The ComfyUI-Stimma plugin is installed** in that ComfyUI — the Stimma nodes
  must be importable or the test won't queue.
- The user has a **working** ComfyUI workflow JSON (it already runs in ComfyUI).
  We add an interface layer; we don't fix a broken graph.

## Scratch space

Write `plan.json`, intermediate workflow copies, and any one-off helper scripts
to the plugin's **`scratch/`** directory (gitignored — safe to fill with junk,
nothing there is committed). Don't scatter temp files in the repo root, the
user's workflow directory, or `/tmp`. The final stimmafied workflow does NOT go
in `scratch/` — see "Output file naming" below.

## Quick Reference

Scripts live at: `<skill-path>/scripts/`
Reference docs at: `<skill-path>/references/`

- `inspect_workflow.py <workflow.json>` — Inspect Stimma nodes in an existing workflow (tweak mode)
- `analyze_workflow.py <workflow.json>` — Analyze a workflow, output JSON report (full pipeline)
- `stimmafy.py <workflow.json> <plan.json> -o <output.json>` — Apply Stimma nodes
- `test_workflow.py <workflow.json>` — Queue to ComfyUI and verify execution

## Process Overview

1. **Analyze** the workflow structure
2. **Plan** which Stimma nodes to add (with user input)
3. **Build** the plan.json and run the modification script
4. **Test** by queuing to ComfyUI (localhost:8188)
5. **Iterate** if anything breaks

## Mode Detection

Before starting, determine which mode to use:

**Tweak Mode** — Use when ALL of these are true:
- The target workflow already contains Stimma nodes (has StimmaToolInfo)
- The user wants specific adjustments (not a full re-stimmafication)
- Examples: "change steps max to 50", "rename the parameter", "add negative_prompt to layout", "change the default sampler"

**Full Pipeline** — Use when:
- Converting a raw ComfyUI workflow to Stimma for the first time
- Major structural changes needed (new LoRA setup, different task type, new subgraph prep)
- Adding new Stimma nodes or changing wiring

## Tweak Mode

For modifying existing Stimma workflows directly, bypassing the plan/script pipeline.

### Step 1: Inspect

Run the inspect script to see the current Stimma configuration:

```bash
python3 <skill-path>/scripts/inspect_workflow.py <workflow.json>
```

This outputs all Stimma nodes with labeled `widgets_values` (including array indices),
wiring targets, and layout groups. Present the summary to the user and confirm the
requested changes.

### Step 2: Edit

Use the Edit tool to directly modify `widgets_values` in the workflow JSON.
The inspect output shows `index:name=value` for each widget, so you know exactly
which array position to change.

Common edits:
- **Change param default/range**: Find the node's `widgets_values` array, edit the value at the relevant index (e.g. index 1 for `value`, 3 for `maximum` on StimmaIntParam)
- **Rename a param**: Edit index 0 (`name`) in the node's `widgets_values`
- **Change default text**: Edit the `default_text` index in StimmaPromptParam's `widgets_values`
- **Change layout groups**: Edit the StimmaLayoutGroup node's `param_names` field (newline-separated)
- **Change tool metadata**: Edit StimmaToolInfo node's `widgets_values` (slug, display_name, task_types, badges, description, model_vendor, model)
- **Toggle layout collapsed**: Edit the `collapsed` boolean in StimmaLayoutGroup's `widgets_values`

### Step 3: Test

```bash
python3 <skill-path>/scripts/test_workflow.py <workflow.json> --timeout 120
```

---

## Full Pipeline

The full pipeline is for converting raw ComfyUI workflows to Stimma from scratch.

## Phase 1: Analyze

Run the analysis script:

```bash
python3 <skill-path>/scripts/analyze_workflow.py <workflow.json> --pretty
```

This outputs JSON with: model loaders, samplers, text encoders, save nodes,
latent image nodes, LoRA nodes, detected model family, guidance-distilled
status, task type, inner-subgraph sampling nodes (`subgraphs[]`), and wiring
recommendations (including `sampler_location`, `subgraph_prep_suggestions`, and
`warnings`).

Present a summary to the user:
- Model family detected (flux, sdxl, qwen, etc.)
- Whether it's guidance-distilled (affects which params to expose)
- Task type (text-to-image, image-to-image, etc.)
- How many samplers/text encoders/save nodes found
- **Where the sampler lives** (`recommendations.sampler_location`: `top-level`,
  `subgraph` = knobs still buried/action needed, `subgraph-exposed` = already
  wired out, `none`) and anything in `recommendations.warnings`
- Any ambiguities (multiple samplers, unclear text encoder roles)

## Phase 2: Plan

Read `references/model-heuristics.md` for model-specific decisions.

### Canonical Parameter Names — REQUIRED BY THE BACKEND/AGENT

A Stimma tool now exposes a **single `parameter_schema`** — there is no separate
`input_schema`. Every Stimma node (prompt, image, seed, resolution, steps, cfg, …)
becomes one entry in that one flat schema. The backend and the generation agent look
certain parameters up **by name**, so the media/prompt nodes below MUST be named
exactly as shown or the tool won't wire up correctly.

| Task type | Required parameter name(s) | Node |
|-----------|---------------------------|------|
| `text-to-image` | `prompt` | `StimmaPromptParam` (auto-named) |
| `image-to-image` | `prompt`, **`input_images`** (plural!) | `StimmaImageParam(name="input_images")` |
| `inpaint-image` | `input_image`, `mask` | `StimmaImageParam(name="input_image")` + `StimmaMaskParam` |
| `outpaint-image` | `input_image` | `StimmaImageParam(name="input_image")` |
| `upscale-image` | `input_image` | `StimmaImageParam(name="input_image")` |
| `text-to-video` | `prompt` | `StimmaPromptParam` (auto-named) |
| `image-to-video` | `input_image` (or `start_image`) | `StimmaImageParam(name="input_image")` |
| `video-extend` | `input_video` | `StimmaVideoParam(name="input_video")` |

**Critical naming rules:**
- `image-to-image` uses **`input_images`** (plural, with s) — NOT `image1`, NOT `input_image`
- All other image-input tasks use **`input_image`** (singular)
- `StimmaPromptParam` always produces `prompt` — no name change needed
- These names are matched in `parameter_schema.properties` by **name**, not by type

---

### What to always add — MANDATORY CHECKLIST
Every Stimma workflow MUST have ALL of the following. Missing any of these is a bug.

- **StimmaToolInfo** — slug, display_name, task_types, badges, description (optional: model_vendor, model)
- **StimmaPromptParam** — for the positive prompt (wire to CLIPTextEncode.text); produces field `prompt`
- **StimmaImageParam** (if image task) — name MUST match the task_type schema above (e.g. `input_images` for image-to-image)
- **Resolution control** — one of:
  - **StimmaResolutionParam** (for text-to-image/text-to-video) — wire to EmptyLatentImage/EmptyImage width/height
  - **StimmaFloatParam(megapixels)** + **ImageScaleToTotalPixels** (for image-to-image/image-to-video) — scales the input image, wire downstream. (Preserves the input image's aspect ratio while bounding total pixels.)
- **StimmaSeedParam** — wire to KSampler.seed or RandomNoise.seed
- **Sampling knobs that exist in the workflow** — see "Expose the sampling knobs" below. **Steps, sampler, and scheduler are exposed by default** whenever the workflow has a tunable sampler.
- **StimmaLoraLoader** — insert inline between model/checkpoint loader and downstream consumers. Takes MODEL + CLIP inputs, outputs MODEL + CLIP. Set path_filter to match model family (e.g. `*ltx*;*LTX*`). NEVER skip this.
- **StimmaImageOutput** or **StimmaVideoOutput** — replace each SaveImage/SaveVideo with a Stimma output
- **StimmaLayoutGroup** — group advanced params under "Advanced"

### Expose the sampling knobs — DON'T silently drop these
**Steps, sampler, and scheduler should almost always be exposed** when the
workflow has a sampler with those widgets — they are the knobs users most expect
to tune. Forgetting them is the single most common stimmafy mistake.

Drive this off the analysis output, then verify nothing is hidden:

Check `recommendations.sampler_location` first — it is `top-level`, `subgraph`,
`subgraph-exposed`, or `none`:

0. **`subgraph-exposed`** — the sampler is inside a subgraph but its knobs are
   ALREADY wired out to Stimma params (e.g. you're re-running on a finished tool).
   `subgraph_prep_suggestions` is empty and there are no warnings. Nothing to do.
1. **`top-level`** — for every entry in `recommendations.wiring.targets`, create
   the matching param. The analyzer emits targets for `steps`, `sampler_name`,
   `scheduler`, and (when applicable) `cfg`, `denoise`, `guidance`, `shift`. Each
   target that exists → one Stimma param wired to it. Use `StimmaIntParam` for
   steps, `StimmaDropdownParam` for sampler_name and scheduler (auto-resolves the
   enum).
2. **`subgraph`** — the sampler lives inside a subgraph, so it is absent from
   `wiring.targets`. The analyzer surfaces it anyway: use
   `recommendations.subgraph_prep_suggestions`, which contains ready-made
   `subgraph_prep.add_inner_inputs` entries (with `inner_node_id` /
   `inner_widget_name` already resolved) for steps/sampler/scheduler/seed (and cfg
   when not guidance-distilled). Drop those into the plan's `subgraph_prep`
   section, then wire Stimma params to the new boundary inputs. Also read
   `recommendations.warnings`. Do not ship a subgraph workflow with these knobs
   buried.
3. **`none`** — no sampler found at all. Only then is it legitimate to omit
   steps/sampler/scheduler (e.g. a pure API-style template with a fixed internal
   sampler that isn't promotable). Confirm with the user and say why.

### What to conditionally add
- **Negative prompt** (StimmaStringParam, NOT StimmaPromptParam) — only if NOT guidance-distilled. Use `type: "string"` with `ui_control: "textarea"`. Use StimmaStringParam because StimmaPromptParam always claims the canonical `prompt` name (reserved for the main positive prompt) and renders as the primary prompt editor; a negative prompt is just another named string parameter. Place it in a layout group with `full_width: true`.
- **CFG** (StimmaFloatParam) — only if NOT guidance-distilled
- **Denoise** (StimmaFloatParam) — only for img2img/inpaint task types
- **Guidance** (StimmaFloatParam) — if FluxGuidance node present (Flux models)
- **Shift** (StimmaFloatParam) — if ModelSamplingFlux/AuraFlow present
- **StimmaImageParam** — for img2img/inpaint workflows
- **StimmaMaskParam** — for inpaint workflows
- **StimmaVideoParam/Output** — for video workflows

### How to decide slug and display_name
- **slug**: lowercase, hyphenated. Use model name + variant. e.g. `flux-dev`, `qwen-image-2512`, `sdxl-lightning`
- **display_name**: Human-readable. e.g. "Flux Dev", "Qwen Image 2512", "SDXL Lightning"
- Ask the user if not obvious from the workflow filename/content

### Guidance-distilled heuristics
A model is guidance-distilled when CFG has no useful effect. Signals:
1. CFG = 1.0 in the KSampler widget values (strongest signal)
2. Filename contains: lightning, turbo, schnell, klein, hyper
3. Model family is flux (all variants)
4. Z-Image Turbo is always guidance-distilled

When guidance-distilled: skip CFG param, skip negative prompt.

### Layout strategy
- **Top level** (no group): prompt, reference images
- **"Video Settings" group** (not collapsed, for video): resolution/megapixels, duration, fps
- **"Advanced" group** (collapsed): steps, CFG, sampler, scheduler, denoise, guidance, shift, loras, seed
- Seed always goes last (ui_order 80+)
- "loras" must always be in the Advanced group param_names list

### Fields vs config params — one schema, two roles
Everything a Stimma tool exposes lands in a **single `parameter_schema`** — there is no
separate `input_schema` anymore. The split below is a *plan-authoring* convenience (it maps
to the `inputs` vs `params` sections of plan.json and to how the nodes render), not two
different schemas:
- **Fields** — primary per-generation content: prompt, images, seed, resolution. These use
  `StimmaPromptParam`, `StimmaImageParam`, `StimmaSeedParam`, `StimmaResolutionParam` and go
  in the plan's `inputs` section.
- **Config params** — knobs the user tunes: steps, CFG, sampler, scheduler, negative prompt,
  denoise, duration, fps. These use `StimmaIntParam`, `StimmaFloatParam`, `StimmaStringParam`,
  `StimmaDropdownParam`, `StimmaBoolParam` and go in the plan's `params` section.
- **Negative prompt is a config param** — use `StimmaStringParam` with `ui_control: "textarea"`
  (see the reasoning above; don't use StimmaPromptParam for it).
- **Duration and FPS are config params** (in the `params` section), consumed by
  `StimmaDurationToFrames` (added as a deferred entry in the `inputs` section).

### Ask the user when:
- Multiple KSamplers exist and it's unclear which to wire
- Task type is ambiguous
- Model family can't be detected
- LoRA path_filter preference (or leave blank)
- Any parameter the user might want to customize differently

## Phase 3: Build and Modify

Create a `plan.json` (write it to the plugin's `scratch/` directory) based on the
analysis and user decisions. See `references/node-catalog.md` for the exact format
of each node type.

The plan structure:
```json
{
  "tool_info": { ... },
  "inputs": [ ... ],
  "params": [ ... ],
  "lora_loader": { ... },
  "outputs": [ ... ],
  "layout": [ ... ]
}
```

The analysis output includes `recommendations.wiring.targets` which maps
parameter names to their target node_id and input_name — use these directly
in the plan's `wire_to` fields.

Run the modification:
```bash
python3 <skill-path>/scripts/stimmafy.py <source.json> plan.json -o <output.json>
```

### Output file naming
- **Save to ComfyUI's user workflows directory** as `Stimma-<SlugTitleCase>.json`
  (e.g. `Stimma-ZImage-Turbo.json`). This is the file ComfyUI/the Stimma plugin
  discovers and serves. Resolve the directory at runtime via ComfyUI's
  `folder_paths` (the `user/default/workflows` dir) rather than hardcoding a path.
- Use the slug to derive the filename: title-case each hyphen-separated word,
  join with hyphens, prefix with `Stimma-`. e.g. slug `qwen-image-2512` →
  `Stimma-Qwen-Image-2512.json`
- **Only if you are contributing to the ComfyUI-Stimma repo itself**: also copy
  the output into the repo's `workflows/` directory so it ships as a bundled
  sample. A user stimmafying their own workflow does NOT need this step — the
  user-directory copy above is what gets used.

### Visual layout
The script positions Stimma nodes to the LEFT of the existing workflow.
StimmaToolInfo goes top-left, inputs below it, params in a column to the
right, and the LoRA loader near the model chain. Output nodes go near the
original save nodes.

### Handling subgraphs and group nodes
If the workflow uses subgraph/component nodes (UUID-type nodes):
- **The analyzer descends into subgraph definitions** and reports inner
  sampling nodes under `subgraphs[]`, sets `recommendations.sampler_location` to
  `subgraph`, and emits ready-made `recommendations.subgraph_prep_suggestions`
  for the inner steps/sampler/scheduler/seed. Use those rather than hand-tracing
  (see "Expose the sampling knobs"). For samplers nested more than one level
  deep, the analyzer warns instead of guessing — expose through each boundary,
  innermost first.
- Stimma parameter nodes should be at the TOP LEVEL
- Wire into subgraph exposed inputs where possible
- **Use `subgraph_prep`** to expose inner node widgets/outputs as new subgraph inputs/outputs. This is the preferred approach — cleaner than Set/Get nodes.
  - `add_inner_inputs`: Expose inner node widgets (seed, steps, sampler_name, scheduler, negative_prompt, cfg) as new link inputs on the subgraph node
  - `add_inner_outputs`: Expose inner node outputs (e.g., IMAGE frames from VAEDecode) as new outputs on the subgraph node
  - See `references/node-catalog.md` → Subgraph Prep for the full format
- **How to find inner node IDs**: Open the source workflow JSON, search for `definitions.subgraphs`. Find the definition matching the subgraph node's type UUID. The inner nodes are in `defn.nodes[]` with their class_types and widget values.
- Keep the top-level graph clean and readable

### Handling multiple samplers
Some workflows (especially video, multi-stage) have 2+ KSamplers:
1. If they share the same model and similar params: wire the same Stimma nodes to both
2. If they're independent stages: create separate params with clear names (e.g. "steps_stage1", "steps_stage2")
3. When in doubt: ask the user

## Phase 4: Test

Run the workflow through the full executor pipeline against a live ComfyUI instance:

```bash
python3 <skill-path>/scripts/test_workflow.py <output.json> --timeout 120
```

For workflows with image inputs, provide a test image:

```bash
python3 <skill-path>/scripts/test_workflow.py <output.json> --test-image /path/to/test.jpg --timeout 120
```

This uses the **same code path as the real executor**:
1. `_convert_ui_to_api` — full UI→API conversion (subgraph expansion, group nodes, muted nodes)
2. `_inject_test_defaults` — randomizes seeds; injects test image if provided
3. `_resolve_stimma_links` — propagates Stimma node values to downstream ComfyUI nodes
4. Strips Stimma nodes + unknown node types (same transitive cascade as executor)
5. Queues via `/prompt`, monitors via WebSocket, reports outputs

If the test passes here, it will work through the STP executor.

If the test fails:
- Check for missing custom nodes (the Stimma plugin must be installed)
- Check for type mismatches in links (STRING vs COMBO is common)
- Check that muted nodes aren't accidentally depended upon
- Look at ComfyUI's console output for detailed error messages

## Phase 5: Iterate

After testing, review with the user:
- Did it produce the expected outputs?
- Are the right parameters exposed?
- Is the layout clean in the ComfyUI editor?
- Any parameters to add, remove, or rename?

Make adjustments by editing the plan.json and re-running, or by directly
editing the output workflow JSON.

## Batch Processing

When processing multiple workflows:
- Keep a consistent slug naming convention across the batch
- Reuse the same model-family heuristics
- Remember user preferences from earlier workflows (e.g., "always skip neg prompt for flux")
- Process one at a time for user review, unless the user says otherwise
- Use consistent ui_order values across similar workflows

## Common Patterns by Task Type

### text-to-image
Prompt + StimmaResolutionParam + seed + steps + sampler + scheduler (+ CFG/negative prompt unless guidance-distilled) + StimmaLoraLoader + StimmaImageOutput

### image-to-image
Same as t2i but with: StimmaImageParam + megapixels(StimmaFloatParam)+ImageScaleToTotalPixels instead of StimmaResolutionParam + denoise param (lower default, e.g. 0.7)

### inpaint
Same as i2i plus: StimmaMaskParam (with source_image_field referencing the image input)

### text-to-video
Prompt + resolution (StimmaResolutionParam) + duration + fps + seed + LoRA + video output.
- **Duration**: Use `StimmaFloatParam(duration)` (default 5.0s, 0.5-15.0, step 0.5) — NOT raw frame count
- **FPS**: Use `StimmaIntParam(fps)` (model-specific default, 8-60)
- **StimmaDurationToFrames**: Converter node that takes duration + fps → frame count. Set `frame_step` to the model's frame alignment (e.g. 4 for Wan, 8 for LTX2). Wire its output to the latent video node's length/frames input.
- **StimmaVideoOutput**: Replace SaveVideo nodes. Wire fps param to its fps input.
- Layout: "Video Settings" group (not collapsed) with resolution, duration, fps

### image-to-video
Same as t2v but with:
- **StimmaImageParam** for the start/reference image
- **StimmaFloatParam(megapixels)** + **ImageScaleToTotalPixels** instead of StimmaResolutionParam (preserves input image aspect ratio)
- Layout: "Video Settings" group with megapixels, duration, fps

### upscale
Usually simpler: image input + output, fewer params to tune.
May not need seed, sampler, or scheduler at all.
