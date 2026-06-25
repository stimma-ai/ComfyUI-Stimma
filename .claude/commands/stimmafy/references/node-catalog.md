# Stimma Node Catalog

Reference for all Stimma node types in ComfyUI. Use when building plan.json.

## Plan JSON Format

The `stimmafy.py` script takes a plan.json with this structure:

```json
{
  "subgraph_prep": { "node_id": 92, "add_inner_inputs": [...], "add_inner_outputs": [...] },
  "pipeline_setup": { "delete": [94, 103], "mute": [75], "activate": [], "bypass": [] },
  "tool_info": { "slug": "...", "display_name": "...", "task_types": "...", "badges": "...", "description": "...", "model_vendor": "...", "model": "..." },
  "inputs": [ { "type": "prompt|seed|resolution|image|duration_to_frames", ... } ],
  "params": [ { "type": "int|float|string|dropdown|bool", ... } ],
  "checkpoint_loader": { "path_filter": "", "replace_node_id": N, "default": "" },
  "lora_loader": { "path_filter": "", "model_source": {...}, "clip_source": {...} },
  "outputs": [ { "replace_node_id": N, "output_type": "image|video" } ],
  "layout": [ { "label": "Advanced", "param_names": [...], "collapsed": true } ]
}
```

**Processing order**: subgraph_prep → pipeline_setup → tool_info → inputs → params → deferred inputs (duration_to_frames) → helper_nodes → checkpoint_loader → lora_loader → outputs → layout

## Subgraph Prep

When a workflow uses subgraph/component nodes (UUID-type nodes), inner node widgets and outputs aren't accessible from the top level. The `subgraph_prep` section exposes them as new definition inputs/outputs before any other processing.

```json
{
  "subgraph_prep": {
    "node_id": 92,
    "add_inner_inputs": [
      {"name": "seed", "type": "INT", "inner_node_id": 11, "inner_widget_name": "noise_seed"},
      {"name": "negative_prompt", "type": "STRING", "inner_node_id": 4, "inner_widget_name": "text"},
      {"name": "cfg", "type": "FLOAT", "inner_node_id": 47, "inner_widget_name": "cfg"},
      {"name": "sampler_name", "type": "COMBO", "inner_node_id": 8, "inner_widget_name": "sampler_name"}
    ],
    "add_inner_outputs": [
      {"name": "frames", "type": "IMAGE", "inner_node_id": 95, "inner_output_slot": 0}
    ]
  }
}
```

- `node_id`: The top-level subgraph node ID
- `add_inner_inputs`: Creates definition inputs on the -10 (input) node, with internal links to inner node widgets. Stimma params then wire to these via the subgraph node's link inputs.
- `add_inner_outputs`: Creates definition outputs on the -20 (output) node, with internal links from inner node outputs. The subgraph node gets a new output slot that Stimma output nodes can wire from.
- **Use case**: Exposing seed, negative_prompt, cfg, sampler_name from inner nodes of a collapsed subgraph. Also exposing intermediate outputs (e.g., IMAGE frames from an inner VAEDecode) so StimmaVideoOutput can capture them.
- **How to find inner node IDs**: Open the source workflow JSON, find the subgraph definition in `definitions.subgraphs[]` matching the subgraph node's type UUID. Inner nodes are in `defn.nodes[]`.

### Pipeline Setup

Cleanup before adding Stimma nodes:

```json
{
  "pipeline_setup": {
    "delete": [94, 103, 104],
    "mute": [75],
    "activate": [],
    "bypass": [{"node_id": 10, "input_slot": 0, "output_slot": 0}]
  }
}
```

- `delete`: Remove nodes entirely (e.g., MarkdownNotes, original LoadImage nodes replaced by StimmaImageParam)
- `mute`: Set mode=4 (NEVER) — node stays but doesn't execute. Used for SaveImage/SaveVideo replaced by Stimma outputs.
- `activate`: Set mode=0 (ALWAYS) — un-mute nodes
- `bypass`: Reconnect a node's input directly to its output consumers, then remove

## Node Types

### StimmaToolInfo
**Required — exactly one per workflow.**

```json
{ "slug": "my-tool-slug", "display_name": "My Tool", "task_types": "text-to-image", "badges": "Open Weights", "description": "Tool description", "model_vendor": "Black Forest Labs", "model": "FLUX.1 [dev]" }
```

- **slug**: Unique identifier (lowercase, hyphens). Convention: `model-variant` e.g. `flux-dev`, `qwen-image-2512`
- **task_types**: Comma-separated list of task types (the first is treated as primary). Values: text-to-image, image-to-image, inpaint-image, outpaint-image, upscale-image, style-transfer, text-to-video, image-to-video, video-extend, other. (Aliases like `image-to-image`→`image-edit` are normalized at discovery.)
- **badges**: Optional. Newline-separated short labels shown in the UI (e.g. `Open Weights`).
- **model_vendor** / **model**: Optional. Free-text vendor and model name (e.g. `Black Forest Labs` / `FLUX.1 [dev]`). Emit them as a pair or omit both.
- widgets_values order: `[slug, display_name, task_types, badges, description]`, plus `[model_vendor, model]` only when provided. (There is no `task_type` or `model_family` field anymore.)

### StimmaPromptParam
Text prompt. Connects to CLIPTextEncode's `text` input.

Plan entry:
```json
{
  "type": "prompt", "name": "prompt",
  "default_text": "A beautiful sunset over mountains",
  "required": true, "ui_order": 0,
  "ui_description": "Describe what you want to generate",
  "wire_to": { "node_id": 6, "input_name": "text" }
}
```

- **Do NOT use StimmaPromptParam for negative prompts.** Use StimmaStringParam instead (see below). StimmaPromptParam claims the canonical `prompt` parameter and renders as the primary prompt editor; a negative prompt should be a separate named string parameter (textarea).
- widgets_values: `[name, default_text, required, ui_order, ui_description]`
- Output: STRING (slot 0)

### StimmaSeedParam
Random seed. Connects to KSampler's `seed` or RandomNoise's `seed`.

```json
{
  "type": "seed", "name": "seed", "value": 0, "ui_order": 80,
  "wire_to": { "node_id": 3, "input_name": "seed" }
}
```

- widgets_values: `[name, value, ui_order]`
- Output: INT (slot 0)

### StimmaResolutionParam
Width/height pair. Connects to EmptyLatentImage or EmptySD3LatentImage.

```json
{
  "type": "resolution", "width": 1024, "height": 1024,
  "min_size": 512, "max_size": 2048, "step": 64,
  "supported_resolutions": "1024x1024\n1024x768\n768x1024",
  "ui_order": 1,
  "wire_to": { "node_id": 11, "width_input": "width", "height_input": "height" }
}
```

- `supported_resolutions`: Newline-separated "WxH" strings. Leave empty for free-form.
- widgets_values: `[width, height, min_size, max_size, step, supported_resolutions, ui_order]`
- Outputs: width INT (slot 0), height INT (slot 1)

### StimmaIntParam
Integer parameter (e.g., steps).

```json
{
  "type": "int", "name": "steps", "value": 20,
  "min": 1, "max": 100, "step": 1,
  "ui_control": "input", "ui_order": 20, "ui_description": "Number of sampling steps",
  "wire_to": { "node_id": 3, "input_name": "steps" }
}
```

- `ui_control`: "input" (number box) or "slider"
- widgets_values: `[name, value, minimum, maximum, step, ui_control, ui_order, ui_description]`
- Output: INT (slot 0)

### StimmaFloatParam
Float parameter (e.g., CFG, denoise, guidance, shift).

```json
{
  "type": "float", "name": "cfg_scale", "value": 7.5,
  "min": 1.0, "max": 20.0, "step": 0.1,
  "ui_control": "input", "ui_order": 30, "ui_description": "Classifier-free guidance strength",
  "wire_to": { "node_id": 3, "input_name": "cfg" }
}
```

- widgets_values: `[name, value, minimum, maximum, step, ui_control, ui_order, ui_description]`
- Output: FLOAT (slot 0)

### StimmaStringParam
String parameter — free text (textarea) or static dropdown (with enum_values).
**Use this for negative prompts** — it's an ordinary named string parameter, distinct from the
primary prompt (which `StimmaPromptParam` owns under the canonical `prompt` name).

```json
{
  "type": "string", "name": "negative_prompt",
  "value": "blurry, low quality, watermark",
  "ui_control": "textarea", "ui_order": 1,
  "ui_description": "What to avoid in the generated output",
  "wire_to": { "node_id": 6, "input_name": "text" }
}
```

- `ui_control`: "textarea" (free text, default) or "dropdown" (with enum_values)
- `enum_values`: optional, newline-separated list of allowed values (only for dropdown mode)
- widgets_values: `[name, value, enum_values, ui_control, ui_order, ui_description]`
- Output: STRING (slot 0)
- **Key distinction**: every Stimma node lands in the tool's single `parameter_schema`. Use StimmaStringParam for ordinary config text (negative prompts, style presets); reserve StimmaPromptParam for the one primary positive prompt (it owns the canonical `prompt` name).

### StimmaDropdownParam
Auto-resolving dropdown. Connects to COMBO inputs on ComfyUI nodes. At discovery time, the STP server reads the target node's object_info spec to populate the enum.

```json
{
  "type": "dropdown", "name": "sampler", "value": "euler",
  "ui_order": 40, "ui_description": "",
  "wire_to": { "node_id": 3, "input_name": "sampler_name", "link_type": "STRING" }
}
```

- The `link_type` in `wire_to` should typically be "STRING" since the output type is "*"
- For KSampler inputs: sampler_name, scheduler
- For KSamplerSelect: sampler_name
- For BasicScheduler: scheduler
- widgets_values: `[name, value, ui_order, ui_description]`
- Output: * / STRING (slot 0)

### StimmaBoolParam
Boolean toggle.

```json
{
  "type": "bool", "name": "enable_hires", "value": false,
  "ui_order": 50, "ui_description": "Enable high-resolution pass",
  "wire_to": { "node_id": 10, "input_name": "enable" }
}
```

- widgets_values: `[name, value, ui_order, ui_description]`
- Output: BOOLEAN (slot 0)

### StimmaImageParam
Single image upload (for img2img, reference images, etc.).

```json
{
  "type": "image", "name": "start_image",
  "required": true, "ui_order": 2,
  "wire_to": {"node_id": 92, "input_name": "image"}
}
```

- `wire_to`: Optional. Wires IMAGE output (slot 0) to the target node. Supports subgraph inputs.
- widgets_values: `[name, image_filename, required, ui_order]`
- Outputs: IMAGE (slot 0), MASK (slot 1)

### StimmaCheckpointLoader
Wraps `CheckpointLoaderSimple` with a `path_filter`, so the user picks the checkpoint as a
tool parameter. Outputs MODEL, CLIP, VAE (slots 0/1/2). Use it to replace an existing
checkpoint loader in the workflow.

Plan section (`checkpoint_loader`, not part of `params`):
```json
{
  "path_filter": "flux/**",
  "replace_node_id": 4,
  "default": "flux1-dev.safetensors",
  "ui_order": 50
}
```

- `replace_node_id`: ID of an existing `CheckpointLoaderSimple` to replace. All consumers of its
  MODEL/CLIP/VAE outputs are rewired to the StimmaCheckpointLoader and the old node is deleted.
  Omit to add a standalone loader.
- `path_filter`: fnmatch pattern to filter the checkpoint list (empty = show all).
- `default`: Optional default `ckpt_name`.
- widgets_values: `[ckpt_name, path_filter, ui_order]`
- Inputs: ckpt_name (COMBO widget). Outputs: model MODEL (0), clip CLIP (1), vae VAE (2)
- Insert the LoRA loader (below) *after* this node when both are present.

### StimmaLoraLoader
Always insert right after the model/checkpoint loader. Intercepts MODEL and CLIP chains.

```json
{
  "path_filter": "",
  "ui_order": 50,
  "model_source": { "node_id": 3, "slot": 0 },
  "clip_source": { "node_id": 2, "slot": 0 }
}
```

- `model_source`/`clip_source`: The node+slot that currently output MODEL/CLIP. The script intercepts these outputs and routes them through the LoRA loader.
- For CheckpointLoaderSimple: model_source slot=0 (MODEL), clip_source same node slot=1 (CLIP)
- For UNETLoader + DualCLIPLoader: model_source={UNETLoader, slot 0}, clip_source={DualCLIPLoader, slot 0}
- `path_filter`: fnmatch pattern to filter LoRA list. Empty = show all. Convention: `model-family/**`
- widgets_values: `[path_filter, ui_order, lora_1, strength_1, ..., lora_10, strength_10]`
- Inputs: model MODEL (slot 0), clip CLIP (slot 1), then widgets
- Outputs: model MODEL (slot 0), clip CLIP (slot 1)

### StimmaDurationToFrames
Converts duration (seconds) + fps → frame count, snapped to valid frame steps.
Used for video workflows instead of exposing raw frame count.

Plan entry (as a deferred input — processed after params so refs resolve):
```json
{
  "type": "duration_to_frames", "name": "duration_to_frames",
  "frame_step": 8,
  "duration_source": {"ref": "duration"},
  "fps_source": {"ref": "fps"},
  "wire_to": {"node_id": 92, "input_name": "value"}
}
```

- `frame_step`: Frame alignment for the model (4 for Wan, 8 for LTX2). Computes `frame_step * round((duration*fps - 1) / frame_step) + 1`.
- `duration_source`/`fps_source`: Refs to param nodes by name (created in the params section).
- Wire its output (INT slot 0) to the latent video node's length/frames input.
- Inputs: duration FLOAT (link), fps INT (link), frame_step INT (widget)
- Output: frames INT (slot 0)
- **widget_values gotcha**: Connected widget-type inputs (FLOAT, INT) still consume `widgets_values` slots. The widget_values must be `[0.0, 25, frame_step]` — placeholders for duration and fps, then the actual frame_step value.
- **Always use this for video** instead of exposing raw frame_count. Users think in seconds, not frames.

### StimmaPairedLoraLoader
For dual-model architectures (e.g. Wan 2.2 high/low noise). Takes two MODEL inputs, applies paired LoRAs.

```json
{
  "high_noise_model": ["high_noise_model_node", 0],
  "low_noise_model": ["low_noise_model_node", 0],
  "path_filter": "*t2v*;*T2V*",
  "ui_order": 50
}
```

### StimmaImageOutput / StimmaVideoOutput
Replace SaveImage/SaveVideo nodes.

Simple replacement (auto-detects source from the save node):
```json
{ "replace_node_id": 15, "output_type": "image" }
```

Direct wiring (for subgraph outputs or custom sources):
```json
{
  "output_type": "video",
  "wire_from": {"node_id": 92, "output_name": "frames"},
  "fps_source": {"ref": "fps"},
  "mute_nodes": [75]
}
```

- `replace_node_id`: Mutes the save node and wires from its source. Use when a SaveImage/SaveVideo exists.
- `wire_from`: Direct source. Use `output_name` to reference a named output (e.g., subgraph outputs added via `subgraph_prep`). Or use `src_slot` for positional.
- `fps_source`: For video outputs, ref to the fps param node. Required for correct video encoding.
- `mute_nodes`: Additional nodes to mute (e.g., original SaveVideo that isn't being replaced via replace_node_id).

### StimmaLayoutGroup
Groups parameters in the Stimma UI.

```json
{
  "label": "Advanced",
  "param_names": ["cfg_scale", "sampler", "seed"],
  "collapsed": true, "ui_order": 10
}
```

Per-param layout overrides using dicts:
```json
{
  "label": "Prompt Settings",
  "param_names": [{"name": "negative_prompt", "full_width": true}],
  "collapsed": false, "ui_order": 0
}
```

- `param_names`: List of strings or dicts. Strings are param names. Dicts support `{"name": "...", "full_width": true}`.
- `full_width`: Makes the parameter span all columns in the Stimma UI. Use for textareas (negative_prompt).
- Encoded in widget text as `name !full_width` (parsed by tool_builder).
- Parameters NOT in any group appear at the top level.
- widgets_values: `[group_label, param_names_newline_separated, collapsed, ui_order]`

## ui_order Conventions

| Range | Content |
|-------|---------|
| 0 | Main prompt |
| 1-4 | Resolution, reference images |
| 5-9 | Additional inputs (negative prompt, masks) |
| 10-19 | Layout groups |
| 20-39 | Key params (steps, cfg) |
| 40-59 | Sampler/scheduler, LoRA |
| 60-79 | Advanced params (denoise, shift, guidance) |
| 80-99 | Seed (always last) |
