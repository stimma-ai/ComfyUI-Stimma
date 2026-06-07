# ComfyUI-Stimma

A ComfyUI plugin that exposes ComfyUI workflows as Stimma tools via the Stimma Tools Protocol (STP).

## What This Project Does

Users save ComfyUI workflows containing special Stimma nodes (fields, params, outputs, LoRAs, layout). This plugin discovers those workflows, builds tool descriptors from the Stimma nodes, and serves them over a JSON-RPC WebSocket so the Stimma UI can execute them remotely.

## Repository Layout

```
ComfyUI-Stimma/
  __init__.py          # Entry point — exports NODE_CLASS_MAPPINGS, hooks STP server into ComfyUI
  nodes/               # Custom ComfyUI node definitions
    tool_info.py       #   StimmaToolInfo — metadata (slug, task_type, display_name)
    fields.py          #   StimmaPromptParam, StimmaImageParam, StimmaVideoParam, etc.
    params.py          #   StimmaIntParam, StimmaFloatParam, StimmaStringParam, etc.
    loras.py           #   StimmaLoraLoader, StimmaPairedLoraLoader
    outputs.py         #   StimmaImageOutput, StimmaVideoOutput
    layout.py          #   StimmaLayoutGroup
  stp_server/          # STP server implementation
    config.py          #   Config dataclasses, YAML loading
    startup.py         #   Hooks routes into ComfyUI's aiohttp app
    transport.py       #   ComfyUITransport — wraps ComfyUI's existing aiohttp server
    provider.py        #   StimmaPluginProvider — subclass of stimma_tools_protocol.Provider
    discovery.py       #   Scans workflow JSON files for Stimma nodes
    tool_builder.py    #   Converts DiscoveredWorkflow → Tool objects
    executor.py        #   Injects params/fields/loras into workflow, queues on ComfyUI, captures output
    comfy_client.py    #   HTTP client for ComfyUI API (/prompt, /object_info, /upload/image)
    workflow_install.py #  Syncs bundled workflows to ComfyUI's user directory
  stimma_tools_protocol/        # EMBEDDED copy of the STP framework (see below)
  workflows/           # Bundled workflow JSON files
  tests/               # Tests (run with: python tests/test_subgraph_expansion.py)
  config.yaml.default  # Example config
```

## Key Concept: `stimma_tools_protocol` is EMBEDDED

The `stimma_tools_protocol/` directory at the repo root is a **vendored copy** of the STP framework library. It is NOT imported from `../stimma-tools-python/` at runtime. The `__init__.py` adds the plugin directory to `sys.path` so `import stimma_tools_protocol` resolves to this local copy.

When you see imports like `from stimma_tools_protocol.provider import Provider` or `from stimma_tools_protocol.protocol import JsonRpcResponse`, those resolve to files **in this repo** at `stimma_tools_protocol/`. Do not reference or modify `../stimma-tools-python/` unless explicitly asked.

## Protocol Reference

The Stimma Tools Protocol (STP) spec lives at `../stimma/docs/TOOLS_PROTOCOL.md`. Key points:

- JSON-RPC 2.0 over WebSocket at `/stp-v1`
- Provider registers → Stimma calls `tools.list` → `tools.execute` / `tools.cancel`
- `tools.refresh` re-scans workflows and LoRA lists
- `tools.upload` / `tools.upload_complete` for file uploads (e.g. LoRAs)
- Assets transferred out-of-band via HTTP endpoints, never embedded in JSON
- Progress via `tools.progress` notifications, results via `tools.result`

Protocol reference: `../stimma/docs/TOOLS_PROTOCOL.md`

## How It Works (end to end)

1. **Import time** (`__init__.py`): `setup_stp_server()` hooks WebSocket + asset routes into ComfyUI's aiohttp app
2. **Server start** (`startup.py`): `_run_provider()` creates `StimmaPluginProvider`, starts the event loop
3. **First `tools.list`** (`provider.py`): Triggers `discover_and_register_tools()` — fetches `/object_info`, scans workflow files, builds Tool objects
4. **Discovery** (`discovery.py`): Finds JSON files containing `StimmaToolInfo` nodes, extracts all Stimma node data, returns `DiscoveredWorkflow` objects
5. **Tool building** (`tool_builder.py`): Converts each `DiscoveredWorkflow` into a `Tool` with typed parameters, LoRA enums (filtered by `path_filter`), and layout groups
6. **Execution** (`executor.py`): Injects param values + input assets into the workflow prompt, resolves Stimma node references, strips unused optional chains, queues on ComfyUI via `/prompt`, monitors progress via websocket, captures output files
7. **File watcher** (`provider.py`): Polls workflow dirs every 5s, re-discovers on changes, sends `tools.changed` notification

## Node Types

| Node | Purpose | Key Inputs |
|------|---------|------------|
| `StimmaToolInfo` | Workflow metadata | `slug`, `display_name`, `task_type`, `task_types` |
| `StimmaPromptParam` | Text prompt | `name`, `default_text`, `required` |
| `StimmaImageParam` | Image upload | `name`, `required` |
| `StimmaVideoParam` | Video upload | `name`, `required` |
| `StimmaMaskParam` | Mask editor | `name`, `source_image_field` |
| `StimmaResolutionParam` | Width/height | `width`, `height`, `step`, `supported_resolutions` |
| `StimmaSeedParam` | Random seed | `name` |
| `StimmaIntParam` | Integer slider | `name`, `value`, `minimum`, `maximum`, `step` |
| `StimmaFloatParam` | Float slider | `name`, `value`, `minimum`, `maximum`, `step` |
| `StimmaStringParam` | String/dropdown | `name`, `value`, `enum_values` |
| `StimmaDropdownParam` | Auto-resolved dropdown | `name` (enum from connected node's object_info) |
| `StimmaBoolParam` | Boolean toggle | `name`, `value` |
| `StimmaLoraLoader` | LoRA selection (10 slots) | `path_filter`, `lora_1`..`lora_10`, `strength_1`..`strength_10` |
| `StimmaPairedLoraLoader` | Paired LoRA (high/low noise) | `path_filter` |
| `StimmaImageOutput` | Output image | `_stimma_output_dir` (hidden) |
| `StimmaVideoOutput` | Output video | `_stimma_output_dir` (hidden) |
| `StimmaLayoutGroup` | UI layout group | `group_label`, `param_names`, `collapsed` |

## Important Patterns

### LoRA Handling
Two node types: `StimmaLoraLoader` has 10 slots that the executor fills directly. `path_filter` uses glob patterns (`flux/**`, `wan/**;flux/**`) to filter the LoRA list from `/object_info`.

### Composite Node Expansion
`_convert_ui_to_api` in `executor.py` expands group nodes (`workflow>Name`) and subgraph nodes into their inner nodes before API conversion. PrimitiveNode/Reroute are virtual and get followed through.

### Optional Input Chain Stripping
When an optional input (e.g. `StimmaImageParam` with `required=False`) is not provided, the executor strips the entire downstream chain of required-input nodes, stopping at nodes that have the connection as optional.

### Dynamic Enums
`StimmaDropdownParam` auto-resolves its enum by tracing its output connection to the target ComfyUI node and looking up that input's COMBO spec in `/object_info`.

## Testing

```bash
python tests/test_subgraph_expansion.py   # 27 tests, needs live ComfyUI object_info
python tests/test_executor.py
```

## Development Notes

- ComfyUI's `folder_paths` module is only available at runtime inside ComfyUI — lazy-import it
- `/object_info` returns node type specs including input types, defaults, and COMBO enum lists
- The `/upload/image` endpoint uses the field name `"image"` for all file types including videos
- Muted nodes (mode=4) are skipped during prompt conversion, matching ComfyUI's frontend behavior
