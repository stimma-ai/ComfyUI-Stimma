"""Tool builder — converts DiscoveredWorkflow objects into STP Tool objects."""

import logging
import re
from typing import Dict, Any, List, Optional, TYPE_CHECKING

from stimma_tools_protocol.tool import Tool, ToolParameter, Group, Param

from .discovery import DiscoveredWorkflow, _resolve_model_combo_value

if TYPE_CHECKING:
    from .config import Config
    from .provider import StimmaPluginProvider

logger = logging.getLogger(__name__)


# Normalize legacy camelCase x-control values stored in older saved workflows to snake_case.
_CONTROL_ALIASES = {
    "promptEditor": "prompt_editor",
    "imagePicker": "image_picker",
    "videoPicker": "video_picker",
    "videoFramePicker": "video_frame_picker",
    "maskEditor": "mask_editor",
    "loraPicker": "lora_picker",
}


def _norm_control(control: Optional[str]) -> Optional[str]:
    return _CONTROL_ALIASES.get(control, control)


# Every control name a media param may legitimately carry, post-alias.
_KNOWN_MEDIA_CONTROLS = {
    "image_picker", "video_frame_picker", "video_picker", "audio_picker",
}


def _resolve_media_control(inputs: dict, default: str) -> str:
    """Resolve a media param's `ui_control`, tolerating stale widget alignment.

    A workflow saved against an older node signature has fewer entries in
    `widgets_values` than the node now declares, so positional mapping shifts
    every later widget — `ui_control` then comes back as an int (the ui_order
    slot) instead of a control name. The host keys media handling off
    `x-control`, so a shifted value silently turns an image input into a plain
    string param: its path is never uploaded as an asset, and a remote provider
    receives a local filesystem path it cannot open.

    Recover by taking the declared value when it is a real control name, else
    scanning the node's other widget values for one, else the default.
    """
    control = _norm_control(inputs.get("ui_control"))
    if control in _KNOWN_MEDIA_CONTROLS:
        return control
    for value in inputs.values():
        if isinstance(value, str):
            candidate = _norm_control(value)
            if candidate in _KNOWN_MEDIA_CONTROLS:
                logger.warning(
                    "ui_control misaligned (got %r); recovered %r from widget values — "
                    "re-save this workflow in ComfyUI to fix it at the source",
                    inputs.get("ui_control"), candidate,
                )
                return candidate
    logger.warning(
        "ui_control missing/invalid (got %r); defaulting to %r",
        inputs.get("ui_control"), default,
    )
    return default


def _match_lora_filter(name: str, pattern: str) -> bool:
    """Match a LoRA path against a filter pattern.

    Supports semicolon-delimited multiple patterns, * (single directory level),
    and ** (recursive across directories).
    """
    # ComfyUI exposes nested model names with the host-native separator. Keep
    # workflows portable by treating both slash styles as directory separators.
    name = name.replace("\\", "/")
    for part in pattern.split(";"):
        part = part.strip().replace("\\", "/")
        if not part:
            continue
        regex = ""
        i = 0
        while i < len(part):
            if part[i] == "*":
                if i + 1 < len(part) and part[i + 1] == "*":
                    regex += ".*"
                    i += 2
                    if i < len(part) and part[i] == "/":
                        i += 1
                else:
                    regex += "[^/]*"
                    i += 1
            elif part[i] == "?":
                regex += "[^/]"
                i += 1
            else:
                regex += re.escape(part[i])
                i += 1
        if re.fullmatch(regex, name):
            return True
    return False


def _extract_combo_values(spec) -> Optional[List[str]]:
    """Extract enum values from a ComfyUI input spec.

    Handles both formats:
    - Old: [["euler", "euler_ancestral", ...], {}]
    - New: ["COMBO", {"options": ["euler", ...]}]
    """
    if not isinstance(spec, (list, tuple)) or len(spec) == 0:
        return None
    # Old format: spec[0] is the list of values
    if isinstance(spec[0], list):
        return spec[0]
    # New format: spec[0] is "COMBO", spec[1] has "options"
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        options = spec[1].get("options")
        if isinstance(options, list):
            return options
    return None


def _parse_controlnet_types(raw: Any) -> Optional[List[str]]:
    """Parse controlnet types from comma/newline-delimited text."""
    if not isinstance(raw, str):
        return None
    parts = [p.strip() for p in re.split(r"[\n,]+", raw) if p.strip()]
    if not parts:
        return None
    seen = set()
    out = []
    for item in parts:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _get_lora_list(object_info: Optional[Dict[str, Any]]) -> List[str]:
    """Extract the list of available LoRA names from object_info."""
    if not object_info:
        return []
    lora_loader = object_info.get("LoraLoader", {})
    input_def = lora_loader.get("input", {})
    required = input_def.get("required", {})
    lora_name_spec = required.get("lora_name", [])
    values = _extract_combo_values(lora_name_spec)
    return values if values else []


def _get_checkpoint_list(object_info: Optional[Dict[str, Any]]) -> List[str]:
    """Extract the list of available checkpoint names from object_info."""
    if not object_info:
        return []
    ckpt_loader = object_info.get("CheckpointLoaderSimple", {})
    input_def = ckpt_loader.get("input", {})
    required = input_def.get("required", {})
    ckpt_name_spec = required.get("ckpt_name", [])
    values = _extract_combo_values(ckpt_name_spec)
    return values if values else []


def _build_checkpoint_parameter(
    node: Dict[str, Any],
    object_info: Optional[Dict[str, Any]],
) -> Optional[ToolParameter]:
    """Build a ToolParameter for checkpoint selection."""
    inputs = node["inputs"]
    name = inputs.get("name", "checkpoint")
    path_filter = inputs.get("path_filter", "")
    default_ckpt = inputs.get("ckpt_name", "")

    available = _get_checkpoint_list(object_info)
    if path_filter:
        available = [c for c in available if _match_lora_filter(c, path_filter)]

    resolved_default, _ = _resolve_model_combo_value(default_ckpt, available)
    if resolved_default is not None:
        default_ckpt = resolved_default

    return ToolParameter(
        name=name,
        type="string",
        description="Model checkpoint to use",
        required=False,
        default=default_ckpt,
        enum=available,
        ui_hints={
            "control": "dropdown",
            "format": "filename",
        },
    )


def _build_field_parameter(node: Dict[str, Any]) -> Optional[ToolParameter]:
    """Build a ToolParameter from a Stimma field node."""
    class_type = node["class_type"]
    inputs = node["inputs"]
    name = inputs.get("name", "")

    if class_type == "StimmaPromptParam":
        return ToolParameter(
            name=name,
            type="string",
            description=inputs.get("ui_description", ""),
            required=inputs.get("required", True),
            default=inputs.get("default_text", ""),
            ui_hints={"control": "prompt_editor"},
        )
    elif class_type == "StimmaImageParam":
        required = bool(inputs.get("required", True))
        control = _resolve_media_control(inputs, "image_picker")
        controlnet_types = _parse_controlnet_types(inputs.get("controlnet_types", ""))
        ui_hints = {
            "control": control,
            "min-items": 1 if required else 0,
            "max-items": 1,
        }
        if controlnet_types:
            ui_hints["controlnet"] = controlnet_types
        # Surface the prep controls (Scale / Extend Canvas / Paint). Defaults on
        # for image inputs; the frontend also enables prep when controlnet is set.
        if inputs.get("allow_prep", True):
            ui_hints["allow-prep"] = True
        return ToolParameter(
            name="input_images",
            type="array",
            description="",
            required=required,
            items={"type": "string"},
            ui_hints=ui_hints,
        )
    elif class_type == "StimmaMaskParam":
        source = inputs.get("source_image_field", "")
        return ToolParameter(
            name=name,
            type="string",
            description=f"Asset ID for mask (source: {source})",
            required=True,
            ui_hints={
                "control": "mask_editor",
                "source_image_field": source,
            },
        )
    elif class_type == "StimmaImagesParam":
        min_images = inputs.get("min_images", 1)
        max_images = inputs.get("max_images", 3)
        control = _resolve_media_control(inputs, "image_picker")
        controlnet_types = _parse_controlnet_types(inputs.get("controlnet_types", ""))
        ui_hints = {
            "control": control,
            "min-items": min_images,
            "max-items": max_images,
        }
        if controlnet_types:
            ui_hints["controlnet"] = controlnet_types
        if inputs.get("allow_prep", True):
            ui_hints["allow-prep"] = True
        return ToolParameter(
            name="input_images",
            type="array",
            description="",
            required=int(min_images) > 0,
            items={"type": "string"},
            ui_hints=ui_hints,
        )
    elif class_type == "StimmaVideoParam":
        control = _norm_control(inputs.get("ui_control", "video_picker"))
        required = bool(inputs.get("required", True))
        return ToolParameter(
            name="input_videos",
            type="array",
            description="",
            required=required,
            items={"type": "string"},
            ui_hints={
                "control": control,
                "min-items": 1 if required else 0,
                "max-items": 1,
            },
        )
    elif class_type == "StimmaVideosParam":
        min_videos = inputs.get("min_videos", 1)
        max_videos = inputs.get("max_videos", 3)
        control = _norm_control(inputs.get("ui_control", "video_picker"))
        return ToolParameter(
            name="input_videos",
            type="array",
            description="",
            required=int(min_videos) > 0,
            items={"type": "string"},
            ui_hints={
                "control": control,
                "min-items": min_videos,
                "max-items": max_videos,
            },
        )
    elif class_type == "StimmaAudioParam":
        control = _norm_control(inputs.get("ui_control", "audio_picker"))
        required = bool(inputs.get("required", True))
        return ToolParameter(
            name="input_audios",
            type="array",
            description="",
            required=required,
            items={"type": "string"},
            ui_hints={
                "control": control,
                "min-items": 1 if required else 0,
                "max-items": 1,
                "label": inputs.get("ui_label") or "Audio",
                "audio-role": inputs.get("audio_role") or "driving",
                # ComfyUI decodes audio through av, so we accept the common
                # containers directly and only ask the host to convert what it
                # can't. WAV is the lossless target, so an mp3 source is never
                # re-compressed a second time.
                "accept-media": {
                    "mime_types": [
                        "audio/wav", "audio/mpeg", "audio/mp4", "audio/flac", "audio/ogg",
                    ],
                    "transcode_to": "audio/wav",
                },
            },
        )
    elif class_type == "StimmaSeedParam":
        return ToolParameter(
            name=name,
            type="integer",
            description="Random seed",
            required=False,
            ui_hints={"control": "seed"},
        )
    elif class_type == "StimmaResolutionParam":
        step = inputs.get("step", 64)
        min_size = inputs.get("min_size", 512)
        max_size = inputs.get("max_size", 4096)

        # Parse supported resolutions: one "WxH" per line
        supported = None
        res_text = inputs.get("supported_resolutions", "")
        if res_text and res_text.strip():
            supported = []
            for line in res_text.strip().split("\n"):
                line = line.strip()
                if "x" in line:
                    parts = line.split("x", 1)
                    try:
                        supported.append([int(parts[0]), int(parts[1])])
                    except ValueError:
                        pass

        width_hints = {"control": "resolution", "step": step, "paired-with": "height"}
        if supported:
            width_hints["supported_resolutions"] = supported

        return [
            ToolParameter(
                name="width",
                type="integer",
                description="Image width in pixels",
                required=False,
                default=inputs.get("width", 1024),
                minimum=min_size,
                maximum=max_size,
                ui_hints=width_hints,
            ),
            ToolParameter(
                name="height",
                type="integer",
                description="Image height in pixels",
                required=False,
                default=inputs.get("height", 1024),
                minimum=min_size,
                maximum=max_size,
                ui_hints={"control": "resolution", "step": step, "paired-with": "width"},
            ),
        ]

    logger.warning(f"Unknown input node type: {class_type}")
    return None


def _build_param_parameter(
    node: Dict[str, Any],
    object_info: Optional[Dict[str, Any]],
) -> Optional[ToolParameter]:
    """Build a ToolParameter from a Stimma parameter node."""
    class_type = node["class_type"]
    inputs = node["inputs"]
    name = inputs.get("name", "")

    _ACRONYMS = {"cfg"}
    # A trailing _pct marks the unit for the wire, where a bare `expand_top`
    # would be ambiguous. The label is not the wire — it sits next to a value
    # already rendered as "25%", so repeating it reads as "Expand Top Pct".
    label_source = name[:-4] if name.endswith("_pct") else name
    label = " ".join(
        w.upper() if w.lower() in _ACRONYMS else w.capitalize()
        for w in label_source.replace("_", " ").split()
    )

    if class_type == "StimmaIntParam":
        ui_control = inputs.get("ui_control", "input")
        # A bare number whose unit the user has to guess is a broken control, so
        # the format hint rides through to `x-format` and the host renders "25%".
        ui_format = inputs.get("ui_format") or ""
        return ToolParameter(
            name=name,
            type="integer",
            description=inputs.get("ui_description", ""),
            required=False,
            default=inputs.get("value", 0),
            minimum=inputs.get("minimum"),
            maximum=inputs.get("maximum"),
            ui_hints={
                "control": ui_control,
                "step": inputs.get("step", 1),
                "label": label,
                **({"format": ui_format} if ui_format else {}),
            },
        )
    elif class_type == "StimmaFloatParam":
        ui_control = inputs.get("ui_control", "input")
        return ToolParameter(
            name=name,
            type="number",
            description=inputs.get("ui_description", ""),
            required=False,
            default=inputs.get("value", 0.0),
            minimum=inputs.get("minimum"),
            maximum=inputs.get("maximum"),
            ui_hints={
                "control": ui_control,
                "step": inputs.get("step", 0.1),
                "label": label,
            },
        )
    elif class_type == "StimmaStringParam":
        ui_control = inputs.get("ui_control", "dropdown")
        enum_text = inputs.get("enum_values", "")
        enum_values = None

        if enum_text and enum_text.strip():
            enum_values = [v.strip() for v in enum_text.strip().split("\n") if v.strip()]

        param = ToolParameter(
            name=name,
            type="string",
            description=inputs.get("ui_description", ""),
            required=False,
            default=inputs.get("value", ""),
            enum=enum_values,
            ui_hints={
                "control": ui_control if enum_values else "textarea",
                "label": label,
            },
        )
        return param
    elif class_type == "StimmaDropdownParam":
        # Resolve enum from the connected ComfyUI node's object_info spec
        enum_values = None
        target_ct = node.get("target_class_type")
        target_inp = node.get("target_input_name")
        if target_ct and target_inp and object_info:
            node_info = object_info.get(target_ct, {})
            input_def = node_info.get("input", {})
            for category in ["required", "optional"]:
                if category in input_def and target_inp in input_def[category]:
                    values = _extract_combo_values(input_def[category][target_inp])
                    if values:
                        enum_values = values
                    break
        if not enum_values:
            logger.warning(
                f"StimmaDropdownParam '{name}': could not resolve enum "
                f"(target={target_ct}.{target_inp})"
            )
        return ToolParameter(
            name=name,
            type="string",
            description=inputs.get("ui_description", ""),
            required=False,
            default=inputs.get("value", ""),
            enum=enum_values,
            ui_hints={
                "control": "dropdown",
                "label": label,
            },
        )
    elif class_type == "StimmaBoolParam":
        return ToolParameter(
            name=name,
            type="boolean",
            description=inputs.get("ui_description", ""),
            required=False,
            default=inputs.get("value", False),
            ui_hints={"label": label},
        )

    logger.warning(f"Unknown param node type: {class_type}")
    return None


_LORA_PAIR_PATTERNS = [
    ("_HIGH_", "_LOW_"),
    ("_high_noise_", "_low_noise_"),
    ("_highnoise_", "_lownoise_"),
    ("_hi_", "_lo_"),
    # Also match at end of stem (before extension)
    ("_HIGH", "_LOW"),
    ("_high_noise", "_low_noise"),
    ("_highnoise", "_lownoise"),
    ("_hi", "_lo"),
]


def _find_lora_pair(name: str, all_loras: List[str]) -> Optional[dict]:
    """Find the paired LoRA for a given filename.

    Returns {"display": "<base>", "high": "<high_file>", "low": "<low_file>"}
    or None if no pair found. Only returns a pair when `name` is the HIGH variant.
    """
    all_set = set(all_loras)
    for hi_pat, lo_pat in _LORA_PAIR_PATTERNS:
        if hi_pat in name:
            lo_name = name.replace(hi_pat, lo_pat, 1)
            if lo_name in all_set:
                # Build display name: remove the HIGH discriminator
                display = name.replace(hi_pat, "_", 1)
                return {"display": display, "high": name, "low": lo_name}
        # Check case-insensitive match
        name_lower = name.lower()
        hi_lower = hi_pat.lower()
        if hi_lower in name_lower:
            idx = name_lower.index(hi_lower)
            lo_name = name[:idx] + lo_pat + name[idx + len(hi_pat):]
            if lo_name in all_set and lo_name != name:
                display = name[:idx] + "_" + name[idx + len(hi_pat):]
                return {"display": display, "high": name, "low": lo_name}
    return None


def _get_paired_lora_list(
    object_info: Optional[Dict[str, Any]],
    path_filter: str,
) -> List[dict]:
    """Build a list of paired LoRAs matching path_filter.

    Returns list of {"display": "<base_name>", "high": "<high_file>", "low": "<low_file>"}.
    """
    all_loras = _get_lora_list(object_info)
    if path_filter:
        filtered = [l for l in all_loras if _match_lora_filter(l, path_filter)]
    else:
        filtered = list(all_loras)

    pairs = []
    seen_highs = set()
    for name in filtered:
        pair = _find_lora_pair(name, all_loras)
        if pair and pair["high"] not in seen_highs:
            seen_highs.add(pair["high"])
            pairs.append(pair)
    return pairs


def _build_lora_parameter(
    node: Dict[str, Any],
    object_info: Optional[Dict[str, Any]],
) -> Optional[ToolParameter]:
    """Build a ToolParameter for LoRA selection."""
    class_type = node["class_type"]
    inputs = node["inputs"]

    if class_type == "StimmaPairedLoraLoader":
        return _build_paired_lora_parameter(node, object_info)

    if class_type != "StimmaLoraLoader":
        logger.warning(f"Unknown lora node type: {class_type}")
        return None

    path_filter = inputs.get("path_filter", "")
    available = _get_lora_list(object_info)
    if path_filter:
        available = [l for l in available if _match_lora_filter(l, path_filter)]

    # Extract defaults from lora_1..lora_N slots
    defaults = []
    for i in range(1, 11):
        name = inputs.get(f"lora_{i}", "None")
        weight = inputs.get(f"strength_{i}", 1.0)
        if name != "None":
            resolved_name, _ = _resolve_model_combo_value(name, available)
            if resolved_name is not None:
                name = resolved_name
            defaults.append({"name": name, "weight": weight})

    return ToolParameter(
        name="loras",
        type="array",
        description="LoRAs to apply",
        required=False,
        default=defaults,
        items={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "enum": available,
                    "x-accept-upload": {
                        "extensions": [".safetensors"],
                        "max_size": 2147483648,  # 2 GB
                    },
                },
                "name": {"type": "string", "enum": available},
                "weight": {"type": "number", "default": 1.0, "minimum": -2.0, "maximum": 2.0},
            },
            "required": ["path"],
        },
        ui_hints={"control": "lora_picker"},
    )


def _build_paired_lora_parameter(
    node: Dict[str, Any],
    object_info: Optional[Dict[str, Any]],
) -> Optional[ToolParameter]:
    """Build a ToolParameter for paired LoRA selection (Wan 2.2 dual-model)."""
    inputs = node["inputs"]
    path_filter = inputs.get("path_filter", "")
    pairs = _get_paired_lora_list(object_info, path_filter)

    display_names = [p["display"] for p in pairs]

    return ToolParameter(
        name="loras",
        type="array",
        description="Paired LoRAs to apply (high + low noise variants)",
        required=False,
        default=[],
        items={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "enum": display_names,
                    "x-accept-upload": {
                        "extensions": [".safetensors"],
                        "max_size": 2147483648,  # 2 GB
                    },
                },
                "name": {"type": "string", "enum": display_names},
                "weight": {"type": "number", "default": 1.0, "minimum": -2.0, "maximum": 2.0},
            },
            "required": ["path"],
            "x-paired": True,
            "x-pairs": pairs,
        },
        ui_hints={"control": "lora_picker"},
    )


def _build_layout(
    layout_nodes: List[Dict[str, Any]],
    param_nodes: List[Dict[str, Any]],
    lora_nodes: List[Dict[str, Any]],
) -> Optional[List[Group]]:
    """Build layout groups from StimmaLayoutGroup nodes."""
    if not layout_nodes:
        return None

    groups = []
    for node in layout_nodes:
        inputs = node["inputs"]
        label = inputs.get("group_label", "")
        param_names_text = inputs.get("param_names", "")
        collapsed = inputs.get("collapsed", False)

        params = []
        for line in param_names_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            full_width = False
            if " !full_width" in line:
                line = line.replace(" !full_width", "")
                full_width = True
            params.append(Param(name=line, full_width=full_width))

        groups.append(Group(label=label, params=params, collapsed=collapsed))

    return groups if groups else None


def _merge_media_parameters(params: List[ToolParameter]) -> List[ToolParameter]:
    """Merge duplicate canonical media inputs from multiple nodes."""
    grouped: Dict[str, List[ToolParameter]] = {}
    passthrough: List[ToolParameter] = []

    for p in params:
        if p.name in ("input_images", "input_videos", "input_audios"):
            grouped.setdefault(p.name, []).append(p)
        else:
            passthrough.append(p)

    merged: List[ToolParameter] = []
    for name in ("input_images", "input_videos", "input_audios"):
        items = grouped.get(name, [])
        if not items:
            continue
        if len(items) == 1:
            merged.append(items[0])
            continue

        mins = []
        maxs = []
        controls = []
        controlnet_types = []
        allow_prep = False
        accept_media = None
        label = None
        audio_role = None
        for p in items:
            hints = p.ui_hints or {}
            audio_role = audio_role or hints.get("audio-role")
            mins.append(int(hints.get("min-items", 1)))
            maxs.append(int(hints.get("max-items", 1)))
            controls.append(hints.get("control"))
            allow_prep = allow_prep or bool(hints.get("allow-prep"))
            accept_media = accept_media or hints.get("accept-media")
            label = label or hints.get("label")
            for ctype in hints.get("controlnet", []) or []:
                if ctype not in controlnet_types:
                    controlnet_types.append(ctype)

        if name == "input_images":
            control = "video_frame_picker" if "video_frame_picker" in controls else "image_picker"
        elif name == "input_audios":
            control = "audio_picker"
        else:
            control = "video_picker"

        # Images are a single multi-upload field ("1 primary + optional extras"),
        # so min is max(mins). Video params are distinct positional slots; a stitch
        # needs at least two clips but accepts up to the number of slots, so min is 2
        # and max is the slot count (sum of per-node maxes). Audio params are
        # positional slots too (a driving track and a voice reference are different
        # inputs), but every declared slot is meant, so min is the sum.
        if name in ("input_videos", "input_audios"):
            min_items = sum(mins)
        else:
            min_items = max(mins)

        merged.append(
            ToolParameter(
                name=name,
                type="array",
                description="",
                required=min_items > 0,
                items={"type": "string"},
                ui_hints={
                    "control": control,
                    "min-items": min_items,
                    "max-items": sum(maxs),
                    **({"controlnet": controlnet_types} if controlnet_types and name == "input_images" else {}),
                    **({"allow-prep": True} if allow_prep else {}),
                    **({"accept-media": accept_media} if accept_media else {}),
                    **({"label": label} if label else {}),
                    **({"audio-role": audio_role} if audio_role else {}),
                },
            )
        )

    return passthrough + merged


def build_tools_from_workflows(
    workflows: List[DiscoveredWorkflow],
    object_info: Optional[Dict[str, Any]],
    config: "Config",
    provider: "StimmaPluginProvider",
) -> List[Tool]:
    """Convert a list of DiscoveredWorkflows into STP Tool objects."""
    tools = []

    for workflow in workflows:
        try:
            tool = _build_single_tool(workflow, object_info, config, provider)
            if tool:
                tools.append(tool)
        except Exception as e:
            logger.error(
                f"Failed to build tool from {workflow.file_path}: {e}",
                exc_info=True,
            )

    return tools


def _build_single_tool(
    workflow: DiscoveredWorkflow,
    object_info: Optional[Dict[str, Any]],
    config: "Config",
    provider: "StimmaPluginProvider",
) -> Optional[Tool]:
    """Build a single Tool from a DiscoveredWorkflow."""
    info = workflow.tool_info
    slug = info["slug"]

    if not slug:
        return None

    # Build media/prompt/seed/resolution parameters
    media_parameters = []
    for node in workflow.field_nodes:
        result = _build_field_parameter(node)
        if result:
            if isinstance(result, list):
                media_parameters.extend(result)
            else:
                media_parameters.append(result)
    media_parameters = _merge_media_parameters(media_parameters)
    if "reference-to-video" in (info.get("task_types", []) or []):
        for parameter in media_parameters:
            hints = parameter.ui_hints or {}
            if parameter.name == "input_images":
                hints.update({
                    "label": "Reference Images",
                })
                parameter.description = (
                    "Shown to the model as <Picture 1>, <Picture 2>, and so on, in this order."
                )
            elif parameter.name == "input_videos":
                hints.update({
                    "label": "Reference Videos",
                })
                parameter.description = (
                    "Each soundtrack is <Audio N> immediately before its matching <Video N>; "
                    "reordering this list changes both tags together."
                )
            elif parameter.name == "input_audios":
                hints.update({
                    "label": "Standalone Reference Audio",
                    "audio-role": "reference",
                })
                parameter.description = (
                    "Numbered after the reference-video soundtracks as <Audio N>."
                )
            parameter.ui_hints = hints
    # Build configurable parameters
    parameters = []
    for node in workflow.param_nodes:
        param = _build_param_parameter(node, object_info)
        if param:
            parameters.append(param)

    # Build checkpoint parameters
    for node in workflow.checkpoint_nodes:
        param = _build_checkpoint_parameter(node, object_info)
        if param:
            parameters.append(param)

    # Build LoRA parameters
    for node in workflow.lora_nodes:
        param = _build_lora_parameter(node, object_info)
        if param:
            parameters.append(param)

    # Build layout
    layout = _build_layout(workflow.layout_nodes, workflow.param_nodes, workflow.lora_nodes)

    # All parameters share one namespace: media/prompt/seed/resolution first, then configurable params.
    all_parameters = media_parameters + parameters

    # Create execution function closure
    async def execute_fn(context, params, _wf=workflow, _provider=provider):
        from .executor import execute_workflow
        return await execute_workflow(context, _wf, params, _provider)

    metadata = {}
    badges = info.get("badges", []) or []
    if badges:
        metadata["badges"] = badges

    task_types = info.get("task_types", []) or []
    tool = Tool(
        slug=slug,
        display_name=info["display_name"],
        task_type=task_types[0] if task_types else "other",
        function=execute_fn,
        description=info.get("description", ""),
        parameters=all_parameters,
        layout=layout,
        task_types=task_types,
        metadata=metadata,
        model_vendor=(info.get("model_vendor") or "").strip() or None,
        model=(info.get("model") or "").strip() or None,
        required_any=(
            [["input_images"], ["input_videos"], ["input_audios"]]
            if "reference-to-video" in task_types else None
        ),
    )

    if workflow.warnings:
        tool.status = "needs_setup"
        tool.status_items = [
            {
                "kind": (
                    "missing_node" if i.get("kind") == "missing_node"
                    else "missing_model" if i.get("kind") in ("missing_model", "missing_checkpoint")
                    else "other"
                ),
                "name": i.get("name", ""),
                "detail": i.get("folder") or i.get("class_type") or "",
            }
            for i in (workflow.issues or [])
        ] or [{"kind": "other", "name": w, "detail": ""} for w in workflow.warnings]

    return tool
