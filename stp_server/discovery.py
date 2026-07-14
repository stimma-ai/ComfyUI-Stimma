"""Workflow discovery — scans directories for Stimma-enabled ComfyUI workflows."""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import Config

logger = logging.getLogger(__name__)

# Stimma node class types
STIMMA_TOOL_INFO = "StimmaToolInfo"
STIMMA_FIELD_TYPES = {
    "StimmaPromptParam", "StimmaImageParam", "StimmaMaskParam",
    "StimmaImagesParam", "StimmaVideoParam", "StimmaVideosParam", "StimmaSeedParam",
    "StimmaResolutionParam",
}
STIMMA_PARAM_TYPES = {
    "StimmaIntParam", "StimmaFloatParam", "StimmaStringParam",
    "StimmaDropdownParam", "StimmaBoolParam",
}
STIMMA_LORA_TYPES = {"StimmaLoraLoader", "StimmaPairedLoraLoader"}
STIMMA_CHECKPOINT_TYPES = {"StimmaCheckpointLoader"}
STIMMA_OUTPUT_TYPES = {"StimmaImageOutput", "StimmaVideoOutput"}
STIMMA_LAYOUT_TYPES = {"StimmaLayoutGroup"}
ALL_STIMMA_TYPES = (
    {STIMMA_TOOL_INFO}
    | STIMMA_FIELD_TYPES
    | STIMMA_PARAM_TYPES
    | STIMMA_LORA_TYPES
    | STIMMA_CHECKPOINT_TYPES
    | STIMMA_OUTPUT_TYPES
    | STIMMA_LAYOUT_TYPES
)

# Map Stimma node types to their "data" field (the field whose value
# becomes the node's output).  Only types that produce a simple scalar
# (STRING / INT / FLOAT / BOOLEAN) are listed — pipe-output types like
# StimmaImageParam are excluded because their outputs can't be inlined.
# Single string = one output (slot 0).  Tuple of strings = per-slot fields.
_STIMMA_OUTPUT_FIELD = {
    "StimmaPromptParam": "default_text",
    "StimmaSeedParam": "value",
    "StimmaIntParam": "value",
    "StimmaFloatParam": "value",
    "StimmaStringParam": "value",
    "StimmaDropdownParam": "value",
    "StimmaBoolParam": "value",
    "StimmaResolutionParam": ("width", "height"),
}

_IMAGE_INPUT_TYPES = {"StimmaImageParam", "StimmaImagesParam"}
_VIDEO_INPUT_TYPES = {"StimmaVideoParam", "StimmaVideosParam"}

# Annotation-only nodes that don't affect execution — skip in validation
_ANNOTATION_NODE_TYPES = {"Note", "MarkdownNote"}


@dataclass
class DiscoveredWorkflow:
    """A workflow file that contains Stimma nodes."""
    file_path: str
    api_prompt: Dict[str, Any]  # API-format prompt dict
    tool_info: Dict[str, Any]   # slug, display_name, task_types, description, badges
    field_nodes: List[Dict[str, Any]] = field(default_factory=list)   # sorted by order
    param_nodes: List[Dict[str, Any]] = field(default_factory=list)   # sorted by order
    output_nodes: List[Dict[str, Any]] = field(default_factory=list)
    layout_nodes: List[Dict[str, Any]] = field(default_factory=list)  # sorted by order
    lora_nodes: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_nodes: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # validation warnings


# --- Model file extensions used to identify model-file COMBOs ---
_MODEL_EXTENSIONS = frozenset({
    ".safetensors", ".ckpt", ".bin", ".pt", ".pth", ".sft", ".gguf",
})


def _is_model_combo(values: list) -> bool:
    """Return True if a COMBO's enum values look like model filenames."""
    if not values:
        return False
    # Check first few values — if any contain a model extension, it's a model combo
    for v in values[:10]:
        if isinstance(v, str):
            lower = v.lower()
            if any(lower.endswith(ext) for ext in _MODEL_EXTENSIONS):
                return True
    return False


def _extract_combo_values_for_validation(spec) -> Optional[list]:
    """Extract enum values from a ComfyUI input spec (both old and new format)."""
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


def _match_path_filter(name: str, pattern: str) -> bool:
    """Match a file path against one or more glob patterns (semicolon-delimited).

    Supports * (single level) and ** (recursive). Mirrors
    tool_builder._match_lora_filter — duplicated here to avoid a circular import.
    """
    import re as _re
    for part in pattern.split(";"):
        part = part.strip()
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
                regex += _re.escape(part[i])
                i += 1
        if _re.fullmatch(regex, name):
            return True
    return False


def _validate_workflow(
    api_prompt: Dict[str, Any],
    object_info: Optional[Dict[str, Any]],
) -> List[str]:
    """Validate a workflow's api_prompt against object_info.

    Returns a list of warning strings for missing nodes and missing models.
    Non-blocking — workflows with warnings can still be listed and attempted.
    """
    if not object_info:
        return []

    warnings = []
    seen_missing_nodes = set()
    seen_missing_models = set()

    for node_id, node_data in api_prompt.items():
        class_type = node_data.get("class_type", "")

        # StimmaCheckpointLoader needs special handling — it's "present" as a
        # node (we provide it), but the user-selected ckpt_name comes from
        # ComfyUI's installed checkpoint list filtered by path_filter. If the
        # filter matches zero installed checkpoints, the dropdown is empty and
        # the tool can't run.
        if class_type == "StimmaCheckpointLoader":
            node_inputs = node_data.get("inputs", {})
            path_filter = node_inputs.get("path_filter", "") or ""
            node_spec = object_info.get("StimmaCheckpointLoader", {})
            ckpt_spec = (
                node_spec.get("input", {})
                .get("required", {})
                .get("ckpt_name", [])
            )
            installed = _extract_combo_values_for_validation(ckpt_spec) or []
            if path_filter:
                matching = [c for c in installed if _match_path_filter(c, path_filter)]
            else:
                matching = list(installed)
            if not matching:
                key = ("checkpoint", path_filter)
                if key not in seen_missing_models:
                    seen_missing_models.add(key)
                    desc = (
                        f"any checkpoint matching '{path_filter}'"
                        if path_filter else "any checkpoint"
                    )
                    warnings.append(
                        f"missing model: {desc} (StimmaCheckpointLoader.ckpt_name)"
                    )
            continue

        # Skip our own Stimma nodes — always present, values injected at runtime
        if class_type in ALL_STIMMA_TYPES:
            continue

        # Skip annotation-only nodes that don't affect execution
        if class_type in _ANNOTATION_NODE_TYPES:
            continue

        # Missing custom node check
        if class_type not in object_info:
            if class_type not in seen_missing_nodes:
                seen_missing_nodes.add(class_type)
                warnings.append(f"missing node: {class_type}")
            continue

        # Missing model check — inspect string inputs against COMBO specs
        node_spec = object_info[class_type]
        input_def = node_spec.get("input", {})
        node_inputs = node_data.get("inputs", {})

        for input_name, input_value in node_inputs.items():
            # Skip linked inputs (value is [node_id, slot])
            if isinstance(input_value, list):
                continue
            # Only check string values
            if not isinstance(input_value, str) or not input_value:
                continue

            # Look up the input spec in required or optional
            for category in ("required", "optional"):
                spec = input_def.get(category, {}).get(input_name)
                if spec is not None:
                    combo_values = _extract_combo_values_for_validation(spec)
                    if combo_values and _is_model_combo(combo_values):
                        if input_value not in combo_values:
                            key = (input_value, class_type, input_name)
                            if key not in seen_missing_models:
                                seen_missing_models.add(key)
                                warnings.append(
                                    f"missing model: {input_value} ({class_type}.{input_name})"
                                )
                    break

    return warnings


def _is_api_format(data: Any) -> bool:
    """Check if a parsed JSON is in ComfyUI API format (flat dict of {class_type, inputs})."""
    if not isinstance(data, dict):
        return False
    return any(
        isinstance(v, dict) and "class_type" in v
        for v in data.values()
    )


def _is_ui_format(data: Any) -> bool:
    """Check if a parsed JSON is in ComfyUI UI/workflow format (has nodes array)."""
    if not isinstance(data, dict):
        return False
    return "nodes" in data and isinstance(data.get("nodes"), list)


# --- Composite node expansion helpers ---
# These replicate the frontend's graphToPrompt() expansion of group nodes
# and subgraph nodes into their constituent inner nodes.

# Virtual node types that are skipped during expansion (never appear in API prompt)
_VIRTUAL_INNER_TYPES = frozenset({"PrimitiveNode", "Reroute", "Note", "MarkdownNote"})


def _parse_link(link) -> Optional[Dict[str, Any]]:
    """Parse a link in either array or object (SerialisableLLink) format.

    Returns dict with keys: id, src_node, src_slot, dst_node, dst_slot, type.
    """
    if isinstance(link, dict):
        return {
            "id": link.get("id"),
            "src_node": link.get("origin_id"),
            "src_slot": link.get("origin_slot"),
            "dst_node": link.get("target_id"),
            "dst_slot": link.get("target_slot"),
            "type": link.get("type"),
        }
    elif isinstance(link, (list, tuple)) and len(link) >= 6:
        return {
            "id": link[0],
            "src_node": link[1],
            "src_slot": link[2],
            "dst_node": link[3],
            "dst_slot": link[4],
            "type": link[5],
        }
    return None


def _is_group_node_type(node_type: str) -> bool:
    """Check if a node type is a legacy group node (workflow>Name or workflow/Name)."""
    return node_type.startswith("workflow>") or node_type.startswith("workflow/")


def _get_group_name(node_type: str) -> str:
    """Extract group name from 'workflow>Name' or 'workflow/Name'."""
    for sep in (">", "/"):
        if sep in node_type:
            return node_type.split(sep, 1)[1]
    return node_type


def _is_connection_type(inp_type: Any) -> bool:
    """Check if an input type is a connection type (MODEL, CLIP, IMAGE, etc.)."""
    return (
        isinstance(inp_type, str)
        and inp_type.isupper()
        and inp_type not in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")
    )


def _get_input_specs(class_type: str, object_info: Dict) -> List:
    """Get ordered input specs [(name, spec), ...] from object_info."""
    if not object_info or class_type not in object_info:
        return []
    node_info = object_info[class_type]
    input_def = node_info.get("input", {})
    specs = []
    for category in ["required", "optional"]:
        if category in input_def:
            for inp_name in input_def[category]:
                specs.append((inp_name, input_def[category][inp_name]))
    return specs


def _has_control_after_generate(inp_spec) -> bool:
    """Check if an input spec has control_after_generate (frontend-only extra widget)."""
    if isinstance(inp_spec, (list, tuple)) and len(inp_spec) >= 2:
        opts = inp_spec[1]
        if isinstance(opts, dict):
            return bool(opts.get("control_after_generate"))
    return False


def _is_dynamic_combo(inp_spec) -> bool:
    """Check if an input spec is a ComfyUI V3 dynamic combo (COMFY_DYNAMICCOMBO_V3).

    A dynamic combo is a widget whose selected key reveals a set of sub-inputs
    (themselves widgets/connections). In widgets_values it serializes as the key
    followed by the active option's sub-widget values, before any remaining
    top-level widgets. The frontend exposes connected sub-inputs under dotted
    names like ``resize_type.width``.
    """
    return (
        isinstance(inp_spec, (list, tuple))
        and len(inp_spec) >= 1
        and inp_spec[0] == "COMFY_DYNAMICCOMBO_V3"
    )


def _dynamic_combo_suboptions(inp_spec, selected_key) -> List:
    """Return ordered [(sub_name, sub_spec), ...] for the active option of a dynamic combo."""
    if not (isinstance(inp_spec, (list, tuple)) and len(inp_spec) >= 2):
        return []
    opts = inp_spec[1].get("options", []) if isinstance(inp_spec[1], dict) else []
    for opt in opts:
        if opt.get("key") == selected_key:
            sub = []
            opt_inputs = opt.get("inputs", {})
            for cat in ("required", "optional"):
                for sub_name, sub_spec in opt_inputs.get(cat, {}).items():
                    sub.append((sub_name, sub_spec))
            return sub
    return []


def _convert_inner_node_widgets(
    inner_node: Dict, class_type: str, object_info: Optional[Dict],
) -> Dict[str, Any]:
    """Convert an inner node's widgets_values to a named inputs dict.

    Uses the same positional mapping as the main conversion, but only for
    widget-type inputs (not connection-type).
    """
    widget_values = inner_node.get("widgets_values", [])
    if not widget_values:
        return {}

    if isinstance(widget_values, dict):
        return dict(widget_values)

    if not object_info or class_type not in object_info:
        return {}

    result = {}
    # Classify each serialized input. Every widget-type input occupies a
    # widgets_values slot (ComfyUI keeps the widget value as a placeholder even
    # when a link is attached), so slots must ALWAYS be advanced to keep later
    # widgets aligned. Whether the stored value is EMITTED depends on how the
    # input is connected:
    #   - not linked            → plain widget; emit its value.
    #   - linked + `widget` key  → converted widget whose stored value is the
    #     real value (e.g. subgraph inner loaders whose internal socket may be
    #     externally unwired); emit it — the expansion overrides it if the link
    #     actually resolves.
    #   - linked, no `widget` key → a genuine external widget-socket link
    #     supplies the value and the slot holds a stale placeholder; advance
    #     past it but do NOT emit (e.g. StimmaDurationToFrames.duration/fps).
    node_inputs = inner_node.get("inputs", [])
    linked_names = {
        inp.get("name")
        for inp in node_inputs
        if inp.get("name") and inp.get("link") is not None
    }
    converted_names = {
        inp.get("name")
        for inp in node_inputs
        if inp.get("name") and inp.get("link") is not None and inp.get("widget")
    }

    specs = _get_input_specs(class_type, object_info)
    _CAG_STRINGS = frozenset(("randomize", "fixed", "increment", "decrement"))

    def _consume(inp_name, inp_spec):
        """Map one widget-type input spec, consuming widgets_values slots.

        Handles plain widgets, connected-but-converted widgets, dynamic combos
        (which recurse into their active option's sub-inputs), and the
        control_after_generate frontend-only extra slot.
        """
        nonlocal widget_idx
        inp_type = inp_spec[0] if isinstance(inp_spec, (list, tuple)) else inp_spec

        # V3 dynamic combo: the selected key occupies one slot, then the active
        # option's sub-inputs follow (in order), using dotted names.
        if _is_dynamic_combo(inp_spec):
            if widget_idx >= len(widget_values):
                return
            selected_key = widget_values[widget_idx]
            result[inp_name] = selected_key
            widget_idx += 1
            for sub_name, sub_spec in _dynamic_combo_suboptions(inp_spec, selected_key):
                _consume(f"{inp_name}.{sub_name}", sub_spec)
            return

        if _is_connection_type(inp_type):
            return
        # Always advance past this widget's slot (see classification above);
        # emit its value unless a bare external link supplies it.
        emit = (inp_name not in linked_names) or (inp_name in converted_names)
        if widget_idx < len(widget_values):
            if emit:
                result[inp_name] = widget_values[widget_idx]
            widget_idx += 1
        # control_after_generate adds a frontend-only widget that occupies
        # the next slot in widgets_values but is not a backend input
        if _has_control_after_generate(inp_spec):
            widget_idx += 1
        elif (widget_idx < len(widget_values)
              and widget_values[widget_idx] in _CAG_STRINGS):
            # Unlabeled control_after_generate slot (present in widgets_values
            # but omitted from object_info).
            widget_idx += 1

    widget_idx = 0
    for inp_name, inp_spec in specs:
        _consume(inp_name, inp_spec)
    return result


def _follow_inner_chain(
    node_idx: int, slot: int, inner_nodes: List, links_to: Dict,
) -> Optional[tuple]:
    """Follow a chain of Reroute/PrimitiveNode to find the real source.

    Source: groupNode.ts:1025-1028 (updateLink Reroute chain)

    Returns one of:
    - ("node", real_idx, real_slot) — real inner node
    - ("primitive", prim_idx) — PrimitiveNode (use its widget value)
    - ("external", reroute_idx, 0) — Reroute with external input
    - None — unresolvable
    """
    visited = set()
    cur_idx, cur_slot = node_idx, slot

    while True:
        if cur_idx in visited or cur_idx < 0 or cur_idx >= len(inner_nodes):
            return None
        visited.add(cur_idx)

        node_type = inner_nodes[cur_idx].get("type", "")

        if node_type == "PrimitiveNode":
            return ("primitive", cur_idx)

        if node_type != "Reroute":
            return ("node", cur_idx, cur_slot)

        # Reroute: follow its single input (slot 0)
        link = links_to.get(cur_idx, {}).get(0)
        if not link:
            return ("external", cur_idx, 0)

        cur_idx = int(link[0])
        cur_slot = int(link[1])


def _build_group_input_mapping(
    inner_nodes: List, links_to: Dict, links_from: Dict, external_from: Dict,
) -> Dict[int, tuple]:
    """Build mapping: group_input_slot → (inner_node_idx, inner_slot).

    Replicates GroupNodeConfig.processInputSlots + processConvertedWidgets ordering:
    1. Regular connection inputs for all nodes (node-index order)
    2. Converted widget inputs for all nodes (deferred, node-index order)

    Source: groupNode.ts:688-773 (processInputSlots, processConvertedWidgets)
    """
    group_slot = 0
    mapping = {}

    # First pass: regular connection inputs (no widget property)
    for idx, inner_node in enumerate(inner_nodes):
        node_type = inner_node.get("type", "")
        if node_type in ("Note", "MarkdownNote"):
            continue
        if node_type == "PrimitiveNode":
            continue

        if node_type == "Reroute":
            # Reroute is external if it's NOT purely internal
            # Purely internal = has linksTo AND linksFrom AND no externalFrom
            has_input = 0 in links_to.get(idx, {})
            has_output = idx in links_from
            no_ext_output = idx not in external_from
            is_internal = has_input and has_output and no_ext_output
            if is_internal:
                continue  # Skip purely internal Reroutes
            # External Reroute — its input becomes a group input if not connected internally
            if not has_input:
                mapping[group_slot] = (idx, 0)
                group_slot += 1
            continue

        node_inputs = inner_node.get("inputs", [])
        for slot_idx, inp in enumerate(node_inputs):
            if inp.get("widget"):
                continue  # Converted widgets handled in second pass
            if links_to.get(idx, {}).get(slot_idx) is not None:
                continue  # Internally connected
            mapping[group_slot] = (idx, slot_idx)
            group_slot += 1

    # Second pass: converted widget inputs
    for idx, inner_node in enumerate(inner_nodes):
        node_type = inner_node.get("type", "")
        if node_type in _VIRTUAL_INNER_TYPES:
            continue

        node_inputs = inner_node.get("inputs", [])
        for slot_idx, inp in enumerate(node_inputs):
            if not inp.get("widget"):
                continue  # Only converted widgets
            if links_to.get(idx, {}).get(slot_idx) is not None:
                continue
            mapping[group_slot] = (idx, slot_idx)
            group_slot += 1

    return mapping


def _build_group_widget_mapping(
    inner_nodes: List, links_to: Dict, object_info: Optional[Dict],
) -> List[tuple]:
    """Build ordered list of (inner_node_idx, widget_name) for group widgets.

    The position in this list corresponds to the group instance's widgets_values index.

    Order: normal widgets for all nodes first, then converted widgets.
    Source: groupNode.ts:572-624 (processWidgetInputs)
    """
    normal_widgets = []
    converted_widgets = []

    for idx, inner_node in enumerate(inner_nodes):
        node_type = inner_node.get("type", "")
        if not node_type:
            continue

        if node_type == "PrimitiveNode":
            # PrimitiveNode exposes a synthetic "value" widget
            normal_widgets.append((idx, "value"))
            continue

        if node_type in ("Reroute", "Note", "MarkdownNote"):
            continue

        if not object_info or node_type not in object_info:
            continue

        node_inputs = inner_node.get("inputs", [])
        converted_names = {
            inp.get("name")
            for inp in node_inputs
            if inp.get("widget") and inp.get("name")
        }

        specs = _get_input_specs(node_type, object_info)
        for inp_name, inp_spec in specs:
            inp_type = inp_spec[0] if isinstance(inp_spec, (list, tuple)) else inp_spec
            if _is_connection_type(inp_type):
                continue  # Not a widget

            if inp_name in converted_names:
                converted_widgets.append((idx, inp_name))
            else:
                normal_widgets.append((idx, inp_name))

    return normal_widgets + converted_widgets


def _expand_group_node(
    instance_node: Dict,
    ui_data: Dict,
    links_to_node: Dict,
    object_info: Optional[Dict],
) -> tuple:
    """Expand a legacy group node into API-format inner node entries.

    Translates the frontend logic from:
    - groupNode.ts: getInnerNodes, updateLink, updateInnerWidgets
    - executableGroupNodeDto.ts: resolveOutput
    - executableGroupNodeChildDTO.ts: resolveInput

    Returns: (api_entries, rewire_map)
        api_entries: {expanded_id: {"class_type": ..., "inputs": {...}}}
        rewire_map: {(str(instance_id), output_slot): (expanded_id, inner_slot)}
    """
    instance_id = instance_node["id"]
    node_type = instance_node.get("type", "")
    group_name = _get_group_name(node_type)

    group_defs = ui_data.get("extra", {}).get("groupNodes", {})
    group_def = group_defs.get(group_name)
    if not group_def:
        logger.warning(
            f"Group definition not found for '{group_name}' (node {instance_id})"
        )
        return {}, {}

    inner_nodes = group_def.get("nodes", [])
    inner_links = group_def.get("links", [])
    external = group_def.get("external", [])

    if not inner_nodes:
        return {}, {}

    # Build inner link lookups (use INDICES, not IDs)
    # Source: groupNode.ts:304-354 (getLinks)
    links_to: Dict[int, Dict[int, list]] = {}
    links_from: Dict[int, Dict[int, list]] = {}
    for link in inner_links:
        if len(link) < 4:
            continue
        src_idx, src_slot = int(link[0]), int(link[1])
        tgt_idx, tgt_slot = int(link[2]), int(link[3])
        links_to.setdefault(tgt_idx, {})[tgt_slot] = link
        links_from.setdefault(src_idx, {}).setdefault(src_slot, []).append(link)

    # Build external output map
    external_from: Dict[int, Dict[int, Any]] = {}
    for ext in external:
        if len(ext) >= 2:
            external_from.setdefault(int(ext[0]), {})[int(ext[1])] = ext[2] if len(ext) > 2 else True

    # Build mappings
    input_mapping = _build_group_input_mapping(inner_nodes, links_to, links_from, external_from)
    input_mapping_inv = {v: k for k, v in input_mapping.items()}
    widget_mapping = _build_group_widget_mapping(inner_nodes, links_to, object_info)

    # Output mapping from external array
    # Source: groupNode.ts:822-886 (processNodeOutputs)
    # The external array entries are ordered to match group output slots
    output_mapping: Dict[int, tuple] = {}
    for k, ext in enumerate(external):
        if len(ext) >= 2:
            output_mapping[k] = (int(ext[0]), int(ext[1]))

    # Get outer connections to group instance inputs
    outer_inputs = links_to_node.get(instance_id, {})
    instance_widgets = instance_node.get("widgets_values", [])

    api_entries = {}
    rewire_map = {}

    for idx, inner_node in enumerate(inner_nodes):
        inner_type = inner_node.get("type", "")
        if not inner_type or inner_type in _VIRTUAL_INNER_TYPES:
            continue

        # Check for nested group nodes (unsupported)
        if _is_group_node_type(inner_type):
            logger.warning(
                f"Nested group node '{inner_type}' inside '{group_name}' "
                f"(node {instance_id}) — skipping (unsupported)"
            )
            continue

        expanded_id = f"{instance_id}:{idx}"
        inputs = {}

        # 1. Convert inner node's own widget values
        inputs.update(_convert_inner_node_widgets(inner_node, inner_type, object_info))

        # 2. Apply widget overrides from group instance
        if isinstance(instance_widgets, list):
            for wk, (map_idx, map_name) in enumerate(widget_mapping):
                if map_idx != idx:
                    continue
                if wk >= len(instance_widgets):
                    continue
                val = instance_widgets[wk]
                if map_name == "value" and inner_nodes[map_idx].get("type") == "PrimitiveNode":
                    continue  # PrimitiveNode values are applied via links
                inputs[map_name] = val
        elif isinstance(instance_widgets, dict):
            # Dict format: try to match by scanning widget_mapping for this node
            for wk, (map_idx, map_name) in enumerate(widget_mapping):
                if map_idx != idx:
                    continue
                # Look for a key containing this widget name
                for wkey, wval in instance_widgets.items():
                    if wkey == map_name or wkey.endswith(f" {map_name}"):
                        inputs[map_name] = wval
                        break

        # 3. Wire connection inputs
        node_inputs = inner_node.get("inputs", [])
        for slot_idx, inp in enumerate(node_inputs):
            inp_name = inp.get("name", f"input_{slot_idx}")

            inner_link = links_to.get(idx, {}).get(slot_idx)
            if inner_link:
                # Internal link — resolve source through virtual nodes
                src_idx = int(inner_link[0])
                src_slot = int(inner_link[1])
                result = _follow_inner_chain(src_idx, src_slot, inner_nodes, links_to)

                if result is None:
                    continue

                if result[0] == "primitive":
                    # PrimitiveNode: use its widget value
                    prim_idx = result[1]
                    prim_node = inner_nodes[prim_idx]
                    prim_vals = prim_node.get("widgets_values", [])
                    if prim_vals:
                        inputs[inp_name] = prim_vals[0] if isinstance(prim_vals, list) else prim_vals
                    # Check for override from group instance
                    for wk, (map_idx, _) in enumerate(widget_mapping):
                        if map_idx == prim_idx:
                            if isinstance(instance_widgets, list) and wk < len(instance_widgets):
                                inputs[inp_name] = instance_widgets[wk]
                            break

                elif result[0] == "external":
                    # Reroute chain ends at external — find group input slot
                    reroute_idx = result[1]
                    group_slot = input_mapping_inv.get((reroute_idx, 0))
                    if group_slot is not None:
                        outer = outer_inputs.get(group_slot)
                        if outer:
                            inputs[inp_name] = [str(outer["src_node"]), outer["src_slot"]]

                elif result[0] == "node":
                    # Real inner node
                    real_idx, real_slot = result[1], result[2]
                    inputs[inp_name] = [f"{instance_id}:{real_idx}", real_slot]
            else:
                # No internal link — external input
                group_slot = input_mapping_inv.get((idx, slot_idx))
                if group_slot is not None:
                    outer = outer_inputs.get(group_slot)
                    if outer:
                        inputs[inp_name] = [str(outer["src_node"]), outer["src_slot"]]

        api_entries[expanded_id] = {
            "class_type": inner_type,
            "inputs": inputs,
        }

    # Build rewire map for outputs
    for group_slot, (inner_idx, inner_slot) in output_mapping.items():
        # Follow Reroute chain for outputs too
        result = _follow_inner_chain(inner_idx, inner_slot, inner_nodes, links_to)
        if result and result[0] == "node":
            final_idx, final_slot = result[1], result[2]
            rewire_map[(str(instance_id), group_slot)] = (
                f"{instance_id}:{final_idx}",
                final_slot,
            )
        elif result is None or result[0] in ("primitive", "external"):
            # Fallback: use the original inner node even if virtual
            rewire_map[(str(instance_id), group_slot)] = (
                f"{instance_id}:{inner_idx}",
                inner_slot,
            )

    logger.debug(
        f"Expanded group node '{group_name}' (#{instance_id}): "
        f"{len(api_entries)} inner nodes, {len(rewire_map)} output rewires"
    )
    return api_entries, rewire_map


def _follow_subgraph_inner_reroutes(
    src_id: Any, src_slot: int,
    inner_node_map: Dict, il_to: Dict,
    input_node_id: Any,
) -> tuple:
    """Follow Reroute chains within a subgraph's inner nodes to find the real source.

    Returns (src_id, src_slot) of the first non-Reroute node encountered, or the
    inputNode if the chain reaches an external boundary.
    """
    visited: set = set()
    while True:
        if src_id in visited:
            break  # cycle guard
        visited.add(src_id)

        node = inner_node_map.get(src_id)
        if node is None:
            break
        if node.get("type") != "Reroute":
            break

        # Reroute: follow its single input (slot 0)
        upstream = il_to.get(src_id, {}).get(0)
        if upstream is None:
            break  # dangling reroute — leave as-is
        src_id = upstream["src_node"]
        src_slot = upstream["src_slot"]

    return src_id, src_slot


def _expand_subgraph_node(
    instance: Dict,
    ui_data: Dict,
    links_to_node: Dict,
    object_info: Optional[Dict],
) -> tuple:
    """Expand a subgraph node into API-format inner node entries.

    Translates the frontend logic from:
    - SubgraphNode.ts: getInnerNodes
    - ExecutableNodeDTO.ts: resolveInput, _resolveSubgraphOutput

    Returns: (api_entries, rewire_map)
    """
    instance_id = instance["id"]
    instance_type = instance.get("type", "")

    definitions = ui_data.get("definitions", {}).get("subgraphs", [])
    defn = None
    for d in definitions:
        if d.get("id") == instance_type:
            defn = d
            break

    if not defn:
        logger.warning(
            f"Subgraph definition not found for type '{instance_type}' "
            f"(instance {instance_id})"
        )
        return {}, {}

    input_node_id = defn.get("inputNode", {}).get("id")
    output_node_id = defn.get("outputNode", {}).get("id")
    inner_nodes = defn.get("nodes", [])
    inner_links = defn.get("links", [])
    defn_inputs = defn.get("inputs", [])
    defn_outputs = defn.get("outputs", [])
    defn_widgets = defn.get("widgets", [])

    if not inner_nodes:
        return {}, {}

    # Build inner link lookups (uses node IDs, not indices)
    # Links are in object format: {id, origin_id, origin_slot, target_id, target_slot, type}
    il_to: Dict[Any, Dict[int, Dict]] = {}   # {target_id: {target_slot: link_dict}}
    il_from: Dict[Any, Dict[int, list]] = {}  # {origin_id: {origin_slot: [link_dicts]}}
    for link in inner_links:
        parsed = _parse_link(link)
        if not parsed:
            continue
        tgt_id = parsed["dst_node"]
        tgt_slot = parsed["dst_slot"]
        src_id = parsed["src_node"]
        src_slot = parsed["src_slot"]
        if tgt_id is not None and tgt_slot is not None:
            il_to.setdefault(tgt_id, {})[tgt_slot] = parsed
        if src_id is not None and src_slot is not None:
            il_from.setdefault(src_id, {}).setdefault(src_slot, []).append(parsed)

    # Build input mapping: inner link ID → subgraph input slot
    # A single subgraph input can fan out to multiple inner nodes (multiple linkIds)
    inner_link_to_sg_slot: Dict[Any, int] = {}  # link_id → sg input slot
    for slot_idx, sg_input in enumerate(defn_inputs):
        for lid in sg_input.get("linkIds", []):
            inner_link_to_sg_slot[lid] = slot_idx

    # Build output mapping: subgraph output slot → inner source
    sg_output_source: Dict[int, Dict] = {}
    for slot_idx, sg_output in enumerate(defn_outputs):
        link_ids = sg_output.get("linkIds", [])
        for lid in link_ids:
            for link in inner_links:
                parsed = _parse_link(link)
                if parsed and parsed["id"] == lid:
                    sg_output_source[slot_idx] = parsed
                    break

    # Widget overrides: instance.widgets_values[k] → inner_node[defn_widgets[k].id].widget[name]
    widget_overrides: Dict[Any, Dict[str, Any]] = {}  # {inner_node_id: {widget_name: value}}
    instance_widget_values = instance.get("widgets_values", [])
    for k, wdef in enumerate(defn_widgets):
        inner_nid = wdef.get("id")
        wname = wdef.get("name")
        if inner_nid is not None and wname and k < len(instance_widget_values):
            widget_overrides.setdefault(inner_nid, {})[wname] = instance_widget_values[k]

    # Build outer connection mapping: definition input index → outer source info.
    # The subgraph node's link input slots may NOT match definition input indices
    # (e.g., only some definition inputs have link inputs on the subgraph node,
    # and they can be in any order). We match by NAME.
    raw_outer = links_to_node.get(instance_id, {})
    instance_inputs = instance.get("inputs", [])
    # Map definition input name → definition input index
    defn_input_name_to_idx = {inp["name"]: idx for idx, inp in enumerate(defn_inputs)}
    # Build definition input index → outer source info
    outer_inputs: Dict[int, Dict] = {}
    for sg_slot, source_info in raw_outer.items():
        if sg_slot < len(instance_inputs):
            sg_input_name = instance_inputs[sg_slot].get("name", "")
            defn_idx = defn_input_name_to_idx.get(sg_input_name)
            if defn_idx is not None:
                outer_inputs[defn_idx] = source_info

    api_entries = {}
    rewire_map = {}

    # Build node lookup for inner nodes
    inner_node_map = {}
    for n in inner_nodes:
        nid = n.get("id")
        if nid is not None:
            inner_node_map[nid] = n

    for inner_node in inner_nodes:
        node_id = inner_node.get("id")
        if node_id is None:
            continue
        if node_id == input_node_id or node_id == output_node_id:
            continue  # Skip I/O bridge nodes

        inner_type = inner_node.get("type", "")
        if not inner_type or inner_type in _VIRTUAL_INNER_TYPES:
            continue

        expanded_id = f"{instance_id}:{node_id}"
        inputs = {}

        # 1. Convert inner node's widget values
        inputs.update(_convert_inner_node_widgets(inner_node, inner_type, object_info))

        # 2. Apply widget overrides
        overrides = widget_overrides.get(node_id, {})
        inputs.update(overrides)

        # 3. Wire connection inputs
        node_inputs = inner_node.get("inputs", [])
        for slot_idx, inp in enumerate(node_inputs):
            inp_name = inp.get("name", f"input_{slot_idx}")

            # Find inner link targeting this node+slot
            inner_link = il_to.get(node_id, {}).get(slot_idx)
            if inner_link:
                src_id = inner_link["src_node"]
                src_slot = inner_link["src_slot"]

                if src_id == input_node_id:
                    # Source is the inputNode → external connection
                    # Find which subgraph input slot this corresponds to
                    sg_slot = inner_link_to_sg_slot.get(inner_link["id"])
                    if sg_slot is not None:
                        outer = outer_inputs.get(sg_slot)
                        if outer:
                            inputs[inp_name] = [str(outer["src_node"]), outer["src_slot"]]
                        else:
                            # No outer link — check for exposed widget value
                            woverride = widget_overrides.get(node_id, {}).get(inp_name)
                            if woverride is not None:
                                inputs[inp_name] = woverride
                elif src_id == output_node_id:
                    # Shouldn't happen — outputNode shouldn't be a source
                    logger.warning(f"Subgraph inner link sources from outputNode (instance {instance_id})")
                else:
                    # Normal internal link — follow any Reroute chains
                    src_id, src_slot = _follow_subgraph_inner_reroutes(
                        src_id, src_slot, inner_node_map, il_to, input_node_id
                    )
                    if src_id == input_node_id:
                        # Reroute chain leads to an external input — resolve it
                        sg_slot = inner_link_to_sg_slot.get(inner_link["id"])
                        if sg_slot is not None:
                            outer = outer_inputs.get(sg_slot)
                            if outer:
                                inputs[inp_name] = [str(outer["src_node"]), outer["src_slot"]]
                    else:
                        inputs[inp_name] = [f"{instance_id}:{src_id}", src_slot]

        api_entries[expanded_id] = {
            "class_type": inner_type,
            "inputs": inputs,
        }

    # Build rewire map for outputs — follow any Reroute chains
    for sg_slot, sg_link in sg_output_source.items():
        src_id = sg_link["src_node"]
        src_slot = sg_link["src_slot"]
        if src_id != input_node_id and src_id != output_node_id:
            src_id, src_slot = _follow_subgraph_inner_reroutes(
                src_id, src_slot, inner_node_map, il_to, input_node_id
            )
            rewire_map[(str(instance_id), sg_slot)] = (
                f"{instance_id}:{src_id}",
                src_slot,
            )

    logger.debug(
        f"Expanded subgraph '{defn.get('name', '?')}' (#{instance_id}): "
        f"{len(api_entries)} inner nodes, {len(rewire_map)} output rewires"
    )
    return api_entries, rewire_map


def _expand_api_subgraph_node(
    instance_id: str,
    api_node: Dict,
    ui_data: Dict,
    object_info: Optional[Dict],
) -> tuple:
    """Expand an already-API-format subgraph node into its inner nodes.

    Used for nested subgraphs: after the first-pass expansion of a top-level
    subgraph node, inner nodes that are themselves subgraph instances appear in
    api_prompt with a UUID class_type but with inputs already resolved to
    [src_node, src_slot] pairs rather than UI-format link IDs.

    Returns: (api_entries, rewire_map)
    """
    class_type = api_node.get("class_type", "")
    resolved_inputs = api_node.get("inputs", {})

    definitions = ui_data.get("definitions", {}).get("subgraphs", [])
    defn = None
    for d in definitions:
        if d.get("id") == class_type:
            defn = d
            break
    if not defn:
        return {}, {}

    input_node_id = defn.get("inputNode", {}).get("id")
    output_node_id = defn.get("outputNode", {}).get("id")
    inner_nodes = defn.get("nodes", [])
    inner_links = defn.get("links", [])
    defn_inputs = defn.get("inputs", [])
    defn_outputs = defn.get("outputs", [])

    if not inner_nodes:
        return {}, {}

    # Build inner link lookups
    il_to: Dict[Any, Dict[int, Dict]] = {}
    for link in inner_links:
        parsed = _parse_link(link)
        if not parsed:
            continue
        tgt_id = parsed["dst_node"]
        tgt_slot = parsed["dst_slot"]
        if tgt_id is not None and tgt_slot is not None:
            il_to.setdefault(tgt_id, {})[tgt_slot] = parsed

    # inner link ID → definition input slot index
    inner_link_to_sg_slot: Dict[Any, int] = {}
    for slot_idx, sg_input in enumerate(defn_inputs):
        for lid in sg_input.get("linkIds", []):
            inner_link_to_sg_slot[lid] = slot_idx

    # definition input name → slot index
    defn_input_name_to_idx = {inp["name"]: idx for idx, inp in enumerate(defn_inputs)}

    # Map definition input slot → resolved (src_node, src_slot) from the already-expanded
    # api_node's inputs dict (matched by definition input NAME)
    outer_inputs: Dict[int, Dict] = {}
    for inp_name, inp_val in resolved_inputs.items():
        if isinstance(inp_val, list) and len(inp_val) == 2:
            defn_idx = defn_input_name_to_idx.get(inp_name)
            if defn_idx is not None:
                outer_inputs[defn_idx] = {"src_node": inp_val[0], "src_slot": inp_val[1]}

    # definition output slot → inner source
    sg_output_source: Dict[int, Dict] = {}
    for slot_idx, sg_output in enumerate(defn_outputs):
        for lid in sg_output.get("linkIds", []):
            for link in inner_links:
                parsed = _parse_link(link)
                if parsed and parsed["id"] == lid:
                    sg_output_source[slot_idx] = parsed
                    break

    inner_node_map = {n.get("id"): n for n in inner_nodes}
    api_entries = {}
    rewire_map = {}

    for inner_node in inner_nodes:
        node_id = inner_node.get("id")
        if node_id is None:
            continue
        if node_id == input_node_id or node_id == output_node_id:
            continue
        inner_type = inner_node.get("type", "")
        if not inner_type or inner_type in _VIRTUAL_INNER_TYPES:
            continue

        expanded_id = f"{instance_id}:{node_id}"
        inputs = {}

        # Widget values
        inputs.update(_convert_inner_node_widgets(inner_node, inner_type, object_info))

        # Wire connection inputs
        node_inputs_list = inner_node.get("inputs", [])
        for slot_idx, inp in enumerate(node_inputs_list):
            inp_name = inp.get("name", f"input_{slot_idx}")
            inner_link = il_to.get(node_id, {}).get(slot_idx)
            if inner_link:
                src_id = inner_link["src_node"]
                src_slot = inner_link["src_slot"]
                if src_id == input_node_id:
                    sg_slot = inner_link_to_sg_slot.get(inner_link["id"])
                    if sg_slot is not None:
                        outer = outer_inputs.get(sg_slot)
                        if outer:
                            inputs[inp_name] = [str(outer["src_node"]), outer["src_slot"]]
                else:
                    inputs[inp_name] = [f"{instance_id}:{src_id}", src_slot]

        api_entries[expanded_id] = {
            "class_type": inner_type,
            "inputs": inputs,
        }

    # rewire_map: (str(instance_id), output_slot) → (inner expanded id, inner slot)
    for sg_slot, src in sg_output_source.items():
        src_node = src["src_node"]
        src_slot_val = src["src_slot"]
        if src_node not in (input_node_id, output_node_id):
            rewire_map[(str(instance_id), sg_slot)] = (
                f"{instance_id}:{src_node}", src_slot_val
            )

    return api_entries, rewire_map


def _resolve_stimma_links(api_prompt: Dict[str, Any]) -> None:
    """Resolve link references to Stimma nodes by replacing with literal values.

    Stimma nodes (inputs, params) are unknown to ComfyUI.  When the executor
    strips unknown node types it also strips every node that transitively
    depends on them.  If a ComfyUI node (e.g. KSampler) references a Stimma
    node via a link, it would be stripped too.

    This function replaces those link references with the current literal
    value stored on the Stimma node, so that after stripping the prompt
    still contains all the real ComfyUI nodes.

    It also fixes the STRING→COMBO type mismatch: ComfyUI rejects STRING
    links targeting COMBO inputs (e.g. sampler_name, scheduler), but
    literal string values are accepted.

    Call this AFTER _inject_params / _inject_fields so that user-provided
    values are already on the Stimma nodes.
    """
    # Build map: (stimma_node_id, slot) → output value
    stimma_values: Dict[tuple, Any] = {}
    stimma_ids: set = set()
    for nid, nd in api_prompt.items():
        ct = nd.get("class_type", "")
        field_info = _STIMMA_OUTPUT_FIELD.get(ct)
        if not field_info:
            continue
        stimma_ids.add(nid)
        inputs = nd.get("inputs", {})
        if isinstance(field_info, tuple):
            # Multi-output: one field per slot
            for slot, field in enumerate(field_info):
                value = inputs.get(field)
                if value is not None:
                    stimma_values[(nid, slot)] = value
        else:
            value = inputs.get(field_info)
            if value is not None:
                stimma_values[(nid, 0)] = value

    if not stimma_values:
        return

    # Replace link references [stimma_node_id, slot] with literal values
    for nid, nd in api_prompt.items():
        if nid in stimma_ids:
            continue  # Don't modify Stimma nodes themselves
        for inp_name, inp_val in list(nd.get("inputs", {}).items()):
            if isinstance(inp_val, list) and len(inp_val) == 2 and isinstance(inp_val[0], str):
                key = (str(inp_val[0]), inp_val[1])
                if key in stimma_values:
                    nd["inputs"][inp_name] = stimma_values[key]


def _convert_ui_to_api(ui_data: Dict[str, Any], object_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Convert a ComfyUI UI-format workflow to API format.

    The UI format stores widget values positionally in `widgets_values` arrays.
    The API format stores them as named keys in `inputs` dicts.
    We use object_info to map positions to names.

    Handles expansion of composite nodes:
    - Legacy group nodes (type: workflow>Name or workflow/Name)
    - Subgraph nodes (type: UUID, in nodes array or workflow.subgraphs)
    """
    nodes = ui_data.get("nodes", [])
    links = ui_data.get("links", [])

    if not nodes and not ui_data.get("subgraphs"):
        return None

    # Build set of known subgraph definition IDs so we can detect subgraph
    # instances in the regular nodes array (they have UUID types matching a def)
    subgraph_def_ids = set()
    for d in ui_data.get("definitions", {}).get("subgraphs", []):
        did = d.get("id")
        if did:
            subgraph_def_ids.add(did)

    # Build link lookups — supports both array and object (SerialisableLLink) formats
    link_map = {}        # link_id → {src_node, src_slot}
    links_to_node = {}   # dst_node_id → {dst_slot: {src_node, src_slot}}
    for link in links:
        parsed = _parse_link(link)
        if not parsed or parsed["id"] is None:
            continue
        link_map[parsed["id"]] = {
            "src_node": parsed["src_node"],
            "src_slot": parsed["src_slot"],
        }
        if parsed["dst_node"] is not None and parsed["dst_slot"] is not None:
            links_to_node.setdefault(parsed["dst_node"], {})[parsed["dst_slot"]] = {
                "src_node": parsed["src_node"],
                "src_slot": parsed["src_slot"],
            }

    api_prompt = {}
    rewire_map = {}  # (str(instance_id), output_slot) → (expanded_id, inner_slot)

    # Mode 4 = NEVER (muted), mode 2 = ON_TRIGGER — the frontend's graphToPrompt
    # skips nodes with mode NEVER or BYPASS (mode 4).  We do the same.
    _MUTED_MODE = 4

    # Build bypass map for muted nodes: (muted_node_id, output_slot) → (src_node, src_slot)
    # Muted nodes pass-through: Nth input connects to Nth output.
    muted_bypass = {}  # (str(node_id), output_slot) → (str(src_node), src_slot)
    for node in nodes:
        if node.get("mode") != _MUTED_MODE:
            continue
        node_id = str(node.get("id"))
        node_inputs = node.get("inputs", [])
        # Build ordered list of connection-type inputs (non-widget inputs)
        conn_inputs = []
        for inp in node_inputs:
            link_id = inp.get("link")
            if link_id is not None and link_id in link_map:
                conn_inputs.append(link_map[link_id])
            else:
                conn_inputs.append(None)
        # Map each output slot to the corresponding input's source
        for slot_idx, src in enumerate(conn_inputs):
            if src is not None:
                muted_bypass[(node_id, slot_idx)] = (str(src["src_node"]), src["src_slot"])

    for node in nodes:
        node_id = str(node.get("id"))
        class_type = node.get("type")
        if not class_type:
            continue

        # Skip muted/disabled nodes (mode 4 = NEVER in LiteGraph)
        if node.get("mode") == _MUTED_MODE:
            continue

        # Expand group nodes
        if _is_group_node_type(class_type):
            expanded, rewires = _expand_group_node(
                node, ui_data, links_to_node, object_info
            )
            api_prompt.update(expanded)
            rewire_map.update(rewires)
            continue

        # Expand subgraph instances found in the nodes array
        if class_type in subgraph_def_ids:
            expanded, rewires = _expand_subgraph_node(
                node, ui_data, links_to_node, object_info
            )
            api_prompt.update(expanded)
            rewire_map.update(rewires)
            continue

        # Normal node conversion
        inputs = {}
        widget_values = node.get("widgets_values", [])

        # Map link connections to inputs
        node_inputs = node.get("inputs", [])
        connected_input_names = set()
        for inp in node_inputs:
            inp_name = inp.get("name")
            link_id = inp.get("link")
            if inp_name and link_id is not None and link_id in link_map:
                link_info = link_map[link_id]
                inputs[inp_name] = [str(link_info["src_node"]), link_info["src_slot"]]
                connected_input_names.add(inp_name)

        if isinstance(widget_values, dict):
            for wname, wval in widget_values.items():
                if wname not in connected_input_names:
                    inputs[wname] = wval
        else:
            if (object_info and class_type not in object_info
                    and class_type in ALL_STIMMA_TYPES):
                logger.warning(
                    f"object_info missing for Stimma node type '{class_type}' "
                    f"(node {node_id}) — widget values cannot be mapped"
                )
            # Positional widget mapping — shared with subgraph inner nodes so
            # dynamic combos (COMFY_DYNAMICCOMBO_V3, e.g. ResizeImageMaskNode),
            # converted widgets, and control_after_generate are handled the same.
            for wname, wval in _convert_inner_node_widgets(node, class_type, object_info).items():
                if wname not in connected_input_names:
                    inputs[wname] = wval

        api_prompt[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }

    # Also expand subgraph instances from the dedicated subgraphs array
    for instance in ui_data.get("subgraphs", []):
        expanded, rewires = _expand_subgraph_node(
            instance, ui_data, links_to_node, object_info
        )
        api_prompt.update(expanded)
        rewire_map.update(rewires)

    # Post-process: expand any nested subgraph nodes that appeared as inner nodes
    # of the first-pass expansion (subgraph-within-subgraph, up to 5 levels deep).
    for _pass in range(5):
        nested = [
            (nid, nd) for nid, nd in list(api_prompt.items())
            if nd.get("class_type") in subgraph_def_ids
        ]
        if not nested:
            break
        for nested_nid, nested_nd in nested:
            expanded, nested_rewires = _expand_api_subgraph_node(
                nested_nid, nested_nd, ui_data, object_info
            )
            del api_prompt[nested_nid]
            api_prompt.update(expanded)
            rewire_map.update(nested_rewires)

    # Post-process: reroute links through muted/bypassed nodes
    if muted_bypass:
        for node_data in api_prompt.values():
            for inp_name, inp_val in list(node_data.get("inputs", {}).items()):
                if isinstance(inp_val, list) and len(inp_val) == 2 and isinstance(inp_val[0], str):
                    key = (inp_val[0], inp_val[1])
                    # Follow bypass chains (muted → muted → ...)
                    seen = set()
                    while key in muted_bypass and key not in seen:
                        seen.add(key)
                        key = muted_bypass[key]
                    if key != (inp_val[0], inp_val[1]):
                        node_data["inputs"][inp_name] = [key[0], key[1]]

    # Post-process: rewire references to composite node outputs
    if rewire_map:
        for node_data in api_prompt.values():
            for inp_name, inp_val in list(node_data.get("inputs", {}).items()):
                if isinstance(inp_val, list) and len(inp_val) == 2:
                    key = (str(inp_val[0]), inp_val[1])
                    if key in rewire_map:
                        new_id, new_slot = rewire_map[key]
                        node_data["inputs"][inp_name] = [new_id, new_slot]

    # Post-process: remove inputs referencing absent (muted/deleted) nodes
    for node_data in api_prompt.values():
        for inp_name, inp_val in list(node_data.get("inputs", {}).items()):
            if isinstance(inp_val, list) and len(inp_val) == 2 and isinstance(inp_val[0], str):
                if inp_val[0] not in api_prompt:
                    del node_data["inputs"][inp_name]

    return api_prompt if api_prompt else None


def _extract_stimma_nodes(api_prompt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract Stimma node information from an API-format prompt.

    Returns None if no StimmaToolInfo node is found.
    """
    tool_info = None
    field_nodes = []
    param_nodes = []
    output_nodes = []
    layout_nodes = []
    lora_nodes = []
    checkpoint_nodes = []

    for node_id, node_data in api_prompt.items():
        class_type = node_data.get("class_type", "")
        if class_type not in ALL_STIMMA_TYPES:
            continue

        inputs = node_data.get("inputs", {})
        node_entry = {
            "node_id": node_id,
            "class_type": class_type,
            "inputs": inputs,
        }

        if class_type == STIMMA_TOOL_INFO:
            # Parse task_types (comma-separated)
            raw_task_types_text = inputs.get("task_types", "")
            task_types = []
            if raw_task_types_text and raw_task_types_text.strip():
                for part in raw_task_types_text.strip().split(","):
                    tt = part.strip()
                    if tt:
                        task_types.append(tt)
            if not task_types:
                task_types = ["other"]

            raw_badges_text = inputs.get("badges", "")
            badges = []
            if raw_badges_text and str(raw_badges_text).strip():
                seen_badges = set()
                for line in str(raw_badges_text).strip().split("\n"):
                    badge = line.strip()
                    if not badge or badge in seen_badges:
                        continue
                    seen_badges.add(badge)
                    badges.append(badge)

            model_vendor = (inputs.get("model_vendor") or "").strip()
            model = (inputs.get("model") or "").strip()

            tool_info = {
                "slug": inputs.get("slug", ""),
                "display_name": inputs.get("display_name", ""),
                "task_types": task_types,
                "description": inputs.get("description", ""),
                "badges": badges,
                "model_vendor": model_vendor,
                "model": model,
            }
        elif class_type in STIMMA_FIELD_TYPES:
            node_entry["order"] = inputs.get("ui_order", 0)
            if class_type in _IMAGE_INPUT_TYPES:
                node_entry["name"] = "input_images"
            elif class_type in _VIDEO_INPUT_TYPES:
                node_entry["name"] = "input_videos"
            else:
                node_entry["name"] = inputs.get("name", "")
            field_nodes.append(node_entry)
        elif class_type in STIMMA_PARAM_TYPES:
            node_entry["order"] = inputs.get("ui_order", 0)
            node_entry["name"] = inputs.get("name", "")
            param_nodes.append(node_entry)
        elif class_type in STIMMA_LORA_TYPES:
            node_entry["order"] = inputs.get("ui_order", 50)
            node_entry["name"] = inputs.get("name", "loras")
            lora_nodes.append(node_entry)
        elif class_type in STIMMA_CHECKPOINT_TYPES:
            node_entry["order"] = inputs.get("ui_order", 50)
            node_entry["name"] = inputs.get("name", "checkpoint")
            checkpoint_nodes.append(node_entry)
        elif class_type in STIMMA_OUTPUT_TYPES:
            output_nodes.append(node_entry)
        elif class_type in STIMMA_LAYOUT_TYPES:
            node_entry["order"] = inputs.get("ui_order", 0)
            layout_nodes.append(node_entry)

    if tool_info is None:
        has_stimma_nodes = any(
            node_data.get("class_type", "") in ALL_STIMMA_TYPES
            for node_data in api_prompt.values()
            if isinstance(node_data, dict)
        )
        if has_stimma_nodes:
            logger.debug(
                "Workflow has Stimma nodes but no StimmaToolInfo — skipping"
            )
        return None

    # Sort by order (coerce to int — widget values may arrive as strings)
    def _order_key(n):
        try:
            return int(n["order"])
        except (ValueError, TypeError):
            return 0

    field_nodes.sort(key=_order_key)
    param_nodes.sort(key=_order_key)
    layout_nodes.sort(key=_order_key)
    lora_nodes.sort(key=_order_key)
    checkpoint_nodes.sort(key=_order_key)

    # Resolve StimmaDropdownParam connections: find the target ComfyUI node
    # and input name so tool_builder can look up the enum from object_info.
    dropdown_ids = {
        n["node_id"] for n in param_nodes if n["class_type"] == "StimmaDropdownParam"
    }
    if dropdown_ids:
        for nid, nd in api_prompt.items():
            if nid in dropdown_ids:
                continue
            ct = nd.get("class_type", "")
            if ct in ALL_STIMMA_TYPES:
                continue
            for inp_name, inp_val in nd.get("inputs", {}).items():
                if (
                    isinstance(inp_val, list)
                    and len(inp_val) == 2
                    and str(inp_val[0]) in dropdown_ids
                ):
                    # Found a non-Stimma node referencing this dropdown
                    src_id = str(inp_val[0])
                    for pn in param_nodes:
                        if pn["node_id"] == src_id:
                            pn["target_class_type"] = ct
                            pn["target_input_name"] = inp_name
                            break

    return {
        "tool_info": tool_info,
        "field_nodes": field_nodes,
        "param_nodes": param_nodes,
        "output_nodes": output_nodes,
        "layout_nodes": layout_nodes,
        "lora_nodes": lora_nodes,
        "checkpoint_nodes": checkpoint_nodes,
    }


def _get_comfyui_workflow_dirs() -> List[str]:
    """Get ComfyUI's workflow directories."""
    dirs = []

    # Try to find ComfyUI's user workflow directory
    try:
        import folder_paths
        user_dir = os.path.join(folder_paths.get_user_directory(), "default", "workflows")
        if os.path.isdir(user_dir):
            dirs.append(user_dir)
    except (ImportError, AttributeError):
        logger.warning("Could not import folder_paths — ComfyUI workflow directories unavailable")

    # Also check common locations relative to ComfyUI root
    try:
        import folder_paths
        base = folder_paths.base_path
        for subdir in ["user/default/workflows", "workflows"]:
            candidate = os.path.join(base, subdir)
            if os.path.isdir(candidate) and candidate not in dirs:
                dirs.append(candidate)
    except (ImportError, AttributeError):
        pass

    return dirs


def _is_scannable_json(filename: str) -> bool:
    """Check if a filename should be included in workflow scanning."""
    if not filename.endswith(".json"):
        return False
    if filename.startswith(".") or filename.endswith(".backup.json"):
        return False
    return True


def _scan_single_file(
    filepath: str,
    filename: str,
    object_info: Optional[Dict[str, Any]],
    workflows: list,
    errors: list,
) -> None:
    """Process one JSON file. Appends to workflows or errors."""
    with open(filepath, "r") as f:
        data = json.load(f)

    api_prompt = None
    if _is_api_format(data):
        api_prompt = data
    elif _is_ui_format(data):
        api_prompt = _convert_ui_to_api(data, object_info)
    else:
        return

    if api_prompt is None:
        return

    result = _extract_stimma_nodes(api_prompt)
    if result is None:
        return

    # Validate workflow against object_info
    wf_warnings = _validate_workflow(api_prompt, object_info)

    workflows.append(DiscoveredWorkflow(
        file_path=filepath,
        api_prompt=api_prompt,
        tool_info=result["tool_info"],
        field_nodes=result["field_nodes"],
        param_nodes=result["param_nodes"],
        output_nodes=result["output_nodes"],
        layout_nodes=result["layout_nodes"],
        lora_nodes=result["lora_nodes"],
        checkpoint_nodes=result["checkpoint_nodes"],
        warnings=wf_warnings,
    ))


@dataclass
class _ScanResult:
    """Result of scanning one directory."""
    directory: str
    workflows: List[DiscoveredWorkflow]
    json_count: int
    errors: List[str]


def _scan_directory(
    directory: str,
    object_info: Optional[Dict[str, Any]] = None,
) -> _ScanResult:
    """Scan a directory recursively for Stimma-enabled workflow JSON files."""
    workflows = []
    json_count = 0
    errors = []

    for root, _dirs, files in os.walk(directory):
        for filename in files:
            if not _is_scannable_json(filename):
                continue
            json_count += 1

            filepath = os.path.join(root, filename)
            try:
                _scan_single_file(
                    filepath, filename, object_info,
                    workflows, errors,
                )
            except Exception as e:
                errors.append(
                    f"\033[31m\u2718 {filename}\033[0m  {type(e).__name__}: {e}"
                )
                if not isinstance(e, (json.JSONDecodeError, OSError, ValueError)):
                    logger.debug(f"Full traceback for {filename}:", exc_info=True)

    return _ScanResult(
        directory=directory,
        workflows=workflows,
        json_count=json_count,
        errors=errors,
    )


def discover_workflows(
    config: Config,
    object_info: Optional[Dict[str, Any]] = None,
) -> List[DiscoveredWorkflow]:
    """Discover all Stimma-enabled workflows.

    Scans:
    1. ComfyUI's own workflow directories
    2. Extra directories from config
    """
    all_workflows = []
    seen_slugs = {}
    duplicates = []
    scan_results = []

    # Scan ComfyUI workflow dirs
    for directory in _get_comfyui_workflow_dirs():
        scan_results.append(_scan_directory(directory, object_info))

    # Scan extra directories
    for directory in config.discovery.extra_workflow_dirs:
        if not os.path.isdir(directory):
            scan_results.append(_ScanResult(
                directory=directory, workflows=[], json_count=0,
                errors=[f"\033[33m\u26a0 Directory not found\033[0m"],
            ))
            continue
        scan_results.append(_scan_directory(directory, object_info))

    # De-duplicate across all scan results
    for sr in scan_results:
        for w in sr.workflows:
            slug = w.tool_info["slug"]
            if not slug:
                all_workflows.append(w)
                continue
            if slug not in seen_slugs:
                all_workflows.append(w)
                seen_slugs[slug] = w.file_path
            else:
                duplicates.append((slug, seen_slugs[slug], w.file_path))

    # --- Build boxed output ---
    W = 50  # box inner width
    BAR = "\u2550"
    lines = []

    # Per-directory sections
    for sr in scan_results:
        dir_short = os.path.basename(sr.directory) or sr.directory
        parts = [f"{sr.json_count} files scanned"]
        if sr.errors:
            parts.append(f"\033[31m{len(sr.errors)} errors\033[0m")
        lines.append(
            f"\033[35m\u2502\033[0m  \033[36m{dir_short}/\033[0m  "
            f"\033[2m{', '.join(parts)}\033[0m"
        )

        # Show each discovered workflow
        for w in sr.workflows:
            slug = w.tool_info["slug"]
            name = w.tool_info["display_name"]
            n_in = len(w.field_nodes)
            n_p = len(w.param_nodes)
            n_l = len(w.lora_nodes)
            n_o = len(w.output_nodes)
            if slug:
                has_warnings = bool(w.warnings)
                check = "\033[33m\u26a0\033[0m" if has_warnings else "\033[32m\u2714\033[0m"
                lines.append(
                    f"\033[35m\u2502\033[0m  {check} "
                    f"\033[1m{slug}\033[0m \033[2m\"{name}\"\033[0m"
                )
                lines.append(
                    f"\033[35m\u2502\033[0m    "
                    f"{n_in} inputs, {n_p} params, {n_l} loras, {n_o} outputs"
                )
                if has_warnings:
                    # One line per missing dependency: "⚠ missing <kind> <name>"
                    for warn in w.warnings:
                        if warn.startswith("missing node: "):
                            item = f"missing node {warn[14:]}"
                        elif warn.startswith("missing model: "):
                            # "missing model: flux1-dev.sft (UNETLoader.unet_name)"
                            model_part = warn[15:]
                            paren_idx = model_part.find(" (")
                            model_name = model_part[:paren_idx] if paren_idx > 0 else model_part
                            item = f"missing model {model_name}"
                        else:
                            item = warn
                        lines.append(
                            f"\033[35m\u2502\033[0m    "
                            f"\033[33m\u26a0 {item}\033[0m"
                        )
            else:
                lines.append(
                    f"\033[35m\u2502\033[0m  \033[33m\u26a0\033[0m "
                    f"\033[2m{os.path.basename(w.file_path)}\033[0m: "
                    f"empty slug"
                )
            # Blank separator between tools for readability
            lines.append(f"\033[35m\u2502\033[0m")

        # Show errors
        for err in sr.errors:
            lines.append(f"\033[35m\u2502\033[0m  {err}")

    # Duplicates
    for slug, kept_path, skipped_path in duplicates:
        lines.append(
            f"\033[35m\u2502\033[0m  \033[33m\u26a0 Duplicate: "
            f"\033[1m{slug}\033[0m\n"
            f"\033[35m\u2502\033[0m    kept: {kept_path}\n"
            f"\033[35m\u2502\033[0m    skipped: {skipped_path}\033[0m"
        )

    # Summary
    valid = [w for w in all_workflows if w.tool_info.get("slug")]
    ready = [w for w in valid if not w.warnings]
    skipped = [w for w in valid if w.warnings]
    if ready:
        summary = f"\033[32m\u2714 {len(ready)} tool(s) registered\033[0m"
    elif valid:
        summary = (
            f"\033[33m\u26a0 0 tools registered\033[0m — "
            f"all {len(valid)} workflow(s) have missing models/nodes"
        )
    else:
        summary = (
            "\033[33m\u26a0 0 tools found\033[0m — "
            "no workflows with StimmaToolInfo + valid slug"
        )
    lines.append(f"\033[35m\u2502\033[0m  {summary}")
    if skipped:
        lines.append(
            f"\033[35m\u2502\033[0m  \033[33m\u26a0 {len(skipped)} skipped\033[0m"
            f" — install the missing models/nodes above to enable them"
        )

    # Assemble box
    top = f"\033[35m\u2552{BAR * 2} Stimma Workflow Scan {BAR * (W - 23)}\033[0m"
    bot = f"\033[35m\u2558{BAR * W}\033[0m"
    body = "\n".join(lines)

    logger.info(f"\n{top}\n{body}\n{bot}")

    return all_workflows
