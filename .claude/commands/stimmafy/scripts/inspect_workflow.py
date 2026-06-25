#!/usr/bin/env python3
"""Inspect a Stimma workflow and output a structured summary of all Stimma nodes.

Reads a workflow JSON and shows all Stimma nodes with their labeled widget_values
(including array indices for easy editing), wiring targets, and layout groups.

Usage:
    python3 inspect_workflow.py <workflow.json>
    python3 inspect_workflow.py <workflow.json> --json
"""

import json
import sys
import os
import argparse


STIMMA_FIELD_TYPES = {
    "StimmaPromptParam", "StimmaImageParam", "StimmaMaskParam",
    "StimmaImagesParam", "StimmaVideoParam", "StimmaSeedParam",
    "StimmaResolutionParam", "StimmaDurationToFrames",
}

STIMMA_PARAM_TYPES = {
    "StimmaIntParam", "StimmaFloatParam", "StimmaStringParam",
    "StimmaDropdownParam", "StimmaBoolParam",
}

STIMMA_LORA_TYPES = {
    "StimmaLoraLoader", "StimmaPairedLoraLoader",
}

STIMMA_CHECKPOINT_TYPES = {
    "StimmaCheckpointLoader",
}

STIMMA_OUTPUT_TYPES = {
    "StimmaImageOutput", "StimmaVideoOutput",
}

STIMMA_LAYOUT_TYPES = {
    "StimmaLayoutGroup",
}


def load_workflow(path):
    with open(path) as f:
        return json.load(f)


def build_indexes(workflow):
    """Build node-by-id and link-by-id lookup dicts."""
    nodes_by_id = {}
    for node in workflow.get("nodes", []):
        nodes_by_id[node["id"]] = node

    links_by_id = {}
    for link in workflow.get("links", []):
        # [link_id, src_node, src_slot, dst_node, dst_slot, type]
        lid, src_node, src_slot, dst_node, dst_slot, link_type = link
        links_by_id[lid] = {
            "src_node": src_node, "src_slot": src_slot,
            "dst_node": dst_node, "dst_slot": dst_slot,
            "type": link_type,
        }

    return nodes_by_id, links_by_id


def get_widget_values(node):
    """Extract labeled widget values as list of (index, name, value) tuples.

    Widget inputs are those with a "widget" key in the node's inputs array.
    The widgets_values array maps 1:1 to these widget inputs in order.
    """
    widget_names = []
    for inp in node.get("inputs", []):
        if "widget" in inp:
            widget_names.append(inp["name"])

    values = node.get("widgets_values", [])
    result = []
    for i, name in enumerate(widget_names):
        result.append((i, name, values[i] if i < len(values) else None))

    # Extra values beyond the widget inputs (e.g. control_after_generate)
    for i in range(len(widget_names), len(values)):
        result.append((i, f"_extra_{i}", values[i]))

    return result


def wv_get(widget_values, name, default=None):
    """Get a single widget value by name."""
    for _, n, v in widget_values:
        if n == name:
            return v
    return default


def wv_to_dict(widget_values):
    """Convert widget values list to a dict (name → value)."""
    return {n: v for _, n, v in widget_values}


def resolve_output_wiring(node, links_by_id, nodes_by_id):
    """For each output slot, find destination nodes."""
    wiring = []
    for slot_idx, output in enumerate(node.get("outputs", [])):
        link_ids = output.get("links") or []
        targets = []
        for lid in link_ids:
            link = links_by_id.get(lid)
            if not link:
                continue
            dst_node = nodes_by_id.get(link["dst_node"])
            if not dst_node:
                continue
            dst_inputs = dst_node.get("inputs", [])
            dst_input_name = "?"
            if link["dst_slot"] < len(dst_inputs):
                dst_input_name = dst_inputs[link["dst_slot"]].get("name", "?")
            targets.append({
                "node_id": link["dst_node"],
                "class_type": dst_node.get("type", "?"),
                "input_name": dst_input_name,
            })
        wiring.append({
            "slot": slot_idx,
            "name": output.get("name", f"output_{slot_idx}"),
            "type": output.get("type", "?"),
            "targets": targets,
        })
    return wiring


def resolve_input_wiring(node, links_by_id, nodes_by_id):
    """For link-only inputs (non-widget), find source nodes."""
    wiring = []
    for inp in node.get("inputs", []):
        lid = inp.get("link")
        if lid is None or "widget" in inp:
            continue
        link = links_by_id.get(lid)
        if not link:
            continue
        src_node = nodes_by_id.get(link["src_node"])
        if not src_node:
            continue
        wiring.append({
            "input_name": inp["name"],
            "input_type": inp.get("type", "?"),
            "source_node_id": link["src_node"],
            "source_class_type": src_node.get("type", "?"),
        })
    return wiring


def categorize_nodes(workflow):
    """Split workflow nodes into Stimma categories."""
    cats = {"tool_info": [], "inputs": [], "params": [],
            "lora": [], "checkpoint": [], "outputs": [], "layout": []}
    for node in workflow.get("nodes", []):
        t = node.get("type", "")
        if t == "StimmaToolInfo":
            cats["tool_info"].append(node)
        elif t in STIMMA_FIELD_TYPES:
            cats["inputs"].append(node)
        elif t in STIMMA_PARAM_TYPES:
            cats["params"].append(node)
        elif t in STIMMA_LORA_TYPES:
            cats["lora"].append(node)
        elif t in STIMMA_CHECKPOINT_TYPES:
            cats["checkpoint"].append(node)
        elif t in STIMMA_OUTPUT_TYPES:
            cats["outputs"].append(node)
        elif t in STIMMA_LAYOUT_TYPES:
            cats["layout"].append(node)
    return cats


def get_display_name(node, wv):
    """Get a human-readable display name for a Stimma node."""
    name = wv_get(wv, "name")
    if name:
        return name
    node_type = node.get("type", "")
    if node_type == "StimmaResolutionParam":
        return "resolution"
    if node_type == "StimmaDurationToFrames":
        return "duration_to_frames"
    return node_type.replace("Stimma", "").lower()


def format_wv_line(wv_list, skip_lora_slots=False):
    """Format widget values as indexed, labeled string.

    Example: [0:name="steps", 1:value=9, 2:minimum=1, ...]
    """
    parts = []
    for idx, name, value in wv_list:
        if name.startswith("_extra_"):
            continue
        if skip_lora_slots and (name.startswith("lora_") or name.startswith("strength_")
                                or name.startswith("lora_low_")):
            continue
        display_val = value
        if isinstance(value, str) and len(value) > 60:
            display_val = value[:57] + "..."
        parts.append(f"{idx}:{name}={json.dumps(display_val)}")
    return "[" + ", ".join(parts) + "]"


def format_output_wiring_lines(wiring):
    """Format output wiring as indented text lines."""
    lines = []
    for out in wiring:
        if not out["targets"]:
            continue
        targets_str = ", ".join(
            f"node {t['node_id']} ({t['class_type']}).{t['input_name']}"
            for t in out["targets"]
        )
        lines.append(f"    output {out['slot']} ({out['type']}) -> {targets_str}")
    return lines


def format_input_wiring_lines(wiring):
    """Format input wiring as indented text lines."""
    return [
        f"    {w['input_name']} <- node {w['source_node_id']} ({w['source_class_type']})"
        for w in wiring
    ]


def format_text(workflow, filepath):
    """Format the inspection as human-readable text."""
    nodes_by_id, links_by_id = build_indexes(workflow)
    cats = categorize_nodes(workflow)
    lines = []

    lines.append(f"=== Stimma Workflow: {os.path.basename(filepath)} ===")
    lines.append("")

    # Tool Info
    for node in cats["tool_info"]:
        wv = get_widget_values(node)
        lines.append(f"Tool Info [node {node['id']}]:")
        for key in ["slug", "display_name", "task_types", "badges",
                     "description", "model_vendor", "model"]:
            val = wv_get(wv, key)
            if val is not None and val != "":
                if isinstance(val, str) and len(val) > 80:
                    val = val[:77] + "..."
                lines.append(f"  {key}: {val}")
        lines.append("")

    # Inputs
    if cats["inputs"]:
        lines.append("Inputs:")
        sorted_inputs = sorted(
            cats["inputs"],
            key=lambda n: wv_get(get_widget_values(n), "ui_order", 0),
        )
        for node in sorted_inputs:
            wv = get_widget_values(node)
            name = get_display_name(node, wv)
            lines.append(f"  {name} [node {node['id']}, {node['type']}]:")
            lines.append(f"    widgets_values: {format_wv_line(wv)}")
            lines.extend(format_output_wiring_lines(
                resolve_output_wiring(node, links_by_id, nodes_by_id)))
            lines.append("")

    # Parameters
    if cats["params"]:
        lines.append("Parameters:")
        sorted_params = sorted(
            cats["params"],
            key=lambda n: wv_get(get_widget_values(n), "ui_order", 0),
        )
        for node in sorted_params:
            wv = get_widget_values(node)
            name = get_display_name(node, wv)
            lines.append(f"  {name} [node {node['id']}, {node['type']}]:")
            lines.append(f"    widgets_values: {format_wv_line(wv)}")
            lines.extend(format_output_wiring_lines(
                resolve_output_wiring(node, links_by_id, nodes_by_id)))
            lines.append("")

    # LoRA
    for node in cats["lora"]:
        wv = get_widget_values(node)
        lines.append(f"LoRA [node {node['id']}, {node['type']}]:")
        lines.append(f"  path_filter: {json.dumps(wv_get(wv, 'path_filter', ''))}")
        lines.append(f"  ui_order: {wv_get(wv, 'ui_order', 0)}")
        # Show active (non-None) lora slots
        active = []
        d = wv_to_dict(wv)
        for i in range(1, 11):
            lora_key = f"lora_{i}"
            strength_key = f"strength_{i}"
            lora_val = d.get(lora_key, "None")
            if lora_val != "None":
                active.append(f"  slot {i}: {lora_val} (strength={d.get(strength_key, 1.0)})")
        if active:
            lines.append("  Active LoRAs:")
            lines.extend(active)
        else:
            lines.append("  No active LoRAs")
        lines.extend(format_input_wiring_lines(
            resolve_input_wiring(node, links_by_id, nodes_by_id)))
        lines.extend(format_output_wiring_lines(
            resolve_output_wiring(node, links_by_id, nodes_by_id)))
        lines.append("")

    # Checkpoint loader
    for node in cats["checkpoint"]:
        wv = get_widget_values(node)
        lines.append(f"Checkpoint [node {node['id']}, {node['type']}]:")
        lines.append(f"  ckpt_name: {json.dumps(wv_get(wv, 'ckpt_name', ''))}")
        lines.append(f"  path_filter: {json.dumps(wv_get(wv, 'path_filter', ''))}")
        lines.append(f"  ui_order: {wv_get(wv, 'ui_order', 0)}")
        lines.extend(format_output_wiring_lines(
            resolve_output_wiring(node, links_by_id, nodes_by_id)))
        lines.append("")

    # Outputs
    if cats["outputs"]:
        lines.append("Outputs:")
        for node in cats["outputs"]:
            wv = get_widget_values(node)
            lines.append(f"  {node['type']} [node {node['id']}]:")
            lines.append(f"    widgets_values: {format_wv_line(wv)}")
            lines.extend(format_input_wiring_lines(
                resolve_input_wiring(node, links_by_id, nodes_by_id)))
            lines.append("")

    # Layout
    if cats["layout"]:
        lines.append("Layout:")
        sorted_layout = sorted(
            cats["layout"],
            key=lambda n: wv_get(get_widget_values(n), "ui_order", 0),
        )
        for node in sorted_layout:
            wv = get_widget_values(node)
            label = wv_get(wv, "group_label", "?")
            collapsed = wv_get(wv, "collapsed", False)
            ui_order = wv_get(wv, "ui_order", 0)
            param_names = wv_get(wv, "param_names", "")
            state = "collapsed" if collapsed else "expanded"
            lines.append(f'  "{label}" [node {node["id"]}] ({state}, ui_order={ui_order}):')
            if param_names:
                names = [n.strip() for n in param_names.split("\n") if n.strip()]
                lines.append(f"    {', '.join(names)}")
            lines.append("")

    return "\n".join(lines)


def build_json_report(workflow, filepath):
    """Build a structured JSON report."""
    nodes_by_id, links_by_id = build_indexes(workflow)
    cats = categorize_nodes(workflow)
    report = {
        "filename": os.path.basename(filepath),
        "tool_info": [],
        "inputs": [],
        "params": [],
        "lora": [],
        "checkpoint": [],
        "outputs": [],
        "layout": [],
    }

    for node in cats["tool_info"]:
        wv = get_widget_values(node)
        report["tool_info"].append({
            "node_id": node["id"],
            "type": node["type"],
            "widgets": [{"index": i, "name": n, "value": v} for i, n, v in wv
                        if not n.startswith("_extra_")],
        })

    for cat_key in ("inputs", "params"):
        for node in cats[cat_key]:
            wv = get_widget_values(node)
            report[cat_key].append({
                "node_id": node["id"],
                "type": node["type"],
                "name": get_display_name(node, wv),
                "widgets": [{"index": i, "name": n, "value": v} for i, n, v in wv
                            if not n.startswith("_extra_")],
                "output_wiring": resolve_output_wiring(node, links_by_id, nodes_by_id),
            })

    for node in cats["lora"]:
        wv = get_widget_values(node)
        d = wv_to_dict(wv)
        active_loras = {}
        for i in range(1, 11):
            lname = d.get(f"lora_{i}", "None")
            if lname != "None":
                active_loras[f"lora_{i}"] = lname
                active_loras[f"strength_{i}"] = d.get(f"strength_{i}", 1.0)
        report["lora"].append({
            "node_id": node["id"],
            "type": node["type"],
            "path_filter": d.get("path_filter", ""),
            "ui_order": d.get("ui_order", 0),
            "active_loras": active_loras,
            "input_wiring": resolve_input_wiring(node, links_by_id, nodes_by_id),
            "output_wiring": resolve_output_wiring(node, links_by_id, nodes_by_id),
        })

    for node in cats["checkpoint"]:
        wv = get_widget_values(node)
        d = wv_to_dict(wv)
        report["checkpoint"].append({
            "node_id": node["id"],
            "type": node["type"],
            "ckpt_name": d.get("ckpt_name", ""),
            "path_filter": d.get("path_filter", ""),
            "ui_order": d.get("ui_order", 0),
            "output_wiring": resolve_output_wiring(node, links_by_id, nodes_by_id),
        })

    for node in cats["outputs"]:
        wv = get_widget_values(node)
        report["outputs"].append({
            "node_id": node["id"],
            "type": node["type"],
            "widgets": [{"index": i, "name": n, "value": v} for i, n, v in wv
                        if not n.startswith("_extra_")],
            "input_wiring": resolve_input_wiring(node, links_by_id, nodes_by_id),
        })

    for node in cats["layout"]:
        wv = get_widget_values(node)
        d = wv_to_dict(wv)
        param_names_raw = d.get("param_names", "")
        report["layout"].append({
            "node_id": node["id"],
            "group_label": d.get("group_label", ""),
            "collapsed": d.get("collapsed", False),
            "ui_order": d.get("ui_order", 0),
            "param_names": [n.strip() for n in param_names_raw.split("\n") if n.strip()],
        })

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a Stimma workflow and summarize all Stimma nodes.",
    )
    parser.add_argument("workflow", help="Path to the workflow JSON file")
    parser.add_argument("--json", action="store_true",
                        help="Output as structured JSON instead of text")
    args = parser.parse_args()

    if not os.path.exists(args.workflow):
        print(f"Error: file not found: {args.workflow}", file=sys.stderr)
        sys.exit(1)

    workflow = load_workflow(args.workflow)

    has_stimma = any(
        node.get("type", "").startswith("Stimma")
        for node in workflow.get("nodes", [])
    )
    if not has_stimma:
        print("Warning: no Stimma nodes found in this workflow.", file=sys.stderr)

    if args.json:
        report = build_json_report(workflow, args.workflow)
        print(json.dumps(report, indent=2))
    else:
        print(format_text(workflow, args.workflow))


if __name__ == "__main__":
    main()
