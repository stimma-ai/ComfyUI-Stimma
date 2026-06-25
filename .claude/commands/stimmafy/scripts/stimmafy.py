#!/usr/bin/env python3
"""Apply Stimma nodes to a ComfyUI workflow based on a plan.

Reads a UI-format workflow JSON and a plan JSON, adds Stimma nodes,
wires them up, and outputs the modified workflow.

Usage:
    python3 stimmafy.py <workflow.json> <plan.json> -o <output.json>
"""

import json
import sys
import argparse
import copy
import uuid
from pathlib import Path


class WorkflowModifier:
    """Modifies a ComfyUI UI-format workflow by adding nodes and links."""

    def __init__(self, workflow):
        if "nodes" not in workflow:
            raise ValueError(
                "Workflow must be in UI format (with 'nodes' array). "
                "Got API format (flat dict). Convert to UI format first."
            )
        self.workflow = copy.deepcopy(workflow)
        self.next_node_id = workflow.get("last_node_id", 0) + 1
        self.next_link_id = workflow.get("last_link_id", 0) + 1
        self.nodes_by_id = {n["id"]: n for n in self.workflow["nodes"]}
        self.named_nodes = {}  # name -> node dict, for cross-referencing in plans

    def _alloc_node_id(self):
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def _alloc_link_id(self):
        lid = self.next_link_id
        self.next_link_id += 1
        return lid

    def _add_node(self, node):
        """Add a node to the workflow."""
        self.workflow["nodes"].append(node)
        self.nodes_by_id[node["id"]] = node
        return node

    def _add_link(self, src_node, src_slot, dst_node, dst_slot, link_type):
        """Add a link and update node input/output references."""
        lid = self._alloc_link_id()
        self.workflow.setdefault("links", []).append(
            [lid, src_node, src_slot, dst_node, dst_slot, link_type]
        )

        # Update destination node's input link
        dst = self.nodes_by_id.get(dst_node)
        if dst:
            inputs = dst.get("inputs", [])
            if dst_slot < len(inputs):
                inputs[dst_slot]["link"] = lid

        # Update source node's output links
        src = self.nodes_by_id.get(src_node)
        if src:
            outputs = src.get("outputs", [])
            if src_slot < len(outputs):
                output = outputs[src_slot]
                if output.get("links") is None:
                    output["links"] = []
                output["links"].append(lid)

        return lid

    def _find_input_slot(self, node_id, input_name):
        """Find the slot index for a named input on a node."""
        node = self.nodes_by_id.get(node_id)
        if not node:
            return None
        for i, inp in enumerate(node.get("inputs", [])):
            if inp.get("name") == input_name:
                return i
        return None

    def _find_output_slot(self, node_id, output_name):
        """Find the slot index for a named output on a node."""
        node = self.nodes_by_id.get(node_id)
        if not node:
            return None
        for i, out in enumerate(node.get("outputs", [])):
            if out.get("name") == output_name:
                return i
        return None

    def _ensure_input_slot(self, node_id, input_name, input_type):
        """Ensure a node has an input slot for the given name. Create if missing."""
        slot = self._find_input_slot(node_id, input_name)
        if slot is not None:
            return slot

        node = self.nodes_by_id.get(node_id)
        if not node:
            return None

        inputs = node.setdefault("inputs", [])
        new_slot = len(inputs)
        inputs.append({
            "name": input_name,
            "type": input_type,
            "widget": {"name": input_name},
            "link": None,
        })
        return new_slot

    def _make_widget_inputs(self, input_specs):
        """Create input slot list from a list of (name, type) tuples."""
        inputs = []
        for name, itype in input_specs:
            entry = {
                "name": name,
                "type": itype,
                "widget": {"name": name},
                "link": None,
            }
            inputs.append(entry)
        return inputs

    def _make_connection_input(self, name, itype, link=None):
        """Create a connection-type input (MODEL, CLIP, IMAGE, etc.)."""
        return {
            "name": name,
            "type": itype,
            "link": link,
        }

    def _make_output(self, name, otype, links=None):
        """Create an output slot."""
        return {
            "name": name,
            "type": otype,
            "links": links or [],
            "slot_index": 0,  # Will be set by position in outputs array
        }

    def _base_node(self, ntype, pos, size):
        """Create a base node structure."""
        nid = self._alloc_node_id()
        return {
            "id": nid,
            "type": ntype,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {"Node name for S&R": ntype},
            "widgets_values": [],
        }

    def _wire_output_to_input(self, src_node_id, src_slot, dst_node_id, input_name, link_type):
        """Wire a source output slot to a destination input by name."""
        dst_slot = self._ensure_input_slot(dst_node_id, input_name, link_type)
        if dst_slot is None:
            print(f"Warning: Could not find/create input '{input_name}' on node {dst_node_id}",
                  file=sys.stderr)
            return None
        return self._add_link(src_node_id, src_slot, dst_node_id, dst_slot, link_type)

    def _intercept_link(self, src_node_id, src_output_name, new_node_id, new_input_slot,
                        new_output_slot, link_type):
        """Insert a new node between a source and all its destinations.

        Finds all outgoing links from src_node_id:src_output_name, rewires them
        to come from new_node_id:new_output_slot instead, and creates a new link
        from src_node_id to new_node_id:new_input_slot.
        """
        src_node = self.nodes_by_id.get(src_node_id)
        if not src_node:
            return

        src_slot = self._find_output_slot(src_node_id, src_output_name)
        if src_slot is None:
            # Try by index 0 if name not found
            src_slot = 0

        # Find all existing links from this source
        links_to_rewire = []
        for link in self.workflow.get("links", []):
            if link[1] == src_node_id and link[2] == src_slot:
                links_to_rewire.append(link)

        # Update source node's output to only point to new node
        src_outputs = src_node.get("outputs", [])
        if src_slot < len(src_outputs):
            src_outputs[src_slot]["links"] = []

        # Create link from source to new node
        new_link_id = self._alloc_link_id()
        self.workflow["links"].append(
            [new_link_id, src_node_id, src_slot, new_node_id, new_input_slot, link_type]
        )
        if src_slot < len(src_outputs):
            src_outputs[src_slot]["links"].append(new_link_id)

        # Update new node's input
        new_node = self.nodes_by_id[new_node_id]
        new_inputs = new_node.get("inputs", [])
        if new_input_slot < len(new_inputs):
            new_inputs[new_input_slot]["link"] = new_link_id

        # Rewire existing destinations to come from new node
        new_outputs = new_node.get("outputs", [])
        for link in links_to_rewire:
            # Update link source to be new node
            link[1] = new_node_id
            link[2] = new_output_slot
            # Add to new node's output links
            if new_output_slot < len(new_outputs):
                if new_outputs[new_output_slot].get("links") is None:
                    new_outputs[new_output_slot]["links"] = []
                new_outputs[new_output_slot]["links"].append(link[0])

    # --- Node reference and wiring helpers ---

    def _resolve_node_ref(self, ref):
        """Resolve a node reference to (node_id, slot).

        ref can be:
        - {"ref": "param_name"} -> looks up named node, slot defaults to 0
        - {"ref": "param_name", "slot": 1} -> looks up named node with explicit slot
        - {"node_id": 42} or {"node_id": 42, "slot": 0} -> direct reference
        """
        if "ref" in ref:
            node = self.named_nodes.get(ref["ref"])
            if node:
                return node["id"], ref.get("slot", 0)
            print(f"Warning: Named node '{ref['ref']}' not found", file=sys.stderr)
            return None, None
        return ref["node_id"], ref.get("slot", 0)

    def _wire_to_targets(self, src_node_id, src_slot, wire_to, link_type):
        """Wire output to one or more targets. Supports single dict or list of dicts."""
        if not wire_to:
            return
        targets = wire_to if isinstance(wire_to, list) else [wire_to]
        for target in targets:
            node_id = target["node_id"]
            input_name = target["input_name"]
            lt = target.get("link_type", link_type)
            self._wire_output_to_input(src_node_id, src_slot, node_id, input_name, lt)

    # --- Pipeline setup methods ---

    def set_node_mode(self, node_id, mode):
        """Set a node's mode. 0=active, 4=muted/bypassed."""
        node = self.nodes_by_id.get(node_id)
        if node:
            node["mode"] = mode
        else:
            print(f"Warning: Node {node_id} not found for mode change", file=sys.stderr)

    def delete_node(self, node_id):
        """Remove a node and all its links from the workflow."""
        node = self.nodes_by_id.get(node_id)
        if not node:
            print(f"Warning: Node {node_id} not found for deletion", file=sys.stderr)
            return

        # Remove links connected to this node
        self.workflow["links"] = [
            link for link in self.workflow.get("links", [])
            if link[1] != node_id and link[3] != node_id
        ]

        # Remove from nodes array
        self.workflow["nodes"] = [n for n in self.workflow["nodes"] if n["id"] != node_id]
        del self.nodes_by_id[node_id]

    def bypass_node(self, node_id, input_slot=0, output_slot=0):
        """Remove a node from a chain by reconnecting upstream directly to downstream.

        Useful for removing LoRA loaders or other passthrough nodes without breaking
        the model/clip chain. Only works for nodes where input_slot and output_slot
        carry the same type (e.g., MODEL in → MODEL out).
        """
        node = self.nodes_by_id.get(node_id)
        if not node:
            print(f"Warning: Node {node_id} not found for bypass", file=sys.stderr)
            return

        # Find upstream source
        inputs = node.get("inputs", [])
        upstream_src = None
        if input_slot < len(inputs):
            input_link_id = inputs[input_slot].get("link")
            if input_link_id is not None:
                for link in self.workflow.get("links", []):
                    if link[0] == input_link_id:
                        upstream_src = (link[1], link[2])
                        break

        # Collect downstream destinations before deletion
        downstream_dests = []
        for link in self.workflow.get("links", []):
            if link[1] == node_id and link[2] == output_slot:
                downstream_dests.append((link[3], link[4], link[5]))

        # Remove the node and all its links
        self.delete_node(node_id)

        # Reconnect upstream to all downstreams
        if upstream_src:
            for dst_node_id, dst_slot, link_type in downstream_dests:
                if dst_node_id in self.nodes_by_id:
                    self._add_link(upstream_src[0], upstream_src[1],
                                   dst_node_id, dst_slot, link_type)

    # --- Stimma node creation methods ---

    def add_tool_info(self, slug, display_name, task_types="text-to-image",
                      badges="", description="", model_vendor="", model="",
                      pos=(-800, -400)):
        """Add StimmaToolInfo node.

        Widget order matches the current node schema:
        [slug, display_name, task_types, badges, description] plus the optional
        [model_vendor, model] pair. `task_types` is a comma-separated STRING.
        """
        node = self._base_node("StimmaToolInfo", pos, [480, 276])
        widget_specs = [
            ("slug", "STRING"), ("display_name", "STRING"),
            ("task_types", "STRING"), ("badges", "STRING"),
            ("description", "STRING"),
        ]
        wv = [slug, display_name, task_types, badges, description]
        # model_vendor / model are optional inputs (shape 7) — emit the pair only
        # when either is provided, matching how bundled workflows omit them.
        if model_vendor or model:
            widget_specs.append(("model_vendor", "STRING"))
            widget_specs.append(("model", "STRING"))
            wv.append(model_vendor)
            wv.append(model)
        node["inputs"] = self._make_widget_inputs(widget_specs)
        if model_vendor or model:
            node["inputs"][-1]["shape"] = 7
            node["inputs"][-2]["shape"] = 7
        node["outputs"] = []
        node["widgets_values"] = wv
        return self._add_node(node)

    def add_prompt_input(self, name, default_text, required=True, ui_order=0,
                         ui_description="", pos=(-800, 0), wire_to=None):
        """Add StimmaPromptParam node and optionally wire it."""
        node = self._base_node("StimmaPromptParam", pos, [480, 360])
        node["inputs"] = self._make_widget_inputs([
            ("name", "STRING"), ("default_text", "STRING"),
            ("required", "BOOLEAN"), ("ui_order", "INT"),
            ("ui_description", "STRING"),
        ])
        node["outputs"] = [self._make_output("text", "STRING")]
        node["widgets_values"] = [name, default_text, required, ui_order, ui_description]
        self._add_node(node)
        self._wire_to_targets(node["id"], 0, wire_to, "STRING")
        return node

    def add_seed_input(self, name="seed", value=0, ui_order=80, pos=(-800, 400), wire_to=None):
        """Add StimmaSeedParam node."""
        node = self._base_node("StimmaSeedParam", pos, [384, 156])
        node["inputs"] = self._make_widget_inputs([
            ("name", "STRING"), ("value", "INT"), ("ui_order", "INT"),
        ])
        node["outputs"] = [self._make_output("seed", "INT")]
        node["widgets_values"] = [name, value, ui_order]
        self._add_node(node)
        self._wire_to_targets(node["id"], 0, wire_to, "INT")
        return node

    def add_resolution_input(self, width=1024, height=1024, min_size=512, max_size=2048,
                             step=64, supported_resolutions="", ui_order=1,
                             pos=(-800, 600), wire_to=None):
        """Add StimmaResolutionParam node."""
        node = self._base_node("StimmaResolutionParam", pos, [480, 360])
        node["inputs"] = self._make_widget_inputs([
            ("width", "INT"), ("height", "INT"), ("min_size", "INT"),
            ("max_size", "INT"), ("step", "INT"),
            ("supported_resolutions", "STRING"), ("ui_order", "INT"),
        ])
        node["outputs"] = [
            self._make_output("width", "INT"),
            self._make_output("height", "INT"),
        ]
        node["outputs"][0]["slot_index"] = 0
        node["outputs"][1]["slot_index"] = 1
        node["widgets_values"] = [width, height, min_size, max_size, step,
                                  supported_resolutions, ui_order]
        self._add_node(node)

        if wire_to:
            targets = wire_to if isinstance(wire_to, list) else [wire_to]
            for target in targets:
                self._wire_output_to_input(
                    node["id"], 0, target["node_id"], target.get("width_input", "width"), "INT"
                )
                self._wire_output_to_input(
                    node["id"], 1, target["node_id"], target.get("height_input", "height"), "INT"
                )
        return node

    def add_int_param(self, name, value, minimum=1, maximum=100, step=1,
                      ui_control="input", ui_order=20, ui_description="",
                      pos=(-800, 800), wire_to=None):
        """Add StimmaIntParam node."""
        node = self._base_node("StimmaIntParam", pos, [384, 355])
        node["inputs"] = self._make_widget_inputs([
            ("name", "STRING"), ("value", "INT"), ("minimum", "INT"),
            ("maximum", "INT"), ("step", "INT"), ("ui_control", "COMBO"),
            ("ui_order", "INT"), ("ui_description", "STRING"),
        ])
        node["outputs"] = [self._make_output("value", "INT")]
        node["widgets_values"] = [name, value, minimum, maximum, step,
                                  ui_control, ui_order, ui_description]
        self._add_node(node)
        self._wire_to_targets(node["id"], 0, wire_to, "INT")
        return node

    def add_float_param(self, name, value, minimum=0.0, maximum=1.0, step=0.01,
                        ui_control="input", ui_order=30, ui_description="",
                        pos=(-800, 1200), wire_to=None):
        """Add StimmaFloatParam node."""
        node = self._base_node("StimmaFloatParam", pos, [384, 355])
        node["inputs"] = self._make_widget_inputs([
            ("name", "STRING"), ("value", "FLOAT"), ("minimum", "FLOAT"),
            ("maximum", "FLOAT"), ("step", "FLOAT"), ("ui_control", "COMBO"),
            ("ui_order", "INT"), ("ui_description", "STRING"),
        ])
        node["outputs"] = [self._make_output("value", "FLOAT")]
        node["widgets_values"] = [name, value, minimum, maximum, step,
                                  ui_control, ui_order, ui_description]
        self._add_node(node)
        self._wire_to_targets(node["id"], 0, wire_to, "FLOAT")
        return node

    def add_string_param(self, name, value="", enum_values="",
                         ui_control="textarea", ui_order=20, ui_description="",
                         pos=(-800, 1400), wire_to=None):
        """Add StimmaStringParam node."""
        node = self._base_node("StimmaStringParam", pos, [384, 355])
        node["inputs"] = self._make_widget_inputs([
            ("name", "STRING"), ("value", "STRING"), ("enum_values", "STRING"),
            ("ui_control", "COMBO"), ("ui_order", "INT"),
            ("ui_description", "STRING"),
        ])
        node["outputs"] = [self._make_output("value", "STRING")]
        node["widgets_values"] = [name, value, enum_values, ui_control,
                                  ui_order, ui_description]
        self._add_node(node)
        self._wire_to_targets(node["id"], 0, wire_to, "STRING")
        return node

    def add_dropdown_param(self, name, value, ui_order=40, ui_description="",
                           pos=(-800, 1600), wire_to=None):
        """Add StimmaDropdownParam node. Enum is auto-resolved at discovery time."""
        node = self._base_node("StimmaDropdownParam", pos, [384, 228])
        node["inputs"] = self._make_widget_inputs([
            ("name", "STRING"), ("value", "STRING"),
            ("ui_order", "INT"), ("ui_description", "STRING"),
        ])
        # StimmaDropdownParam outputs "*" (any type) but we use STRING for the link
        node["outputs"] = [self._make_output("value", "*")]
        node["widgets_values"] = [name, value, ui_order, ui_description]
        self._add_node(node)
        # StimmaDropdownParam outputs "*" (any type) but we link as STRING
        self._wire_to_targets(node["id"], 0, wire_to, "STRING")
        return node

    def add_bool_param(self, name, value=False, ui_order=50, ui_description="",
                       pos=(-800, 1800), wire_to=None):
        """Add StimmaBoolParam node."""
        node = self._base_node("StimmaBoolParam", pos, [384, 180])
        node["inputs"] = self._make_widget_inputs([
            ("name", "STRING"), ("value", "BOOLEAN"),
            ("ui_order", "INT"), ("ui_description", "STRING"),
        ])
        node["outputs"] = [self._make_output("value", "BOOLEAN")]
        node["widgets_values"] = [name, value, ui_order, ui_description]
        self._add_node(node)
        self._wire_to_targets(node["id"], 0, wire_to, "BOOLEAN")
        return node

    def add_duration_to_frames(self, frame_step=4, pos=(-400, 600),
                               duration_source=None, fps_source=None, wire_to=None):
        """Add StimmaDurationToFrames node.

        Converts duration (seconds) + fps → frame count aligned to frame_step.
        duration_source/fps_source: {"ref": "name"} or {"node_id": N, "slot": S}
        """
        node = self._base_node("StimmaDurationToFrames", pos, [384, 200])
        node["inputs"] = [
            self._make_connection_input("duration", "FLOAT"),
            self._make_connection_input("fps", "INT"),
        ]
        node["inputs"].extend(self._make_widget_inputs([
            ("frame_step", "INT"),
        ]))
        node["outputs"] = [self._make_output("frames", "INT")]
        # Widget-type inputs (FLOAT, INT) consume widgets_values slots even when
        # connected as links. Include placeholder values for duration and fps.
        node["widgets_values"] = [0.0, 25, frame_step]
        self._add_node(node)

        # Wire duration input
        if duration_source:
            src_id, src_slot = self._resolve_node_ref(duration_source)
            if src_id is not None:
                self._add_link(src_id, src_slot, node["id"], 0, "FLOAT")

        # Wire fps input
        if fps_source:
            src_id, src_slot = self._resolve_node_ref(fps_source)
            if src_id is not None:
                self._add_link(src_id, src_slot, node["id"], 1, "INT")

        self._wire_to_targets(node["id"], 0, wire_to, "INT")
        return node

    def add_paired_lora_loader(self, path_filter="", ui_order=50, pos=(-800, 2000),
                               high_model_source=None, low_model_source=None):
        """Add StimmaPairedLoraLoader for dual-model architectures (e.g., WAN 2.2).

        Intercepts both high_noise_model and low_noise_model chains.
        """
        node = self._base_node("StimmaPairedLoraLoader", pos, [480, 806])

        node["inputs"] = [
            self._make_connection_input("high_noise_model", "MODEL"),
            self._make_connection_input("low_noise_model", "MODEL"),
        ]
        node["inputs"].extend(self._make_widget_inputs([
            ("path_filter", "STRING"), ("ui_order", "INT"),
        ]))
        # 10 LoRA slots (optional, shape 7)
        for i in range(1, 11):
            lora_input = {"name": f"lora_{i}", "type": "COMBO",
                          "widget": {"name": f"lora_{i}"}, "link": None, "shape": 7}
            strength_input = {"name": f"strength_{i}", "type": "FLOAT",
                              "widget": {"name": f"strength_{i}"}, "link": None, "shape": 7}
            node["inputs"].extend([lora_input, strength_input])

        node["outputs"] = [
            self._make_output("high_noise_model", "MODEL"),
            self._make_output("low_noise_model", "MODEL"),
        ]
        node["outputs"][0]["slot_index"] = 0
        node["outputs"][1]["slot_index"] = 1

        wv = [path_filter, ui_order]
        for _ in range(10):
            wv.extend(["None", 1])
        node["widgets_values"] = wv

        self._add_node(node)

        # Intercept high_noise_model chain
        if high_model_source:
            src_node = self.nodes_by_id.get(high_model_source["node_id"])
            if src_node:
                src_output_name = src_node["outputs"][high_model_source.get("slot", 0)].get("name", "MODEL")
                self._intercept_link(
                    high_model_source["node_id"], src_output_name,
                    node["id"], 0, 0, "MODEL"
                )

        # Intercept low_noise_model chain
        if low_model_source:
            src_node = self.nodes_by_id.get(low_model_source["node_id"])
            if src_node:
                src_output_name = src_node["outputs"][low_model_source.get("slot", 0)].get("name", "MODEL")
                self._intercept_link(
                    low_model_source["node_id"], src_output_name,
                    node["id"], 1, 1, "MODEL"
                )

        return node

    def add_helper_node(self, class_type, pos, size=None, widget_values=None,
                        inputs=None, outputs=None, wire_outputs=None):
        """Add a generic ComfyUI helper node (e.g., ImageScaleToTotalPixels, GetImageSize).

        Args:
            class_type: Node type string
            pos: [x, y] position
            size: [w, h] size (auto-sized if None)
            widget_values: List of widget values in order
            inputs: List of dicts with {name, type, is_widget (optional, default False)}
            outputs: List of dicts with {name, type}
            wire_outputs: Dict of output_name -> wire_to targets (dict or list of dicts)
        """
        if size is None:
            size = [315, 130]
        node = self._base_node(class_type, pos, size)

        if inputs:
            for inp in inputs:
                if inp.get("is_widget", False):
                    node["inputs"].append({
                        "name": inp["name"],
                        "type": inp["type"],
                        "widget": {"name": inp["name"]},
                        "link": None,
                    })
                else:
                    node["inputs"].append(
                        self._make_connection_input(inp["name"], inp["type"])
                    )

        if outputs:
            for i, out in enumerate(outputs):
                o = self._make_output(out["name"], out["type"])
                o["slot_index"] = i
                node["outputs"].append(o)

        node["widgets_values"] = widget_values or []
        self._add_node(node)

        # Wire outputs to targets
        if wire_outputs:
            for out_name, targets in wire_outputs.items():
                out_slot = self._find_output_slot(node["id"], out_name)
                if out_slot is not None:
                    out_type = node["outputs"][out_slot]["type"]
                    self._wire_to_targets(node["id"], out_slot, targets, out_type)

        return node

    def wire_nodes(self, src_node_id, src_slot, dst_node_id, dst_input_name, link_type):
        """Public method to wire two nodes together by input name."""
        return self._wire_output_to_input(src_node_id, src_slot, dst_node_id, dst_input_name, link_type)

    def add_lora_loader(self, path_filter="", ui_order=50, pos=(-800, 2000),
                        model_source=None, clip_source=None):
        """Add StimmaLoraLoader node and intercept model/clip chains."""
        node = self._base_node("StimmaLoraLoader", pos, [480, 806])

        # Build inputs: model, clip, path_filter, ui_order, then 10x (lora_N, strength_N)
        node["inputs"] = [
            self._make_connection_input("model", "MODEL"),
            self._make_connection_input("clip", "CLIP"),
        ]
        node["inputs"].extend(self._make_widget_inputs([
            ("path_filter", "STRING"), ("ui_order", "INT"),
        ]))
        # 10 LoRA slots (optional, shape 7)
        for i in range(1, 11):
            lora_input = {"name": f"lora_{i}", "type": "COMBO",
                          "widget": {"name": f"lora_{i}"}, "link": None, "shape": 7}
            strength_input = {"name": f"strength_{i}", "type": "FLOAT",
                              "widget": {"name": f"strength_{i}"}, "link": None, "shape": 7}
            node["inputs"].extend([lora_input, strength_input])

        node["outputs"] = [
            self._make_output("model", "MODEL"),
            self._make_output("clip", "CLIP"),
        ]
        node["outputs"][0]["slot_index"] = 0
        node["outputs"][1]["slot_index"] = 1

        # widgets_values: path_filter, ui_order, then 10x (lora_name, strength)
        wv = [path_filter, ui_order]
        for _ in range(10):
            wv.extend(["None", 1])
        node["widgets_values"] = wv

        self._add_node(node)

        # Intercept model chain
        if model_source:
            src_node = self.nodes_by_id.get(model_source["node_id"])
            if src_node:
                src_output_name = src_node["outputs"][model_source.get("slot", 0)].get("name", "MODEL")
                self._intercept_link(
                    model_source["node_id"], src_output_name,
                    node["id"], 0, 0, "MODEL"
                )

        # Intercept clip chain
        if clip_source:
            src_node = self.nodes_by_id.get(clip_source["node_id"])
            if src_node:
                src_output_name = src_node["outputs"][clip_source.get("slot", 0)].get("name", "CLIP")
                self._intercept_link(
                    clip_source["node_id"], src_output_name,
                    node["id"], 1, 1, "CLIP"
                )

        return node

    def add_checkpoint_loader(self, path_filter="", ui_order=50, default_ckpt="",
                               pos=(-800, -600), replace_node_id=None):
        """Add StimmaCheckpointLoader node, optionally replacing an existing CheckpointLoaderSimple.

        If replace_node_id is given, all consumers of that node's outputs
        (MODEL slot 0, CLIP slot 1, VAE slot 2) are rewired to the new
        StimmaCheckpointLoader, and the old node is deleted.
        """
        node = self._base_node("StimmaCheckpointLoader", pos, [420, 180])

        # Widget inputs: ckpt_name, path_filter, ui_order
        node["inputs"] = [
            {"name": "ckpt_name", "type": "COMBO",
             "widget": {"name": "ckpt_name"}, "link": None},
        ]
        node["inputs"].extend(self._make_widget_inputs([
            ("path_filter", "STRING"), ("ui_order", "INT"),
        ]))

        node["outputs"] = [
            self._make_output("model", "MODEL"),
            self._make_output("clip", "CLIP"),
            self._make_output("vae", "VAE"),
        ]
        for i, out in enumerate(node["outputs"]):
            out["slot_index"] = i

        node["widgets_values"] = [default_ckpt or "", path_filter, ui_order]
        self._add_node(node)

        # Replace an existing checkpoint loader by rewiring all its consumers
        if replace_node_id is not None:
            old_node = self.nodes_by_id.get(replace_node_id)
            if old_node:
                # For each output slot on the old node, find consumers and rewire
                for old_slot, out in enumerate(old_node.get("outputs", [])):
                    link_ids = out.get("links") or []
                    out_type = out.get("type", "MODEL")
                    for lid in list(link_ids):
                        # Find the link entry
                        for link in self.workflow.get("links", []):
                            if link[0] == lid:
                                # link = [id, src_node, src_slot, dst_node, dst_slot, type]
                                dst_node = link[3]
                                dst_slot = link[4]
                                # Remove old link
                                self.workflow["links"].remove(link)
                                # Clear old link reference on destination
                                dst = self.nodes_by_id.get(dst_node)
                                if dst:
                                    for inp in dst.get("inputs", []):
                                        if inp.get("link") == lid:
                                            inp["link"] = None
                                # Add new link from StimmaCheckpointLoader
                                self._add_link(node["id"], old_slot, dst_node, dst_slot, out_type)
                                break
                # Delete the old node
                self.delete_node(replace_node_id)

        return node

    def add_image_output(self, pos=(800, 0), wire_from=None):
        """Add StimmaImageOutput node, optionally wired from an image source."""
        node = self._base_node("StimmaImageOutput", pos, [360, 300])
        node["inputs"] = [
            self._make_connection_input("images", "IMAGE"),
            {
                "name": "filename_prefix", "type": "STRING",
                "widget": {"name": "filename_prefix"}, "link": None,
            },
            {
                "name": "_stimma_output_dir", "type": "STRING",
                "widget": {"name": "_stimma_output_dir"}, "link": None, "shape": 7,
            },
        ]
        node["outputs"] = []
        node["widgets_values"] = ["Stimma", ""]
        self._add_node(node)

        if wire_from:
            # Wire the same image source that feeds the save node
            src_id = wire_from.get("src_node")
            src_slot = wire_from.get("src_slot", 0)
            if src_id is not None:
                self._add_link(src_id, src_slot, node["id"], 0, "IMAGE")

        return node

    def add_video_output(self, fps=24, pos=(800, 400), wire_from=None, fps_source=None):
        """Add StimmaVideoOutput node.

        Args:
            fps: Default fps widget value (used when fps_source is not provided)
            pos: Node position
            wire_from: {"src_node": N, "src_slot": S} for frames input
            fps_source: {"ref": "name"} or {"node_id": N, "slot": S} to wire fps
                        from a Stimma param node instead of using hardcoded widget value
        """
        node = self._base_node("StimmaVideoOutput", pos, [360, 300])
        node["inputs"] = [
            self._make_connection_input("frames", "IMAGE"),
            {
                "name": "fps", "type": "INT",
                "widget": {"name": "fps"}, "link": None,
            },
            {
                "name": "filename_prefix", "type": "STRING",
                "widget": {"name": "filename_prefix"}, "link": None,
            },
            {
                "name": "_stimma_output_dir", "type": "STRING",
                "widget": {"name": "_stimma_output_dir"}, "link": None, "shape": 7,
            },
        ]
        node["outputs"] = []
        node["widgets_values"] = [fps, "Stimma", ""]
        self._add_node(node)

        if wire_from:
            src_id = wire_from.get("src_node")
            src_slot = wire_from.get("src_slot", 0)
            if src_id is not None:
                self._add_link(src_id, src_slot, node["id"], 0, "IMAGE")

        # Wire fps from a Stimma param node
        if fps_source:
            src_id, src_slot = self._resolve_node_ref(fps_source)
            if src_id is not None:
                self._wire_output_to_input(src_id, src_slot, node["id"], "fps", "INT")

        return node

    def add_image_input(self, name="input_image", required=True, ui_order=5,
                        pos=(-800, -200)):
        """Add StimmaImageParam node."""
        node = self._base_node("StimmaImageParam", pos, [384, 200])
        node["inputs"] = self._make_widget_inputs([
            ("name", "STRING"), ("image", "COMBO"),
            ("required", "BOOLEAN"), ("ui_order", "INT"),
        ])
        node["outputs"] = [
            self._make_output("image", "IMAGE"),
            self._make_output("mask", "MASK"),
        ]
        node["outputs"][0]["slot_index"] = 0
        node["outputs"][1]["slot_index"] = 1
        # Need a placeholder image filename
        node["widgets_values"] = [name, "example.png", required, ui_order]
        self._add_node(node)
        return node

    def add_images_input(self, name="input_images", min_images=1, max_images=3,
                        required=True, ui_order=5, pos=(-800, -200)):
        """Add StimmaImagesParam node (multi-image batch).

        Note: The node name is always 'input_images' (hardcoded in discovery.py).
        The 'required' flag maps to min_images: min_images=0 means optional.
        """
        node = self._base_node("StimmaImagesParam", pos, [384, 240])
        node["inputs"] = self._make_widget_inputs([
            ("image", "COMBO"),
            ("min_images", "INT"), ("max_images", "INT"),
            ("ui_control", "COMBO"), ("ui_order", "INT"),
        ])
        node["outputs"] = [
            self._make_output("image", "IMAGE"),
        ]
        node["outputs"][0]["slot_index"] = 0
        effective_min = 0 if not required else max(1, min_images)
        node["widgets_values"] = ["example.png", effective_min, max_images, "imagePicker", ui_order]
        self._add_node(node)
        return node

    def add_video_input(self, name="input_video", required=True, ui_order=1,
                        pos=(-800, -200)):
        """Add StimmaVideoParam node."""
        node = self._base_node("StimmaVideoParam", pos, [384, 200])
        node["inputs"] = self._make_widget_inputs([
            ("name", "STRING"), ("video", "COMBO"),
            ("required", "BOOLEAN"), ("ui_order", "INT"),
        ])
        node["outputs"] = [
            self._make_output("frames", "IMAGE"),
        ]
        node["outputs"][0]["slot_index"] = 0
        node["widgets_values"] = [name, "example.mp4", required, ui_order]
        self._add_node(node)
        return node

    # --- Subgraph prep methods ---

    def _find_subgraph_definition(self, sg_node_id):
        """Find the subgraph definition matching a subgraph node's type UUID."""
        node = self.nodes_by_id.get(sg_node_id)
        if not node:
            print(f"Warning: Subgraph node {sg_node_id} not found", file=sys.stderr)
            return None
        sg_type = node["type"]
        for defn in self.workflow.get("definitions", {}).get("subgraphs", []):
            if defn["id"] == sg_type:
                return defn
        print(f"Warning: No subgraph definition found for type {sg_type}", file=sys.stderr)
        return None

    def add_subgraph_inner_input(self, sg_node_id, name, type_str, inner_node_id, inner_widget_name):
        """Add a new input to a subgraph definition, wired to an inner node's widget.

        Creates a definition input (on the -10 input node) and an internal link
        from that input to the specified inner node's widget. The subgraph node's
        corresponding link input will be auto-created by _ensure_input_slot when wiring.
        """
        defn = self._find_subgraph_definition(sg_node_id)
        if not defn:
            return

        # Check if definition already has an input with this name
        for existing in defn.get("inputs", []):
            if existing["name"] == name:
                print(f"Subgraph already has input '{name}', skipping", file=sys.stderr)
                return

        # Find the inner node in the definition
        inner_node = None
        for n in defn.get("nodes", []):
            if n["id"] == inner_node_id:
                inner_node = n
                break
        if not inner_node:
            print(f"Warning: Inner node {inner_node_id} not found in subgraph definition",
                  file=sys.stderr)
            return

        # Find or create widget input slot on the inner node
        inner_slot = None
        inner_inputs = inner_node.setdefault("inputs", [])
        for i, inp in enumerate(inner_inputs):
            if inp.get("name") == inner_widget_name:
                inner_slot = i
                break
        if inner_slot is None:
            # Create a widget input slot
            inner_slot = len(inner_inputs)
            inner_inputs.append({
                "name": inner_widget_name,
                "type": type_str,
                "widget": {"name": inner_widget_name},
                "link": None,
            })

        # Allocate a new internal link ID from the definition's state
        state = defn.setdefault("state", {})
        new_link_id = state.get("lastLinkId", 0) + 1
        state["lastLinkId"] = new_link_id

        # Determine the new input's index (origin_slot on -10)
        inputs = defn.setdefault("inputs", [])
        origin_slot = len(inputs)

        # Compute position for the new input (offset from last input or default)
        last_pos = inputs[-1]["pos"] if inputs else [-1230, 3890]
        new_pos = [last_pos[0], last_pos[1] + 20]

        # Create the definition input
        new_input = {
            "id": str(uuid.uuid4()),
            "name": name,
            "type": type_str,
            "linkIds": [new_link_id],
            "pos": new_pos,
        }
        inputs.append(new_input)

        # Set the link on the inner node's input
        inner_inputs[inner_slot]["link"] = new_link_id

        # Add internal link (definition links are objects, not arrays)
        defn_links = defn.setdefault("links", [])
        defn_links.append({
            "id": new_link_id,
            "origin_id": -10,
            "origin_slot": origin_slot,
            "target_id": inner_node_id,
            "target_slot": inner_slot,
            "type": type_str,
        })

    def add_subgraph_inner_output(self, sg_node_id, name, type_str, inner_node_id, inner_output_slot):
        """Add a new output to a subgraph definition, sourced from an inner node's output.

        Creates a definition output (on the -20 output node) and an internal link
        from the specified inner node's output to that definition output. Also adds
        a matching output slot on the top-level subgraph node.
        """
        defn = self._find_subgraph_definition(sg_node_id)
        if not defn:
            return

        # Check if definition already has an output with this name
        for existing in defn.get("outputs", []):
            if existing["name"] == name:
                print(f"Subgraph already has output '{name}', skipping", file=sys.stderr)
                return

        # Allocate a new internal link ID from the definition's state
        state = defn.setdefault("state", {})
        new_link_id = state.get("lastLinkId", 0) + 1
        state["lastLinkId"] = new_link_id

        # Determine the new output's index (target_slot on -20)
        outputs = defn.setdefault("outputs", [])
        target_slot = len(outputs)

        # Compute position for the new output
        out_node = defn.get("outputNode", {})
        base_pos = out_node.get("bounding", [2530, 3620, 120, 60])
        new_pos = [base_pos[0] + 20, base_pos[1] + target_slot * 20]

        # Create the definition output
        new_output = {
            "id": str(uuid.uuid4()),
            "name": name,
            "type": type_str,
            "linkIds": [new_link_id],
            "pos": new_pos,
        }
        outputs.append(new_output)

        # Add internal link (definition links are objects)
        defn_links = defn.setdefault("links", [])
        defn_links.append({
            "id": new_link_id,
            "origin_id": inner_node_id,
            "origin_slot": inner_output_slot,
            "target_id": -20,
            "target_slot": target_slot,
            "type": type_str,
        })

        # Add output slot on the top-level subgraph node
        sg_node = self.nodes_by_id.get(sg_node_id)
        if sg_node:
            sg_outputs = sg_node.setdefault("outputs", [])
            sg_outputs.append({
                "name": name,
                "type": type_str,
                "links": [],
                "slot_index": len(sg_outputs),
            })

    def add_layout_group(self, label, param_names, collapsed=True, ui_order=10,
                         pos=(-1400, 400)):
        """Add StimmaLayoutGroup node.

        param_names can be:
        - List of strings: ["duration", "fps"]
        - List of strings/dicts: ["duration", {"name": "neg_prompt", "full_width": true}]
        - Newline-separated string (legacy)

        Dicts are encoded as "name !full_width" in the widget text.
        """
        node = self._base_node("StimmaLayoutGroup", pos, [365, 240])
        node["inputs"] = self._make_widget_inputs([
            ("group_label", "STRING"), ("param_names", "STRING"),
            ("collapsed", "BOOLEAN"), ("ui_order", "INT"),
        ])
        node["outputs"] = []
        if isinstance(param_names, list):
            lines = []
            for p in param_names:
                if isinstance(p, dict):
                    line = p["name"]
                    if p.get("full_width"):
                        line += " !full_width"
                    lines.append(line)
                else:
                    lines.append(p)
            names_str = "\n".join(lines)
        else:
            names_str = param_names
        node["widgets_values"] = [label, names_str, collapsed, ui_order]
        return self._add_node(node)

    def replace_save_with_stimma(self, save_node_id, output_type="image", fps_source=None):
        """Mute a SaveImage/SaveVideo node and add a Stimma output wired to the same source."""
        save_node = self.nodes_by_id.get(save_node_id)
        if not save_node:
            print(f"Warning: Save node {save_node_id} not found", file=sys.stderr)
            return None

        # Mute the original save node
        save_node["mode"] = 4

        # Find what feeds into the save node's image/frames input
        image_input = save_node.get("inputs", [{}])[0]  # First input is always images/frames
        source_link_id = image_input.get("link")

        wire_from = None
        if source_link_id:
            for link in self.workflow.get("links", []):
                if link[0] == source_link_id:
                    wire_from = {"src_node": link[1], "src_slot": link[2]}
                    break

        # Position the Stimma output near the save node
        save_pos = save_node.get("pos", [800, 0])
        new_pos = [save_pos[0], save_pos[1] - 350]

        if output_type == "video":
            return self.add_video_output(pos=new_pos, wire_from=wire_from, fps_source=fps_source)
        else:
            return self.add_image_output(pos=new_pos, wire_from=wire_from)

    def finalize(self):
        """Update last_node_id and last_link_id, validate output format."""
        self.workflow["last_node_id"] = self.next_node_id - 1
        self.workflow["last_link_id"] = self.next_link_id - 1

        # Validate: output MUST be UI format
        if "nodes" not in self.workflow:
            raise ValueError("Output workflow is missing 'nodes' array — not valid UI format")
        if "links" not in self.workflow:
            raise ValueError("Output workflow is missing 'links' array — not valid UI format")

        return self.workflow


def apply_plan(workflow_path, plan_path, output_path):
    """Apply a Stimma-ification plan to a workflow."""
    workflow = json.loads(Path(workflow_path).read_text())
    plan = json.loads(Path(plan_path).read_text())

    mod = WorkflowModifier(workflow)

    # -1. Subgraph prep (add inner inputs/outputs) — must be before everything else
    subgraph_prep = plan.get("subgraph_prep")
    if subgraph_prep:
        sg_node_id = subgraph_prep["node_id"]
        for inner in subgraph_prep.get("add_inner_inputs", []):
            mod.add_subgraph_inner_input(
                sg_node_id, inner["name"], inner["type"],
                inner["inner_node_id"], inner["inner_widget_name"],
            )
        for inner in subgraph_prep.get("add_inner_outputs", []):
            mod.add_subgraph_inner_output(
                sg_node_id, inner["name"], inner["type"],
                inner["inner_node_id"], inner["inner_output_slot"],
            )

    # 0. Pipeline setup (activate/mute/delete/bypass nodes) — must be first
    pipeline = plan.get("pipeline_setup", {})
    for node_id in pipeline.get("activate", []):
        mod.set_node_mode(node_id, 0)
    for node_id in pipeline.get("mute", []):
        mod.set_node_mode(node_id, 4)
    # Bypass before delete (bypass reconnects chains, then removes node)
    for bypass in pipeline.get("bypass", []):
        if isinstance(bypass, dict):
            mod.bypass_node(bypass["node_id"],
                            input_slot=bypass.get("input_slot", 0),
                            output_slot=bypass.get("output_slot", 0))
        else:
            mod.bypass_node(bypass)  # Just a node ID, use defaults
    for node_id in pipeline.get("delete", []):
        mod.delete_node(node_id)
    # Widget value overrides (set specific widget_values by index)
    for wo in pipeline.get("widget_overrides", []):
        nid = wo["node_id"]
        node = mod.nodes_by_id.get(nid)
        if node:
            wv = node.get("widgets_values", [])
            idx = wo["index"]
            if idx < len(wv):
                wv[idx] = wo["value"]
            else:
                print(f"Warning: widget_overrides index {idx} out of range for node {nid}", file=sys.stderr)

    # 1. Tool info
    ti = plan.get("tool_info", {})
    if ti:
        # Prefer the current `task_types` key; fall back to the legacy singular
        # `task_type` for older plans. `model_family` is no longer a node field.
        task_types = ti.get("task_types") or ti.get("task_type", "text-to-image")
        mod.add_tool_info(
            slug=ti["slug"],
            display_name=ti["display_name"],
            task_types=task_types,
            badges=ti.get("badges", ""),
            description=ti.get("description", ""),
            model_vendor=ti.get("model_vendor", ""),
            model=ti.get("model", ""),
            pos=ti.get("pos", [-1400, -400]),
        )

    # 2. Basic inputs (prompt, seed, resolution, image — no refs to other Stimma nodes)
    deferred_inputs = []
    for inp in plan.get("inputs", []):
        itype = inp["type"]
        node = None
        if itype == "prompt":
            node = mod.add_prompt_input(
                name=inp["name"],
                default_text=inp.get("default_text", ""),
                required=inp.get("required", True),
                ui_order=inp.get("ui_order", 0),
                ui_description=inp.get("ui_description", ""),
                pos=inp.get("pos", [-800, 0]),
                wire_to=inp.get("wire_to"),
            )
        elif itype == "seed":
            node = mod.add_seed_input(
                name=inp.get("name", "seed"),
                value=inp.get("value", 0),
                ui_order=inp.get("ui_order", 80),
                pos=inp.get("pos", [-800, 400]),
                wire_to=inp.get("wire_to"),
            )
        elif itype == "resolution":
            node = mod.add_resolution_input(
                width=inp.get("width", 1024),
                height=inp.get("height", 1024),
                min_size=inp.get("min_size", 512),
                max_size=inp.get("max_size", 2048),
                step=inp.get("step", 64),
                supported_resolutions=inp.get("supported_resolutions", ""),
                ui_order=inp.get("ui_order", 1),
                pos=inp.get("pos", [-800, 600]),
                wire_to=inp.get("wire_to"),
            )
        elif itype == "image":
            node = mod.add_image_input(
                name=inp.get("name", "input_image"),
                required=inp.get("required", True),
                ui_order=inp.get("ui_order", 5),
                pos=inp.get("pos", [-800, -200]),
            )
            if inp.get("wire_to"):
                mod._wire_to_targets(node["id"], 0, inp["wire_to"], "IMAGE")
        elif itype == "images":
            node = mod.add_images_input(
                name=inp.get("name", "input_images"),
                min_images=inp.get("min_images", 1),
                max_images=inp.get("max_images", 3),
                required=inp.get("required", True),
                ui_order=inp.get("ui_order", 5),
                pos=inp.get("pos", [-800, -200]),
            )
            if inp.get("wire_to"):
                mod._wire_to_targets(node["id"], 0, inp["wire_to"], "IMAGE")
        elif itype == "video":
            node = mod.add_video_input(
                name=inp.get("name", "input_video"),
                required=inp.get("required", True),
                ui_order=inp.get("ui_order", 1),
                pos=inp.get("pos", [-800, -200]),
            )
            if inp.get("wire_to"):
                mod._wire_to_targets(node["id"], 0, inp["wire_to"], "IMAGE")
        elif itype == "duration_to_frames":
            # Deferred: needs refs to params (duration, fps) created in step 3
            deferred_inputs.append(inp)
            continue
        else:
            continue

        if node and inp.get("name"):
            mod.named_nodes[inp["name"]] = node

    # 3. Parameters
    for param in plan.get("params", []):
        ptype = param["type"]
        node = None
        if ptype == "int":
            node = mod.add_int_param(
                name=param["name"],
                value=param.get("value", 20),
                minimum=param.get("min", 1),
                maximum=param.get("max", 100),
                step=param.get("step", 1),
                ui_control=param.get("ui_control", "input"),
                ui_order=param.get("ui_order", 20),
                ui_description=param.get("ui_description", ""),
                pos=param.get("pos", [-300, 0]),
                wire_to=param.get("wire_to"),
            )
        elif ptype == "float":
            node = mod.add_float_param(
                name=param["name"],
                value=param.get("value", 1.0),
                minimum=param.get("min", 0.0),
                maximum=param.get("max", 1.0),
                step=param.get("step", 0.01),
                ui_control=param.get("ui_control", "input"),
                ui_order=param.get("ui_order", 30),
                ui_description=param.get("ui_description", ""),
                pos=param.get("pos", [-300, 400]),
                wire_to=param.get("wire_to"),
            )
        elif ptype == "string":
            node = mod.add_string_param(
                name=param["name"],
                value=param.get("value", ""),
                enum_values=param.get("enum_values", ""),
                ui_control=param.get("ui_control", "textarea"),
                ui_order=param.get("ui_order", 20),
                ui_description=param.get("ui_description", ""),
                pos=param.get("pos", [-300, 400]),
                wire_to=param.get("wire_to"),
            )
        elif ptype == "dropdown":
            node = mod.add_dropdown_param(
                name=param["name"],
                value=param.get("value", ""),
                ui_order=param.get("ui_order", 40),
                ui_description=param.get("ui_description", ""),
                pos=param.get("pos", [-300, 800]),
                wire_to=param.get("wire_to"),
            )
        elif ptype == "bool":
            node = mod.add_bool_param(
                name=param["name"],
                value=param.get("value", False),
                ui_order=param.get("ui_order", 50),
                ui_description=param.get("ui_description", ""),
                pos=param.get("pos", [-300, 1200]),
                wire_to=param.get("wire_to"),
            )

        if node and param.get("name"):
            mod.named_nodes[param["name"]] = node

    # 3.5. Deferred inputs (duration_to_frames — needs refs to params)
    for inp in deferred_inputs:
        node = mod.add_duration_to_frames(
            frame_step=inp.get("frame_step", 4),
            pos=inp.get("pos", [-400, 600]),
            duration_source=inp.get("duration_source"),
            fps_source=inp.get("fps_source"),
            wire_to=inp.get("wire_to"),
        )
        if node and inp.get("name"):
            mod.named_nodes[inp["name"]] = node

    # 5. Helper nodes (ImageScaleToTotalPixels, GetImageSize, etc.)
    for helper in plan.get("helper_nodes", []):
        node = mod.add_helper_node(
            class_type=helper["class_type"],
            pos=helper.get("pos", [-300, 600]),
            size=helper.get("size"),
            widget_values=helper.get("widget_values"),
            inputs=helper.get("inputs"),
            outputs=helper.get("outputs"),
            wire_outputs=helper.get("wire_outputs"),
        )
        # Wire inputs from refs
        for inp_name, src_ref in helper.get("wire_inputs", {}).items():
            src_id, src_slot = mod._resolve_node_ref(src_ref)
            if src_id is not None:
                inp_type = src_ref.get("type", "IMAGE")
                mod._wire_output_to_input(src_id, src_slot, node["id"], inp_name, inp_type)

        # Register by id for cross-referencing
        if helper.get("id"):
            mod.named_nodes[helper["id"]] = node

    # 5.5. Checkpoint loader
    ckpt = plan.get("checkpoint_loader")
    if ckpt:
        node = mod.add_checkpoint_loader(
            path_filter=ckpt.get("path_filter", ""),
            ui_order=ckpt.get("ui_order", 50),
            default_ckpt=ckpt.get("default", ""),
            pos=ckpt.get("pos", [-800, -600]),
            replace_node_id=ckpt.get("replace_node_id"),
        )
        mod.named_nodes["checkpoint"] = node

    # 6. LoRA loader
    lora = plan.get("lora_loader")
    if lora:
        lora_type = lora.get("type", "standard")
        # Resolve refs in model_source/clip_source
        def _resolve_source(src):
            if not src:
                return src
            if "ref" in src:
                nid, slot = mod._resolve_node_ref(src)
                if nid is not None:
                    return {"node_id": nid, "slot": slot}
                return None
            return src
        if lora_type == "paired":
            node = mod.add_paired_lora_loader(
                path_filter=lora.get("path_filter", ""),
                ui_order=lora.get("ui_order", 50),
                pos=lora.get("pos", [-800, 2000]),
                high_model_source=_resolve_source(lora.get("high_model_source")),
                low_model_source=_resolve_source(lora.get("low_model_source")),
            )
        else:
            node = mod.add_lora_loader(
                path_filter=lora.get("path_filter", ""),
                ui_order=lora.get("ui_order", 50),
                pos=lora.get("pos", [-800, 2000]),
                model_source=_resolve_source(lora.get("model_source")),
                clip_source=_resolve_source(lora.get("clip_source")),
            )
        mod.named_nodes["loras"] = node

    # 6.5. Post-wires (explicit wires from any ref/node to any node input)
    for pw in plan.get("post_wires", []):
        src_id, src_slot = mod._resolve_node_ref(pw["from"])
        if src_id is not None:
            mod._wire_output_to_input(
                src_id, src_slot, pw["to"]["node_id"], pw["to"]["input_name"], pw.get("type", "MODEL")
            )

    # 7. Outputs
    for out in plan.get("outputs", []):
        fps_source = out.get("fps_source")
        if "replace_node_id" in out:
            # Existing behavior: mute save node and create Stimma output
            mod.replace_save_with_stimma(
                save_node_id=out["replace_node_id"],
                output_type=out.get("output_type", "image"),
                fps_source=fps_source,
            )
        else:
            # Direct output creation with explicit wiring
            output_type = out.get("output_type", "image")
            wire_from = out.get("wire_from")
            # Resolve wire_from refs
            if wire_from and "ref" in wire_from:
                src_id, src_slot = mod._resolve_node_ref(wire_from)
                wire_from = {"src_node": src_id, "src_slot": src_slot} if src_id else None
            elif wire_from and "output_name" in wire_from:
                out_slot = mod._find_output_slot(wire_from["node_id"], wire_from["output_name"])
                if out_slot is not None:
                    wire_from = {"src_node": wire_from["node_id"], "src_slot": out_slot}
                else:
                    print(f"Warning: Output '{wire_from['output_name']}' not found on node {wire_from['node_id']}",
                          file=sys.stderr)
                    wire_from = None
            elif wire_from and "node_id" in wire_from:
                wire_from = {"src_node": wire_from["node_id"], "src_slot": wire_from.get("src_slot", 0)}
            if output_type == "video":
                mod.add_video_output(
                    fps=out.get("fps", 24),
                    pos=out.get("pos", [800, 400]),
                    wire_from=wire_from,
                    fps_source=fps_source,
                )
            else:
                mod.add_image_output(
                    pos=out.get("pos", [800, 0]),
                    wire_from=wire_from,
                )
            # Mute associated nodes
            for mute_id in out.get("mute_nodes", []):
                mod.set_node_mode(mute_id, 4)

    # 8. Layout groups
    for layout in plan.get("layout", []):
        mod.add_layout_group(
            label=layout["label"],
            param_names=layout["param_names"],
            collapsed=layout.get("collapsed", True),
            ui_order=layout.get("ui_order", 10),
            pos=layout.get("pos", [-1400, 400]),
        )

    result = mod.finalize()
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"Wrote Stimma-ified workflow to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Apply Stimma nodes to a ComfyUI workflow")
    parser.add_argument("workflow", help="Source workflow JSON file")
    parser.add_argument("plan", help="Plan JSON file describing what to add")
    parser.add_argument("-o", "--output", required=True, help="Output workflow JSON path")
    args = parser.parse_args()

    apply_plan(args.workflow, args.plan, args.output)


if __name__ == "__main__":
    main()
