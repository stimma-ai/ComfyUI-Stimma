#!/usr/bin/env python3
"""Flatten LTX2 subgraph workflows and add Stimma nodes.

Handles all 4 LTX2 variants: T2V, T2V-Distilled, I2V, I2V-Distilled.
"""

import json
import copy
import sys
import urllib.request
from pathlib import Path

from workflow_layout_cleanup import cleanup_workflow

WORKFLOWS_DIR = Path(__file__).parent.parent / "workflows"
STIMMAFY_SCRIPTS = Path(__file__).parent.parent.parent
# Paths
SCRIPT_DIR = Path(__file__).parent


class WorkflowFlattener:
    """Flatten a subgraph-based workflow into top-level nodes."""

    def __init__(self, workflow):
        self.wf = copy.deepcopy(workflow)
        # Collect ALL node IDs (top-level + all subgraph inner nodes)
        all_ids = {n["id"] for n in self.wf.get("nodes", [])}
        for sg in self.wf.get("definitions", {}).get("subgraphs", []):
            for n in sg.get("nodes", []):
                all_ids.add(n["id"])
        self.all_used_ids = all_ids
        self.next_node_id = max(all_ids) + 1 if all_ids else 1
        # Same for links
        all_link_ids = set()
        for l in self.wf.get("links", []):
            all_link_ids.add(l[0])
        for sg in self.wf.get("definitions", {}).get("subgraphs", []):
            for l in sg.get("links", []):
                all_link_ids.add(l["id"])
        self.all_used_link_ids = all_link_ids
        self.next_link_id = max(all_link_ids) + 1 if all_link_ids else 1
        self.id_remap = {}  # old_id -> new_id for conflicting nodes

    def _alloc_id(self):
        nid = self.next_node_id
        self.next_node_id += 1
        self.all_used_ids.add(nid)
        return nid

    def _alloc_link_id(self):
        lid = self.next_link_id
        self.next_link_id += 1
        self.all_used_link_ids.add(lid)
        return lid

    def _remap_id(self, node_id):
        """Get the remapped ID for a node, or the original if no conflict."""
        return self.id_remap.get(node_id, node_id)

    def _resolve_reroutes(self):
        """Resolve all Reroute nodes by bypassing them with direct connections.

        Reroute nodes are UI-only pass-throughs. After flattening, we replace
        them with direct source→destination links to avoid validation issues.
        """
        nodes_by_id = {n["id"]: n for n in self.wf["nodes"]}
        links = self.wf.get("links", [])

        # Find all Reroute nodes
        reroute_ids = {n["id"] for n in self.wf["nodes"] if n.get("type") == "Reroute"}
        if not reroute_ids:
            return

        # Build link lookups
        # incoming[node_id] = (link_entry, src_id, src_slot)
        incoming = {}
        # outgoing[node_id] = [(link_entry, dst_id, dst_slot), ...]
        outgoing = {}
        for link in links:
            lid, src_id, src_slot, dst_id, dst_slot, ltype = link
            if dst_id in reroute_ids:
                incoming[dst_id] = (link, src_id, src_slot)
            if src_id in reroute_ids:
                outgoing.setdefault(src_id, []).append((link, dst_id, dst_slot))

        def trace_source(node_id, visited=None):
            """Follow Reroute chain to find the real source node and slot."""
            if visited is None:
                visited = set()
            if node_id in visited:
                return None, None
            visited.add(node_id)
            if node_id not in reroute_ids:
                return None, None  # not a reroute
            entry = incoming.get(node_id)
            if not entry:
                return None, None
            _, src_id, src_slot = entry
            if src_id in reroute_ids:
                return trace_source(src_id, visited)
            return src_id, src_slot

        # For each Reroute, rewire its outgoing links to come from the real source
        links_to_remove = set()
        for rr_id in reroute_ids:
            real_src, real_slot = trace_source(rr_id)
            if real_src is None:
                continue

            # Mark incoming link to this Reroute for removal
            if rr_id in incoming:
                links_to_remove.add(incoming[rr_id][0][0])  # link ID

            # Rewire outgoing links
            for link_entry, dst_id, dst_slot in outgoing.get(rr_id, []):
                link_entry[1] = real_src
                link_entry[2] = real_slot
                # Update source node's output links
                src_node = nodes_by_id.get(real_src)
                if src_node:
                    outputs = src_node.get("outputs", [])
                    if real_slot < len(outputs):
                        out = outputs[real_slot]
                        if out.get("links") is None:
                            out["links"] = []
                        if link_entry[0] not in out["links"]:
                            out["links"].append(link_entry[0])

        # Remove the incoming links to Reroutes (they're now bypassed)
        self.wf["links"] = [l for l in self.wf["links"] if l[0] not in links_to_remove]

        # Remove Reroute nodes
        self.wf["nodes"] = [n for n in self.wf["nodes"] if n.get("type") != "Reroute"]

        # Clean up source node output references to removed links
        for node in self.wf["nodes"]:
            for out in node.get("outputs", []):
                if out.get("links"):
                    out["links"] = [lid for lid in out["links"] if lid not in links_to_remove]

    def flatten(self, remove_markdown_notes=True):
        """Flatten the first subgraph into the top level."""
        sg_defs = self.wf.get("definitions", {}).get("subgraphs", [])
        if not sg_defs:
            return self.wf

        sg = sg_defs[0]
        sg_id = sg["id"]

        # Find the subgraph instance node
        sg_node = None
        sg_node_idx = None
        for i, n in enumerate(self.wf["nodes"]):
            if n.get("type") == sg_id:
                sg_node = n
                sg_node_idx = i
                break

        if not sg_node:
            print("Warning: subgraph instance node not found", file=sys.stderr)
            return self.wf

        # Collect IDs
        top_ids = {n["id"] for n in self.wf["nodes"]}
        inner_ids = {n["id"] for n in sg.get("nodes", [])}
        conflicts = top_ids & inner_ids

        # Remap conflicting top-level nodes (they're usually MarkdownNotes)
        for cid in conflicts:
            new_id = self._alloc_id()
            self.id_remap[cid] = new_id

        # Remap top-level nodes with conflicts
        for node in self.wf["nodes"]:
            if node["id"] in self.id_remap:
                old_id = node["id"]
                node["id"] = self.id_remap[old_id]

        # Remap top-level links that reference remapped nodes
        for link in self.wf.get("links", []):
            if link[1] in self.id_remap:
                link[1] = self.id_remap[link[1]]
            if link[3] in self.id_remap:
                link[3] = self.id_remap[link[3]]

        # Build subgraph input/output mapping
        # input mapping: slot_idx -> list of (target_node_id, target_slot)
        sg_input_map = {}
        # output mapping: slot_idx -> (source_node_id, source_slot, link_type)
        sg_output_map = {}

        for link in sg.get("links", []):
            if link["origin_id"] == -10:
                slot = link["origin_slot"]
                sg_input_map.setdefault(slot, []).append(
                    (link["target_id"], link["target_slot"], link["type"])
                )
            elif link["target_id"] == -20:
                slot = link["target_slot"]
                sg_output_map[slot] = (link["origin_id"], link["origin_slot"], link["type"])

        # Convert inner links to top-level format (skip -10 and -20 links)
        for link in sg.get("links", []):
            if link["origin_id"] in (-10, -20) or link["target_id"] in (-10, -20):
                continue
            lid = self._alloc_link_id()
            self.wf.setdefault("links", []).append([
                lid,
                link["origin_id"],
                link["origin_slot"],
                link["target_id"],
                link["target_slot"],
                link["type"],
            ])
            # Update inner node input/output link references
            for n in sg.get("nodes", []):
                if n["id"] == link["target_id"]:
                    for inp in n.get("inputs", []):
                        if inp.get("link") == link["id"]:
                            inp["link"] = lid
                if n["id"] == link["origin_id"]:
                    for out in n.get("outputs", []):
                        if out.get("links") and link["id"] in out["links"]:
                            out["links"] = [lid if x == link["id"] else x for x in out["links"]]

        # Move inner nodes to top level
        for node in sg.get("nodes", []):
            if remove_markdown_notes and node.get("type") == "MarkdownNote":
                continue
            self.wf["nodes"].append(node)

        # Now handle connections that went through the subgraph boundary

        # 1. Subgraph output → downstream nodes
        # Find what the subgraph node's outputs connected to
        top_links_from_sg = []
        for link in self.wf.get("links", []):
            if link[1] == sg_node["id"]:
                top_links_from_sg.append(link)

        for link in top_links_from_sg:
            sg_output_slot = link[2]
            if sg_output_slot in sg_output_map:
                src_id, src_slot, _ = sg_output_map[sg_output_slot]
                # Rewire: inner source → original destination
                link[1] = src_id
                link[2] = src_slot
                # Update the inner source node's output links
                for n in self.wf["nodes"]:
                    if n["id"] == src_id:
                        for out in n.get("outputs", []):
                            if out.get("links") is None:
                                out["links"] = []
                            out["links"].append(link[0])

        # 2. Upstream nodes → subgraph inputs
        # Find top-level links going INTO the subgraph node
        top_links_to_sg = []
        for link in self.wf.get("links", []):
            if link[3] == sg_node["id"]:
                top_links_to_sg.append(link)

        # Build a lookup for inner nodes by id (they're now in the top-level)
        nodes_by_id = {n["id"]: n for n in self.wf["nodes"]}

        for link in top_links_to_sg:
            sg_input_slot_idx = link[4]
            sg_node_inputs = sg_node.get("inputs", [])
            if sg_input_slot_idx < len(sg_node_inputs):
                input_name = sg_node_inputs[sg_input_slot_idx].get("name", "")
                # Find the subgraph input definition by name
                for si, sg_inp in enumerate(sg.get("inputs", [])):
                    if sg_inp["name"] == input_name:
                        if si in sg_input_map:
                            targets = sg_input_map[si]
                            first = True
                            for target_id, target_slot, target_type in targets:
                                target_node = nodes_by_id.get(target_id)
                                if first:
                                    link[3] = target_id
                                    link[4] = target_slot
                                    link[5] = target_type
                                    # Update target node's input to reference this link
                                    if target_node:
                                        inputs = target_node.get("inputs", [])
                                        if target_slot < len(inputs):
                                            inputs[target_slot]["link"] = link[0]
                                    first = False
                                else:
                                    new_lid = self._alloc_link_id()
                                    self.wf["links"].append([
                                        new_lid,
                                        link[1],  # same source
                                        link[2],
                                        target_id,
                                        target_slot,
                                        target_type,
                                    ])
                                    # Update target node's input
                                    if target_node:
                                        inputs = target_node.get("inputs", [])
                                        if target_slot < len(inputs):
                                            inputs[target_slot]["link"] = new_lid
                                    # Update source node's output links
                                    src_node = nodes_by_id.get(link[1])
                                    if src_node:
                                        outputs = src_node.get("outputs", [])
                                        if link[2] < len(outputs):
                                            if outputs[link[2]].get("links") is None:
                                                outputs[link[2]]["links"] = []
                                            outputs[link[2]]["links"].append(new_lid)
                        break

        # Remove the subgraph instance node
        self.wf["nodes"] = [n for n in self.wf["nodes"] if n.get("type") != sg_id]

        # Clean up: remove links referencing the (now removed) subgraph node
        # that weren't already rewired
        self.wf["links"] = [
            l for l in self.wf["links"]
            if l[1] != sg_node["id"] and l[3] != sg_node["id"]
        ]

        # 3. Clean up stale link references on inner nodes
        # Inner nodes may have input slots referencing old subgraph boundary links
        # (from -10). These links were not converted to top-level links, so clear them.
        valid_link_ids = {l[0] for l in self.wf.get("links", [])}
        for node in self.wf["nodes"]:
            for inp in node.get("inputs", []):
                if inp.get("link") is not None and inp["link"] not in valid_link_ids:
                    inp["link"] = None
            for out in node.get("outputs", []):
                if out.get("links"):
                    out["links"] = [lid for lid in out["links"] if lid in valid_link_ids]

        # 4. Resolve Reroute nodes — bypass them with direct connections
        self._resolve_reroutes()

        # 5. Remove top-level MarkdownNote nodes (documentation-only, not executable)
        if remove_markdown_notes:
            md_ids = {n["id"] for n in self.wf["nodes"] if n.get("type") == "MarkdownNote"}
            if md_ids:
                self.wf["nodes"] = [n for n in self.wf["nodes"] if n["id"] not in md_ids]
                self.wf["links"] = [l for l in self.wf["links"] if l[1] not in md_ids and l[3] not in md_ids]

        # Remove subgraph definitions
        if "definitions" in self.wf:
            del self.wf["definitions"]

        # Update counters — must account for ALL node/link IDs including inner nodes
        all_node_ids = [n["id"] for n in self.wf["nodes"]]
        all_link_ids = [l[0] for l in self.wf.get("links", [])]
        self.wf["last_node_id"] = max(all_node_ids) if all_node_ids else 0
        self.wf["last_link_id"] = max(all_link_ids) if all_link_ids else 0

        return self.wf


class StimmaNodeAdder:
    """Add Stimma nodes to a flattened workflow."""

    def __init__(self, workflow):
        self.wf = workflow
        self.next_node_id = workflow.get("last_node_id", 0) + 1
        self.next_link_id = workflow.get("last_link_id", 0) + 1
        self.nodes_by_id = {n["id"]: n for n in self.wf["nodes"]}

    def _alloc_node_id(self):
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def _alloc_link_id(self):
        lid = self.next_link_id
        self.next_link_id += 1
        return lid

    def _add_node(self, node):
        self.wf["nodes"].append(node)
        self.nodes_by_id[node["id"]] = node
        return node

    def _add_link(self, src_id, src_slot, dst_id, dst_slot, link_type):
        lid = self._alloc_link_id()
        self.wf.setdefault("links", []).append([lid, src_id, src_slot, dst_id, dst_slot, link_type])

        # Update source output links
        src = self.nodes_by_id.get(src_id)
        if src:
            outputs = src.get("outputs", [])
            if src_slot < len(outputs):
                if outputs[src_slot].get("links") is None:
                    outputs[src_slot]["links"] = []
                outputs[src_slot]["links"].append(lid)

        # Update dest input link
        dst = self.nodes_by_id.get(dst_id)
        if dst:
            inputs = dst.get("inputs", [])
            if dst_slot < len(inputs):
                inputs[dst_slot]["link"] = lid

        return lid

    def _ensure_input(self, node_id, input_name, input_type):
        """Find or create an input slot on a node. Returns slot index."""
        node = self.nodes_by_id.get(node_id)
        if not node:
            return None
        for i, inp in enumerate(node.get("inputs", [])):
            if inp.get("name") == input_name:
                return i
        # Create new input
        inputs = node.setdefault("inputs", [])
        idx = len(inputs)
        inputs.append({
            "name": input_name,
            "type": input_type,
            "widget": {"name": input_name},
            "link": None,
        })
        return idx

    def _base_node(self, ntype, pos, size):
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

    def _wire(self, src_id, src_slot, dst_id, input_name, link_type):
        """Wire source output to destination input by name."""
        dst_slot = self._ensure_input(dst_id, input_name, link_type)
        if dst_slot is not None:
            return self._add_link(src_id, src_slot, dst_id, dst_slot, link_type)
        return None

    def add_tool_info(self, slug, display_name, task_type, description, pos):
        node = self._base_node("StimmaToolInfo", pos, [480, 276])
        node["inputs"] = [
            {"name": "slug", "type": "STRING", "widget": {"name": "slug"}, "link": None},
            {"name": "display_name", "type": "STRING", "widget": {"name": "display_name"}, "link": None},
            {"name": "task_type", "type": "COMBO", "widget": {"name": "task_type"}, "link": None},
            {"name": "description", "type": "STRING", "widget": {"name": "description"}, "link": None},
        ]
        node["outputs"] = []
        node["widgets_values"] = [slug, display_name, task_type, description]
        return self._add_node(node)

    def add_prompt_input(self, name, default_text, required, ui_order, ui_description, pos, wire_to=None):
        node = self._base_node("StimmaPromptParam", pos, [480, 360])
        node["inputs"] = [
            {"name": "name", "type": "STRING", "widget": {"name": "name"}, "link": None},
            {"name": "default_text", "type": "STRING", "widget": {"name": "default_text"}, "link": None},
            {"name": "required", "type": "BOOLEAN", "widget": {"name": "required"}, "link": None},
            {"name": "ui_order", "type": "INT", "widget": {"name": "ui_order"}, "link": None},
            {"name": "ui_description", "type": "STRING", "widget": {"name": "ui_description"}, "link": None},
        ]
        node["outputs"] = [{"name": "text", "type": "STRING", "links": [], "slot_index": 0}]
        node["widgets_values"] = [name, default_text, required, ui_order, ui_description]
        self._add_node(node)
        if wire_to:
            self._wire(node["id"], 0, wire_to["node_id"], wire_to["input_name"], "STRING")
        return node

    def add_seed_input(self, name, value, ui_order, pos, wire_to=None):
        node = self._base_node("StimmaSeedParam", pos, [384, 156])
        node["inputs"] = [
            {"name": "name", "type": "STRING", "widget": {"name": "name"}, "link": None},
            {"name": "value", "type": "INT", "widget": {"name": "value"}, "link": None},
            {"name": "ui_order", "type": "INT", "widget": {"name": "ui_order"}, "link": None},
        ]
        node["outputs"] = [{"name": "seed", "type": "INT", "links": [], "slot_index": 0}]
        node["widgets_values"] = [name, value, ui_order]
        self._add_node(node)
        if wire_to:
            self._wire(node["id"], 0, wire_to["node_id"], wire_to["input_name"], "INT")
        return node

    def add_int_param(self, name, value, minimum, maximum, step, ui_control, ui_order, ui_description, pos, wire_to=None):
        node = self._base_node("StimmaIntParam", pos, [384, 355])
        node["inputs"] = [
            {"name": "name", "type": "STRING", "widget": {"name": "name"}, "link": None},
            {"name": "value", "type": "INT", "widget": {"name": "value"}, "link": None},
            {"name": "minimum", "type": "INT", "widget": {"name": "minimum"}, "link": None},
            {"name": "maximum", "type": "INT", "widget": {"name": "maximum"}, "link": None},
            {"name": "step", "type": "INT", "widget": {"name": "step"}, "link": None},
            {"name": "ui_control", "type": "COMBO", "widget": {"name": "ui_control"}, "link": None},
            {"name": "ui_order", "type": "INT", "widget": {"name": "ui_order"}, "link": None},
            {"name": "ui_description", "type": "STRING", "widget": {"name": "ui_description"}, "link": None},
        ]
        node["outputs"] = [{"name": "value", "type": "INT", "links": [], "slot_index": 0}]
        node["widgets_values"] = [name, value, minimum, maximum, step, ui_control, ui_order, ui_description]
        self._add_node(node)
        if wire_to:
            self._wire(node["id"], 0, wire_to["node_id"], wire_to["input_name"], "INT")
        return node

    def add_float_param(self, name, value, minimum, maximum, step, ui_control, ui_order, ui_description, pos, wire_to=None):
        node = self._base_node("StimmaFloatParam", pos, [384, 355])
        node["inputs"] = [
            {"name": "name", "type": "STRING", "widget": {"name": "name"}, "link": None},
            {"name": "value", "type": "FLOAT", "widget": {"name": "value"}, "link": None},
            {"name": "minimum", "type": "FLOAT", "widget": {"name": "minimum"}, "link": None},
            {"name": "maximum", "type": "FLOAT", "widget": {"name": "maximum"}, "link": None},
            {"name": "step", "type": "FLOAT", "widget": {"name": "step"}, "link": None},
            {"name": "ui_control", "type": "COMBO", "widget": {"name": "ui_control"}, "link": None},
            {"name": "ui_order", "type": "INT", "widget": {"name": "ui_order"}, "link": None},
            {"name": "ui_description", "type": "STRING", "widget": {"name": "ui_description"}, "link": None},
        ]
        node["outputs"] = [{"name": "value", "type": "FLOAT", "links": [], "slot_index": 0}]
        node["widgets_values"] = [name, value, minimum, maximum, step, ui_control, ui_order, ui_description]
        self._add_node(node)
        if wire_to:
            self._wire(node["id"], 0, wire_to["node_id"], wire_to["input_name"], "FLOAT")
        return node

    def add_video_output(self, fps, pos, src_node_id, src_slot=0):
        node = self._base_node("StimmaVideoOutput", pos, [360, 300])
        node["inputs"] = [
            {"name": "frames", "type": "IMAGE", "link": None},
            {"name": "fps", "type": "INT", "widget": {"name": "fps"}, "link": None},
            {"name": "filename_prefix", "type": "STRING", "widget": {"name": "filename_prefix"}, "link": None},
            {"name": "_stimma_output_dir", "type": "STRING", "widget": {"name": "_stimma_output_dir"}, "link": None, "shape": 7},
        ]
        node["outputs"] = []
        node["widgets_values"] = [fps, "Stimma", ""]
        self._add_node(node)
        # Wire IMAGE frames
        self._add_link(src_node_id, src_slot, node["id"], 0, "IMAGE")
        return node

    def add_image_input(self, ui_control, ui_order, pos):
        node = self._base_node("StimmaImageParam", pos, [384, 200])
        node["inputs"] = [
            {"name": "image", "type": "COMBO", "widget": {"name": "image"}, "link": None},
            {"name": "ui_control", "type": "COMBO", "widget": {"name": "ui_control"}, "link": None},
            {"name": "ui_order", "type": "INT", "widget": {"name": "ui_order"}, "link": None},
        ]
        node["outputs"] = [
            {"name": "image", "type": "IMAGE", "links": [], "slot_index": 0},
            {"name": "mask", "type": "MASK", "links": [], "slot_index": 1},
        ]
        node["widgets_values"] = ["example.png", ui_control, ui_order]
        return self._add_node(node)

    def add_resolution_input(self, width, height, step, min_size, max_size, ui_order, pos, wire_to_width=None, wire_to_height=None):
        node = self._base_node("StimmaResolutionParam", pos, [384, 310])
        node["inputs"] = [
            {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
            {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
            {"name": "min_size", "type": "INT", "widget": {"name": "min_size"}, "link": None},
            {"name": "max_size", "type": "INT", "widget": {"name": "max_size"}, "link": None},
            {"name": "step", "type": "INT", "widget": {"name": "step"}, "link": None},
            {"name": "supported_resolutions", "type": "STRING", "widget": {"name": "supported_resolutions"}, "link": None},
            {"name": "ui_order", "type": "INT", "widget": {"name": "ui_order"}, "link": None},
        ]
        node["outputs"] = [
            {"name": "width", "type": "INT", "links": [], "slot_index": 0},
            {"name": "height", "type": "INT", "links": [], "slot_index": 1},
        ]
        node["widgets_values"] = [width, height, min_size, max_size, step, "", ui_order]
        self._add_node(node)
        if wire_to_width:
            self._wire(node["id"], 0, wire_to_width["node_id"], wire_to_width["input_name"], "INT")
        if wire_to_height:
            self._wire(node["id"], 1, wire_to_height["node_id"], wire_to_height["input_name"], "INT")
        return node

    def add_image_scale_to_total_pixels(self, megapixels_node_id, image_src_node_id, image_src_slot, pos):
        """Add an ImageScaleToTotalPixels node and wire it up."""
        node = self._base_node("ImageScaleToTotalPixels", pos, [315, 82])
        node["inputs"] = [
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "upscale_method", "type": "COMBO", "widget": {"name": "upscale_method"}, "link": None},
            {"name": "megapixels", "type": "FLOAT", "widget": {"name": "megapixels"}, "link": None},
            {"name": "resolution_steps", "type": "INT", "widget": {"name": "resolution_steps"}, "link": None},
        ]
        node["outputs"] = [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "slot_index": 0},
        ]
        node["widgets_values"] = ["lanczos", 0.9, 32]
        self._add_node(node)
        # Wire image input
        self._add_link(image_src_node_id, image_src_slot, node["id"], 0, "IMAGE")
        # Wire megapixels from StimmaFloatParam
        self._add_link(megapixels_node_id, 0, node["id"], 2, "FLOAT")
        return node

    def rewire_image_output(self, old_src_id, old_src_slot, new_src_id, new_src_slot):
        """Rewire all IMAGE links from old source to new source."""
        for link in self.wf.get("links", []):
            if link[1] == old_src_id and link[2] == old_src_slot and link[5] == "IMAGE":
                # Update the link source
                old_node = self.nodes_by_id.get(old_src_id)
                new_node = self.nodes_by_id.get(new_src_id)
                link[1] = new_src_id
                link[2] = new_src_slot
                # Update output link refs
                if old_node:
                    for out in old_node.get("outputs", []):
                        if out.get("links") and link[0] in out["links"]:
                            out["links"].remove(link[0])
                if new_node:
                    for out in new_node.get("outputs", []):
                        if out.get("slot_index") == new_src_slot:
                            if out.get("links") is None:
                                out["links"] = []
                            out["links"].append(link[0])

    def add_layout_group(self, label, param_names, collapsed, ui_order, pos):
        node = self._base_node("StimmaLayoutGroup", pos, [365, 240])
        node["inputs"] = [
            {"name": "group_label", "type": "STRING", "widget": {"name": "group_label"}, "link": None},
            {"name": "param_names", "type": "STRING", "widget": {"name": "param_names"}, "link": None},
            {"name": "collapsed", "type": "BOOLEAN", "widget": {"name": "collapsed"}, "link": None},
            {"name": "ui_order", "type": "INT", "widget": {"name": "ui_order"}, "link": None},
        ]
        node["outputs"] = []
        names_str = "\n".join(param_names)
        node["widgets_values"] = [label, names_str, collapsed, ui_order]
        return self._add_node(node)

    def add_duration_to_frames(self, duration_node_id, fps_node_id, frame_step, pos):
        """Add a StimmaDurationToFrames converter node."""
        node = self._base_node("StimmaDurationToFrames", pos, [315, 82])
        node["inputs"] = [
            {"name": "duration", "type": "FLOAT", "link": None},
            {"name": "fps", "type": "INT", "link": None},
            {"name": "frame_step", "type": "INT", "widget": {"name": "frame_step"}, "link": None},
        ]
        node["outputs"] = [
            {"name": "frames", "type": "INT", "links": [], "slot_index": 0},
        ]
        node["widgets_values"] = [frame_step]
        self._add_node(node)
        # Wire duration and fps inputs
        self._add_link(duration_node_id, 0, node["id"], 0, "FLOAT")
        self._add_link(fps_node_id, 0, node["id"], 1, "INT")
        return node

    def mute_node(self, node_id):
        """Mute a node (mode=4/NEVER)."""
        node = self.nodes_by_id.get(node_id)
        if node:
            node["mode"] = 4

    def remove_link_to(self, dst_node_id, input_name):
        """Remove the link going to a specific input, clearing the link ref."""
        node = self.nodes_by_id.get(dst_node_id)
        if not node:
            return
        for inp in node.get("inputs", []):
            if inp.get("name") == input_name and inp.get("link") is not None:
                old_link_id = inp["link"]
                inp["link"] = None
                # Remove from links array and source output
                self.wf["links"] = [l for l in self.wf["links"] if l[0] != old_link_id]
                # Clean up source node output links
                for link in list(self.wf.get("links", [])):
                    pass  # already removed
                for n in self.wf["nodes"]:
                    for out in n.get("outputs", []):
                        if out.get("links") and old_link_id in out["links"]:
                            out["links"].remove(old_link_id)
                break

    def replace_load_image_with_stimma(self, load_image_id, stimma_image_node):
        """Replace LoadImage links with StimmaImageParam links."""
        load_node = self.nodes_by_id.get(load_image_id)
        if not load_node:
            return

        # Find all outgoing IMAGE links from LoadImage
        links_to_rewire = []
        for link in self.wf.get("links", []):
            if link[1] == load_image_id and link[5] == "IMAGE":
                links_to_rewire.append(link)

        # Rewire to come from StimmaImageParam instead
        for link in links_to_rewire:
            link[1] = stimma_image_node["id"]
            link[2] = 0  # IMAGE output slot
            # Update stimma node's output links
            stimma_image_node["outputs"][0]["links"].append(link[0])
            # Remove from old node's output
            for out in load_node.get("outputs", []):
                if out.get("links") and link[0] in out["links"]:
                    out["links"].remove(link[0])

        # Mute the LoadImage
        load_node["mode"] = 4

    def finalize(self):
        self.wf["last_node_id"] = self.next_node_id - 1
        self.wf["last_link_id"] = self.next_link_id - 1
        return self.wf


# Workflow configurations
CONFIGS = {
    "video_ltx2_t2v.json": {
        "slug": "ltx2-t2v",
        "display_name": "LTX 2.0 T2V",
        "task_type": "text-to-video",
        "description": "LTX Video 2.0 text-to-video generation with audio support",

        "output_name": "Stimma-LTX2-T2V.json",
        # Inner nodes for Stimma wiring
        "prompt_node": 3,        # CLIPTextEncode (positive)
        "prompt_input": "text",  # input name on CLIPTextEncode
        "seed_node": 67,         # RandomNoise (stage 2, exposed by proxy)
        "seed_input": "noise_seed",
        "steps_node": 9,         # LTXVScheduler
        "steps_input": "steps",
        "has_steps": True,
        "default_steps": 20,
        "frame_count_node": 62,  # PrimitiveInt
        "frame_count_input": "value",
        "default_frame_count": 121,
        "image_source_node": 98, # VAEDecodeTiled
        "image_source_slot": 0,
        "fps": 24,
        "save_video_node": 75,
        "create_video_node": 97,
        "is_i2v": False,
        "load_image_node": None,
        # Resolution: EmptyImage receives width/height
        "empty_image_node": 89,
        # LoRA: checkpoint MODEL slot 0, text encoder CLIP slot 0
        "checkpoint_node": 1,
        "text_encoder_node": 60,
        "lora_path_filter": "*ltx*;*LTX*",
    },
    "video_ltx2_t2v_distilled.json": {
        "slug": "ltx2-t2v-distilled",
        "display_name": "LTX 2.0 T2V Distilled",
        "task_type": "text-to-video",
        "description": "LTX Video 2.0 distilled text-to-video (faster, fewer steps)",

        "output_name": "Stimma-LTX2-T2V-Distilled.json",
        "prompt_node": 102,       # CLIPTextEncode
        "prompt_input": "text",
        "seed_node": 123,         # RandomNoise (stage 1, exposed by proxy)
        "seed_input": "noise_seed",
        "has_steps": False,       # ManualSigmas, not configurable
        "frame_count_node": 113,  # PrimitiveInt
        "frame_count_input": "value",
        "default_frame_count": 121,
        "image_source_node": 119, # VAEDecodeTiled
        "image_source_slot": 0,
        "fps": 24,
        "save_video_node": 104,
        "create_video_node": 114,
        "is_i2v": False,
        "load_image_node": None,
        "empty_image_node": 131,
        "checkpoint_node": 100,
        "text_encoder_node": 111,
        "lora_path_filter": "*ltx*;*LTX*",
        "ckpt_remap": {
            "ltx-2-19b-distilled.safetensors": "ltx-2-19b-distilled-fp8.safetensors",
        },
    },
    "video_ltx2_i2v.json": {
        "slug": "ltx2-i2v",
        "display_name": "LTX 2.0 I2V",
        "task_type": "image-to-video",
        "description": "LTX Video 2.0 image-to-video generation",

        "output_name": "Stimma-LTX2-I2V.json",
        "prompt_node": 3,
        "prompt_input": "text",
        "seed_node": 11,          # RandomNoise (stage 1, exposed by proxy)
        "seed_input": "noise_seed",
        "steps_node": 9,
        "steps_input": "steps",
        "has_steps": True,
        "default_steps": 20,
        "frame_count_node": 62,
        "frame_count_input": "value",
        "default_frame_count": 121,
        "image_source_node": 95,  # VAEDecode
        "image_source_slot": 0,
        "fps": 25,
        "save_video_node": 75,
        "create_video_node": 97,
        "is_i2v": True,
        "load_image_node": 98,
        "resize_node": 102,       # ResizeImageMaskNode
        "checkpoint_node": 1,
        "text_encoder_node": 60,
        "lora_path_filter": "*ltx*;*LTX*",
    },
    "video_ltx2_i2v_distilled.json": {
        "slug": "ltx2-i2v-distilled",
        "display_name": "LTX 2.0 I2V Distilled",
        "task_type": "image-to-video",
        "description": "LTX Video 2.0 distilled image-to-video (faster, fewer steps)",

        "output_name": "Stimma-LTX2-I2V-Distilled.json",
        "prompt_node": 3,
        "prompt_input": "text",
        "seed_node": 11,
        "seed_input": "noise_seed",
        "has_steps": False,
        "frame_count_node": 62,
        "frame_count_input": "value",
        "default_frame_count": 121,
        "image_source_node": 95,  # VAEDecode
        "image_source_slot": 0,
        "fps": 25,
        "save_video_node": 75,
        "create_video_node": 97,
        "is_i2v": True,
        "load_image_node": 98,
        "resize_node": 102,
        "checkpoint_node": 1,
        "text_encoder_node": 60,
        "lora_path_filter": "*ltx*;*LTX*",
        "ckpt_remap": {
            "ltx-2-19b-distilled.safetensors": "ltx-2-19b-distilled-fp8.safetensors",
        },
    },
}


def _is_control_after_generate(spec_entry):
    """Check if a spec entry has control_after_generate flag."""
    if isinstance(spec_entry, list) and len(spec_entry) > 1:
        opts = spec_entry[1] if isinstance(spec_entry[1], dict) else {}
        return opts.get("control_after_generate", False)
    return False


def convert_ui_to_api(workflow, object_info):
    """Convert a UI-format workflow to API format using object_info.

    Returns a flat dict: { "node_id": { "class_type": "...", "inputs": {...} } }
    """
    CONNECTION_TYPES = {
        "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
        "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "CONTROL_NET", "STYLE_MODEL",
        "GLIGEN", "UPSCALE_MODEL", "TAESD", "PHOTOMAKER", "CLIP_VISION",
        "CLIP_VISION_OUTPUT", "INSIGHTFACE", "IPADAPTER", "OPTICAL_FLOW",
        "LATENT_OPERATION",
    }
    SKIP_TYPES = {"Note", "MarkdownNote", "Reroute", "PrimitiveNode"}

    prompt = {}
    nodes = {n["id"]: n for n in workflow.get("nodes", [])}
    links_by_id = {}
    for link in workflow.get("links", []):
        lid, src_node, src_slot, dst_node, dst_slot, ltype = link
        links_by_id[lid] = (src_node, src_slot, dst_node, dst_slot, ltype)

    muted_ids = {n["id"] for n in workflow.get("nodes", []) if n.get("mode") == 4}

    def resolve_source(src_id, src_slot, visited=None):
        if visited is None:
            visited = set()
        if src_id in visited:
            return None
        visited.add(src_id)
        if src_id not in muted_ids:
            return (src_id, src_slot)
        muted_node = nodes.get(src_id)
        if not muted_node:
            return None
        muted_inputs = muted_node.get("inputs", [])
        if src_slot < len(muted_inputs):
            upstream_link_id = muted_inputs[src_slot].get("link")
            if upstream_link_id and upstream_link_id in links_by_id:
                up_src, up_slot, _, _, _ = links_by_id[upstream_link_id]
                return resolve_source(up_src, up_slot, visited)
        return None

    for node in workflow.get("nodes", []):
        if node.get("mode") == 4:
            continue
        ntype = node.get("type", "")
        if not ntype or ntype in SKIP_TYPES:
            continue

        nid = str(node["id"])
        inputs_dict = {}

        # Build linked inputs and converted widgets sets
        linked_inputs = {}
        converted_widgets = set()
        for inp in node.get("inputs", []):
            name = inp.get("name", "")
            if inp.get("link") is not None:
                linked_inputs[name] = inp["link"]
            if "widget" in inp:
                converted_widgets.add(name)

        # Use object_info spec for input ordering
        spec = object_info.get(ntype, {})
        input_spec = spec.get("input", {})
        required_spec = input_spec.get("required", {})
        optional_spec = input_spec.get("optional", {})
        all_spec = {**required_spec, **optional_spec}

        input_order_data = spec.get("input_order", {})
        ordered_names = (input_order_data.get("required", []) +
                         input_order_data.get("optional", []))
        if not ordered_names:
            ordered_names = list(required_spec.keys()) + list(optional_spec.keys())

        wv = node.get("widgets_values", []) or []
        widget_idx = 0

        for input_name in ordered_names:
            spec_entry = all_spec.get(input_name)
            if spec_entry is None:
                continue

            is_connection = False
            if isinstance(spec_entry, list) and len(spec_entry) > 0:
                type_info = spec_entry[0]
                if isinstance(type_info, str) and type_info.upper() in CONNECTION_TYPES:
                    is_connection = True

            if is_connection:
                if input_name in linked_inputs:
                    link_data = links_by_id.get(linked_inputs[input_name])
                    if link_data:
                        src_id, src_slot, _, _, _ = link_data
                        resolved = resolve_source(src_id, src_slot)
                        if resolved:
                            inputs_dict[input_name] = [str(resolved[0]), resolved[1]]
            else:
                if input_name in linked_inputs:
                    link_data = links_by_id.get(linked_inputs[input_name])
                    if link_data:
                        src_id, src_slot, _, _, _ = link_data
                        resolved = resolve_source(src_id, src_slot)
                        if resolved:
                            inputs_dict[input_name] = [str(resolved[0]), resolved[1]]
                    if input_name in converted_widgets and widget_idx < len(wv):
                        widget_idx += 1
                        if _is_control_after_generate(spec_entry):
                            if widget_idx < len(wv):
                                widget_idx += 1
                else:
                    if widget_idx < len(wv):
                        inputs_dict[input_name] = wv[widget_idx]
                        widget_idx += 1
                        if _is_control_after_generate(spec_entry):
                            if widget_idx < len(wv):
                                widget_idx += 1

        # Fallback for nodes not in object_info
        if not spec and node.get("inputs"):
            widget_idx = 0
            for inp in node.get("inputs", []):
                if inp.get("link") is not None:
                    link_data = links_by_id.get(inp["link"])
                    if link_data:
                        src_id, src_slot, _, _, _ = link_data
                        resolved = resolve_source(src_id, src_slot)
                        if resolved:
                            inputs_dict[inp["name"]] = [str(resolved[0]), resolved[1]]
                    if "widget" in inp and widget_idx < len(wv):
                        widget_idx += 1
                        if (widget_idx < len(wv) and
                            isinstance(wv[widget_idx], str) and
                            wv[widget_idx] in ("randomize", "fixed", "increment", "decrement")):
                            widget_idx += 1
                elif "widget" in inp:
                    if widget_idx < len(wv):
                        inputs_dict[inp["name"]] = wv[widget_idx]
                        widget_idx += 1
                        if (widget_idx < len(wv) and
                            isinstance(wv[widget_idx], str) and
                            wv[widget_idx] in ("randomize", "fixed", "increment", "decrement")):
                            widget_idx += 1

        prompt[nid] = {"class_type": ntype, "inputs": inputs_dict}

    return prompt


def insert_lora_loader_api(prompt, checkpoint_node, text_encoder_node, path_filter, ui_order=50):
    """Insert a StimmaLoraLoader into an API-format prompt.

    Intercepts MODEL from checkpoint_node (slot 0) and CLIP from text_encoder_node (slot 0),
    rewires all downstream consumers to use the LoRA loader's outputs instead.
    """
    # Find a free node ID
    used_ids = {int(k) for k in prompt.keys()}
    lora_id = str(max(used_ids) + 1)

    ckpt_str = str(checkpoint_node)
    text_enc_str = str(text_encoder_node)

    # Rewire: anything that references checkpoint MODEL (slot 0) → now references lora (slot 0)
    # Rewire: anything that references text_encoder CLIP (slot 0) → now references lora (slot 1)
    for nid, node in prompt.items():
        if nid == lora_id:
            continue
        for iname, ival in node.get("inputs", {}).items():
            if isinstance(ival, list) and len(ival) == 2:
                if str(ival[0]) == ckpt_str and ival[1] == 0:
                    node["inputs"][iname] = [lora_id, 0]
                elif str(ival[0]) == text_enc_str and ival[1] == 0:
                    node["inputs"][iname] = [lora_id, 1]

    # Create the StimmaLoraLoader node
    prompt[lora_id] = {
        "class_type": "StimmaLoraLoader",
        "inputs": {
            "model": [ckpt_str, 0],
            "clip": [text_enc_str, 0],
            "path_filter": path_filter,
            "ui_order": ui_order,
        },
    }

    return lora_id


def fetch_object_info(comfy_url="http://localhost:8188"):
    """Fetch object_info from a running ComfyUI instance."""
    req = urllib.request.Request(f"{comfy_url}/object_info")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def process_workflow(input_file, config, output_dir, object_info=None):
    """Process a single workflow: flatten + add Stimma nodes + convert to API format."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"  Output: {config['output_name']}")

    with open(input_file) as fp:
        workflow = json.load(fp)

    # Step 1: Flatten the subgraph
    print("  Flattening subgraph...")
    flattener = WorkflowFlattener(workflow)
    flat_wf = flattener.flatten(remove_markdown_notes=True)

    # Step 1b: Fix checkpoint names if needed (e.g., distilled models)
    # The distilled models might reference non-fp8 versions that aren't available locally
    ckpt_remap = config.get("ckpt_remap", {})
    if ckpt_remap:
        for node in flat_wf["nodes"]:
            wv = node.get("widgets_values", [])
            for i, val in enumerate(wv):
                if isinstance(val, str) and val in ckpt_remap:
                    wv[i] = ckpt_remap[val]

    # Step 2: Add Stimma nodes
    print("  Adding Stimma nodes...")
    adder = StimmaNodeAdder(flat_wf)

    # Calculate positions - Stimma nodes go to the LEFT of the workflow
    base_x = -1400
    y_cursor = 3400

    # StimmaToolInfo
    adder.add_tool_info(
        slug=config["slug"],
        display_name=config["display_name"],
        task_type=config["task_type"],
        description=config["description"],
        pos=[base_x, y_cursor],
    )
    y_cursor += 320

    # StimmaPromptParam
    adder.add_prompt_input(
        name="prompt",
        default_text="",
        required=True,
        ui_order=0,
        ui_description="Describe the video you want to generate",
        pos=[base_x, y_cursor],
        wire_to={"node_id": config["prompt_node"], "input_name": config["prompt_input"]},
    )
    y_cursor += 400

    # StimmaImageParam (for I2V variants)
    if config["is_i2v"]:
        img_node = adder.add_image_input(
            ui_control="videoFramePicker",
            ui_order=2,
            pos=[base_x, y_cursor],
        )
        # Replace LoadImage with StimmaImageParam
        adder.replace_load_image_with_stimma(config["load_image_node"], img_node)
        y_cursor += 250

    # Resolution handling
    video_settings_params = []
    if config["is_i2v"]:
        # I2V: megapixels param → ImageScaleToTotalPixels replaces ResizeImageMaskNode
        megapixels_node = adder.add_float_param(
            name="megapixels",
            value=0.9,
            minimum=0.1,
            maximum=2.0,
            step=0.1,
            ui_control="slider",
            ui_order=3,
            ui_description="Target resolution in megapixels",
            pos=[base_x, y_cursor],
        )
        y_cursor += 400

        # Add ImageScaleToTotalPixels node, wired from StimmaImageParam
        scale_node = adder.add_image_scale_to_total_pixels(
            megapixels_node_id=megapixels_node["id"],
            image_src_node_id=img_node["id"],
            image_src_slot=0,
            pos=[base_x + 500, y_cursor - 300],
        )

        # Rewire: ResizeImageMaskNode(102) outputs → ImageScaleToTotalPixels outputs
        resize_id = config["resize_node"]
        adder.rewire_image_output(resize_id, 0, scale_node["id"], 0)
        # Mute the old ResizeImageMaskNode
        adder.mute_node(resize_id)

        video_settings_params = ["megapixels", "duration", "fps"]
    else:
        # T2V: StimmaResolutionParam → EmptyImage width/height
        empty_image_id = config["empty_image_node"]
        adder.add_resolution_input(
            width=1280,
            height=720,
            step=32,
            min_size=480,
            max_size=1280,
            ui_order=2,
            pos=[base_x, y_cursor],
            wire_to_width={"node_id": empty_image_id, "input_name": "width"},
            wire_to_height={"node_id": empty_image_id, "input_name": "height"},
        )
        y_cursor += 350
        video_settings_params = ["width", "height", "duration", "fps"]

    # StimmaFloatParam - duration (in seconds)
    duration_node = adder.add_float_param(
        name="duration",
        value=5.0,
        minimum=0.5,
        maximum=15.0,
        step=0.5,
        ui_control="slider",
        ui_order=4,
        ui_description="Video duration in seconds",
        pos=[base_x, y_cursor],
    )
    y_cursor += 400

    # StimmaIntParam - fps
    fps_node = adder.add_int_param(
        name="fps",
        value=config["fps"],
        minimum=8,
        maximum=60,
        step=1,
        ui_control="input",
        ui_order=5,
        ui_description="Frames per second",
        pos=[base_x + 500, y_cursor - 400],
    )

    # StimmaDurationToFrames - converts duration+fps to frame count (8n+1 for LTX2)
    d2f_node = adder.add_duration_to_frames(
        duration_node_id=duration_node["id"],
        fps_node_id=fps_node["id"],
        frame_step=8,
        pos=[base_x + 500, y_cursor - 200],
    )
    # Wire frame count to the target node
    adder._wire(
        d2f_node["id"], 0,
        config["frame_count_node"], config["frame_count_input"],
        "INT",
    )

    # StimmaIntParam - steps (only for non-distilled)
    advanced_params = ["loras", "seed"]
    if config.get("has_steps"):
        adder.add_int_param(
            name="steps",
            value=config["default_steps"],
            minimum=1,
            maximum=50,
            step=1,
            ui_control="input",
            ui_order=20,
            ui_description="Number of sampling steps",
            pos=[base_x + 500, y_cursor],
            wire_to={"node_id": config["steps_node"], "input_name": config["steps_input"]},
        )
        advanced_params.insert(0, "steps")
        y_cursor += 400

    # StimmaSeedParam
    adder.add_seed_input(
        name="seed",
        value=0,
        ui_order=80,
        pos=[base_x + 500, y_cursor],
        wire_to={"node_id": config["seed_node"], "input_name": config["seed_input"]},
    )
    y_cursor += 200

    # StimmaVideoOutput - wire to IMAGE frames source, with fps from param
    save_node = adder.nodes_by_id.get(config["save_video_node"])
    save_pos = save_node.get("pos", [800, 3800]) if save_node else [800, 3800]
    video_out = adder.add_video_output(
        fps=config["fps"],
        pos=[save_pos[0], save_pos[1] - 350],
        src_node_id=config["image_source_node"],
        src_slot=config["image_source_slot"],
    )
    # Wire fps param to video output
    adder._wire(fps_node["id"], 0, video_out["id"], "fps", "INT")

    # Mute original SaveVideo
    adder.mute_node(config["save_video_node"])

    # Layout groups
    adder.add_layout_group(
        label="Video Settings",
        param_names=video_settings_params,
        collapsed=False,
        ui_order=0,
        pos=[base_x - 500, y_cursor],
    )
    adder.add_layout_group(
        label="Advanced",
        param_names=advanced_params,
        collapsed=True,
        ui_order=10,
        pos=[base_x - 500, y_cursor + 280],
    )

    # Finalize UI format
    ui_result = adder.finalize()

    # Cleanup Stimma node placement and visual groups for readability.
    cleanup_workflow(ui_result)

    # Convert to API format if object_info is available
    if object_info:
        print("  Converting to API format...")
        result = convert_ui_to_api(ui_result, object_info)
    else:
        result = ui_result

    # Insert StimmaLoraLoader (operates on API format)
    if isinstance(result, dict) and "nodes" not in result:
        ckpt_node = config.get("checkpoint_node")
        text_enc_node = config.get("text_encoder_node")
        if ckpt_node and text_enc_node:
            print("  Inserting StimmaLoraLoader...")
            insert_lora_loader_api(
                result,
                checkpoint_node=ckpt_node,
                text_encoder_node=text_enc_node,
                path_filter=config.get("lora_path_filter", ""),
            )

    # Write output
    output_path = output_dir / config["output_name"]
    output_path.write_text(json.dumps(result, indent=2))
    print(f"  Written to: {output_path}")
    return output_path


def main():
    output_dir = WORKFLOWS_DIR

    # Fetch object_info from ComfyUI for UI→API conversion
    print("Fetching object_info from ComfyUI...")
    try:
        object_info = fetch_object_info()
        print(f"  Got {len(object_info)} node types")
    except Exception as e:
        print(f"  Warning: Could not fetch object_info ({e})")
        print("  Output will be in UI format (may not work with Stimma executor)")
        object_info = None

    for input_name, config in CONFIGS.items():
        input_path = WORKFLOWS_DIR / input_name
        if not input_path.exists():
            print(f"Skipping {input_name}: file not found")
            continue
        process_workflow(input_path, config, output_dir, object_info)

    print(f"\nDone! Generated {len(CONFIGS)} Stimma workflows.")


if __name__ == "__main__":
    main()
