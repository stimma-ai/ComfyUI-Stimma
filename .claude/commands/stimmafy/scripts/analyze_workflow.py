#!/usr/bin/env python3
"""Analyze a ComfyUI workflow for Stimma-ification.

Parses a UI-format workflow JSON file, queries ComfyUI's object_info endpoint
for node specs, and outputs a structured JSON report of the workflow's key
components and heuristic recommendations for Stimma node additions.

Usage:
    python3 analyze_workflow.py <workflow.json> [--comfy-url http://localhost:8188]
"""

import json
import sys
import argparse
import urllib.request
from pathlib import Path

# --- Node type categories ---

MODEL_LOADERS = {
    "CheckpointLoaderSimple", "UNETLoader", "CheckpointLoader",
    "unCLIPCheckpointLoader",
}
CLIP_LOADERS = {"CLIPLoader", "DualCLIPLoader", "TripleCLIPLoader"}
VAE_LOADERS = {"VAELoader"}
COMBINED_LOADERS = {"CheckpointLoaderSimple", "CheckpointLoader"}  # output MODEL+CLIP+VAE

KSAMPLER_TYPES = {"KSampler", "KSamplerAdvanced"}
ADVANCED_SAMPLER_TYPES = {"SamplerCustomAdvanced", "SamplerCustom"}
ALL_SAMPLER_TYPES = KSAMPLER_TYPES | ADVANCED_SAMPLER_TYPES

TEXT_ENCODERS = {"CLIPTextEncode", "CLIPTextEncodeFlux", "CLIPTextEncodeSD3"}
SAVE_IMAGE_NODES = {"SaveImage", "SaveAnimatedWEBP", "SaveAnimatedPNG", "PreviewImage"}
SAVE_VIDEO_NODES = {"VHS_VideoCombine", "SaveVideo"}
LATENT_IMAGE_NODES = {"EmptyLatentImage", "EmptySD3LatentImage"}
LORA_NODES = {"LoraLoader", "LoraLoaderModelOnly"}

GUIDANCE_NODES = {"FluxGuidance"}
MODEL_SAMPLING_NODES = {"ModelSamplingFlux", "ModelSamplingAuraFlow", "ModelSamplingSD3"}
NOISE_NODES = {"RandomNoise"}
SCHEDULER_NODES = {"BasicScheduler", "KarrasScheduler", "ExponentialScheduler",
                   "PolyexponentialScheduler", "SDTurboScheduler", "AlignYourStepsScheduler"}
SAMPLER_SELECT_NODES = {"KSamplerSelect"}
GUIDER_NODES = {"BasicGuider", "CFGGuider", "DualCFGGuider"}
IMAGE_LOAD_NODES = {"LoadImage"}
VIDEO_LOAD_NODES = {"VHS_LoadVideo", "LoadVideo"}

# Types that are connections, not widgets
CONNECTION_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
    "NOISE", "GUIDER", "SAMPLER", "SIGMAS", "CONTROL_NET", "STYLE_MODEL",
    "GLIGEN", "UPSCALE_MODEL", "TAESD", "PHOTOMAKER", "CLIP_VISION",
    "CLIP_VISION_OUTPUT", "INSIGHTFACE", "IPADAPTER",
}

GUIDANCE_DISTILLED_KEYWORDS = {"lightning", "turbo", "schnell", "klein", "hyper"}
FLUX_KEYWORDS = {"flux"}
# Flux is always guidance-distilled for our purposes (no negative prompt, cfg=1 or uses FluxGuidance)


def fetch_object_info(comfy_url):
    """Fetch node specifications from ComfyUI."""
    url = f"{comfy_url}/object_info"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"Warning: Could not fetch object_info from {comfy_url}: {e}", file=sys.stderr)
        return None


def build_graph(workflow):
    """Build node and link lookup structures."""
    nodes = {}
    for n in workflow.get("nodes", []):
        nodes[n["id"]] = n

    links = {}
    for link in workflow.get("links", []):
        lid, src_node, src_slot, dst_node, dst_slot, ltype = link
        links[lid] = {
            "id": lid,
            "src_node": src_node,
            "src_slot": src_slot,
            "dst_node": dst_node,
            "dst_slot": dst_slot,
            "type": ltype,
        }

    return nodes, links


def get_output_links(node, links):
    """Get all outgoing links from a node, grouped by output slot."""
    result = {}
    for output in node.get("outputs", []):
        slot_links = output.get("links") or []
        slot_name = output.get("name", "")
        for lid in slot_links:
            if lid in links:
                result.setdefault(slot_name, []).append(links[lid])
    return result


def get_input_links(node, links):
    """Get all incoming links to a node."""
    result = {}
    for inp in node.get("inputs", []):
        lid = inp.get("link")
        if lid and lid in links:
            result[inp["name"]] = links[lid]
    return result


def extract_widget_values_ksampler(node):
    """Extract parameter values from a KSampler node.
    widgets_values order: seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise
    """
    wv = node.get("widgets_values", [])
    if len(wv) >= 7:
        return {
            "seed": wv[0],
            "steps": wv[2],
            "cfg": wv[3],
            "sampler_name": wv[4],
            "scheduler": wv[5],
            "denoise": wv[6],
        }
    return {}


def extract_widget_values_ksampler_advanced(node):
    """Extract from KSamplerAdvanced.
    widgets_values order: add_noise, noise_seed, control_after_generate, steps, cfg,
                          sampler_name, scheduler, start_at_step, end_at_step, return_with_leftover_noise
    """
    wv = node.get("widgets_values", [])
    if len(wv) >= 10:
        return {
            "add_noise": wv[0],
            "seed": wv[1],
            "steps": wv[3],
            "cfg": wv[4],
            "sampler_name": wv[5],
            "scheduler": wv[6],
            "start_at_step": wv[7],
            "end_at_step": wv[8],
        }
    return {}


def extract_widget_values_basic_scheduler(node):
    """widgets_values: scheduler, steps, denoise"""
    wv = node.get("widgets_values", [])
    if len(wv) >= 3:
        return {"scheduler": wv[0], "steps": wv[1], "denoise": wv[2]}
    return {}


def extract_widget_values_sampler_select(node):
    """widgets_values: sampler_name"""
    wv = node.get("widgets_values", [])
    return {"sampler_name": wv[0]} if wv else {}


def extract_widget_values_random_noise(node):
    """widgets_values: seed, control_after_generate"""
    wv = node.get("widgets_values", [])
    return {"seed": wv[0]} if wv else {}


def extract_widget_values_flux_guidance(node):
    """widgets_values: guidance"""
    wv = node.get("widgets_values", [])
    return {"guidance": wv[0]} if wv else {}


def extract_widget_values_model_sampling(node, ntype):
    """Extract from ModelSamplingFlux/AuraFlow."""
    wv = node.get("widgets_values", [])
    if ntype == "ModelSamplingFlux" and len(wv) >= 4:
        return {"max_shift": wv[0], "base_shift": wv[1], "width": wv[2], "height": wv[3]}
    elif ntype == "ModelSamplingAuraFlow" and wv:
        return {"shift": wv[0]}
    return {}


def extract_text_from_encoder(node):
    """Get the text from a CLIPTextEncode node."""
    wv = node.get("widgets_values", [])
    if wv:
        return wv[0] if isinstance(wv[0], str) else ""
    return ""


def extract_model_name(node, ntype):
    """Get the model/checkpoint filename from a loader node."""
    wv = node.get("widgets_values", [])
    if ntype == "CheckpointLoaderSimple" and wv:
        return wv[0]
    elif ntype == "UNETLoader" and wv:
        return wv[0]
    elif ntype == "CheckpointLoader" and len(wv) >= 2:
        return wv[1]  # config_name, ckpt_name
    return ""


def extract_latent_dimensions(node, ntype):
    """Get width/height from a latent image node."""
    wv = node.get("widgets_values", [])
    if ntype in ("EmptyLatentImage", "EmptySD3LatentImage") and len(wv) >= 2:
        return {"width": wv[0], "height": wv[1]}
    return {}


def detect_model_family(model_loaders, guidance_nodes, model_sampling_nodes, all_nodes):
    """Detect model family from loader names and node types."""
    model_names = [ml["model_name"].lower() for ml in model_loaders if ml.get("model_name")]

    # Check for specific model families
    for name in model_names:
        if any(k in name for k in ["flux"]):
            return "flux"
        if any(k in name for k in ["sdxl", "sd_xl"]):
            return "sdxl"
        if any(k in name for k in ["sd3", "sd3.5"]):
            return "sd3"
        if any(k in name for k in ["z_image", "zimage", "lumina"]):
            return "z-image"
        if any(k in name for k in ["qwen", "q-image"]):
            return "qwen"
        if any(k in name for k in ["wan", "wan2"]):
            return "wan"
        if any(k in name for k in ["hunyuan"]):
            return "hunyuan"
        if any(k in name for k in ["pixart"]):
            return "pixart"
        if any(k in name for k in ["kolors"]):
            return "kolors"

    # Check by node types present
    node_types = {n["type"] for n in all_nodes.values() if not n.get("mode") == 4}
    if "ModelSamplingFlux" in node_types or "FluxGuidance" in node_types:
        return "flux"
    if "ModelSamplingAuraFlow" in node_types:
        return "z-image"  # or lumina-based
    if any("SD3" in t for t in node_types):
        return "sd3"
    if "CLIPTextEncodeFlux" in node_types:
        return "flux"

    return "unknown"


def detect_guidance_distilled(model_loaders, samplers, guidance_nodes, model_family):
    """Detect if the model is guidance-distilled (no CFG needed)."""
    model_names = [ml["model_name"].lower() for ml in model_loaders if ml.get("model_name")]

    # Keyword detection
    for name in model_names:
        if any(k in name for k in GUIDANCE_DISTILLED_KEYWORDS):
            return True

    # Flux family is always guidance-distilled for CFG purposes
    if model_family == "flux":
        return True

    # CFG = 1.0 in any sampler is a strong signal
    for s in samplers:
        params = s.get("params", {})
        cfg = params.get("cfg")
        if cfg is not None and float(cfg) == 1.0:
            return True

    # Z-Image Turbo is guidance-distilled
    if model_family == "z-image":
        return True

    return False


def detect_task_type(all_nodes, links):
    """Guess the task type from workflow structure."""
    node_types = {n["type"] for nid, n in all_nodes.items() if n.get("mode", 0) != 4}

    has_image_input = bool(IMAGE_LOAD_NODES & node_types)
    has_video_input = bool(VIDEO_LOAD_NODES & node_types)
    has_video_output = bool(SAVE_VIDEO_NODES & node_types)
    has_image_output = bool(SAVE_IMAGE_NODES & node_types)

    if has_video_input and has_video_output:
        return "video-to-video"
    if has_video_output and not has_video_input:
        if has_image_input:
            return "image-to-video"
        return "text-to-video"
    if has_image_input:
        # Could be img2img, inpaint, style-transfer, upscale
        return "image-to-image"

    return "text-to-image"


def find_clip_source(model_loaders, clip_loaders, nodes, links):
    """Find where CLIP comes from for StimmaLoraLoader wiring."""
    # First check if a combined loader (CheckpointLoaderSimple) provides CLIP
    for ml in model_loaders:
        node = nodes[ml["id"]]
        if node["type"] in COMBINED_LOADERS:
            for output in node.get("outputs", []):
                if output.get("type") == "CLIP" or output.get("name") == "CLIP":
                    slot = node["outputs"].index(output)
                    return {"node_id": ml["id"], "slot": slot}

    # Otherwise look for dedicated CLIP loaders
    for cl in clip_loaders:
        node = nodes[cl["id"]]
        for output in node.get("outputs", []):
            if output.get("type") == "CLIP":
                slot = node["outputs"].index(output)
                return {"node_id": cl["id"], "slot": slot}

    return None


def _scan_inner_sampling(def_nodes):
    """Collect sampling-relevant nodes from a subgraph definition's node list.

    Mirrors the top-level sampler/scheduler scan but for nodes living inside a
    subgraph definition (which build_graph never sees). Returns the same shaped
    dicts used elsewhere so widget extractors and downstream logic apply.
    """
    found = {
        "samplers": [], "scheduler_nodes": [], "sampler_select_nodes": [],
        "noise_nodes": [], "guidance_nodes": [], "text_encoders": [],
    }
    for n in def_nodes:
        t = n.get("type")
        muted = n.get("mode") == 4
        nid = n.get("id")
        if t in KSAMPLER_TYPES:
            params = (extract_widget_values_ksampler(n) if t == "KSampler"
                      else extract_widget_values_ksampler_advanced(n))
            found["samplers"].append({"id": nid, "type": t, "muted": muted,
                                      "params": params, "pipeline_type": "standard"})
        elif t in ADVANCED_SAMPLER_TYPES:
            found["samplers"].append({"id": nid, "type": t, "muted": muted,
                                      "params": {}, "pipeline_type": "advanced"})
        elif t in SCHEDULER_NODES:
            params = extract_widget_values_basic_scheduler(n) if t == "BasicScheduler" else {}
            found["scheduler_nodes"].append({"id": nid, "type": t, "muted": muted, "params": params})
        elif t in SAMPLER_SELECT_NODES:
            found["sampler_select_nodes"].append({"id": nid, "type": t, "muted": muted,
                                                  "params": extract_widget_values_sampler_select(n)})
        elif t in NOISE_NODES:
            found["noise_nodes"].append({"id": nid, "type": t, "muted": muted,
                                         "params": extract_widget_values_random_noise(n)})
        elif t in GUIDANCE_NODES:
            found["guidance_nodes"].append({"id": nid, "type": t, "muted": muted,
                                            "params": extract_widget_values_flux_guidance(n)})
        elif t in TEXT_ENCODERS:
            found["text_encoders"].append({"id": nid, "type": t, "muted": muted,
                                           "text_preview": extract_text_from_encoder(n)[:120]})
    return found


def collect_subgraphs(node_list, subgraph_defs, chain=(), seen_defs=frozenset()):
    """Walk top-level nodes, descend into subgraph definitions (recursively), and
    report sampling-relevant inner nodes per subgraph occurrence.

    `chain` is the list of top→inner subgraph instance node ids leading to this
    occurrence (length 1 = directly under the top level). The cycle guard
    (`seen_defs`) prevents infinite recursion on self-referential definitions.
    """
    occurrences = []
    for n in node_list:
        defn = subgraph_defs.get(n.get("type"))
        if defn is None:
            continue
        def_id = defn.get("id")
        if def_id in seen_defs:
            continue
        inst_id = n.get("id")
        new_chain = chain + (inst_id,)
        inner = defn.get("nodes", [])
        occurrences.append({
            "instance_node_id": inst_id,
            "definition_id": def_id,
            "title": n.get("title") or defn.get("name") or "",
            "chain": list(new_chain),
            **_scan_inner_sampling(inner),
        })
        occurrences.extend(
            collect_subgraphs(inner, subgraph_defs, new_chain, seen_defs | {def_id})
        )
    return occurrences


def build_subgraph_prep_suggestions(occurrences, guidance_distilled):
    """From subgraph occurrences that contain samplers, produce ready-to-use
    `subgraph_prep.add_inner_inputs` entries plus human warnings.

    Output entry shape matches references/node-catalog.md so it can be dropped
    into a plan's `subgraph_prep` section, with an extra `param` field naming the
    Stimma param to wire to the new boundary input.
    """
    suggestions = []
    warnings = []
    for occ in occurrences:
        samplers = [s for s in occ["samplers"] if not s["muted"]]
        scheds = [s for s in occ["scheduler_nodes"] if not s["muted"]]
        selects = [s for s in occ["sampler_select_nodes"] if not s["muted"]]
        noises = [s for s in occ["noise_nodes"] if not s["muted"]]
        guids = [g for g in occ["guidance_nodes"] if not g["muted"]]
        if not (samplers or scheds or selects):
            continue

        if len(occ["chain"]) > 1:
            warnings.append(
                f"Sampler is nested {len(occ['chain'])} subgraph levels deep "
                f"(instance chain {occ['chain']}). Expose steps/sampler/scheduler "
                f"through EACH subgraph boundary with subgraph_prep, innermost first."
            )
            continue

        sg_id = occ["instance_node_id"]
        expose = []

        def add(name, typ, inner_node_id, widget):
            expose.append({"name": name, "type": typ, "inner_node_id": inner_node_id,
                           "inner_widget_name": widget, "param": name})

        # Standard KSampler carries all knobs on one node.
        for s in samplers:
            if s["type"] in KSAMPLER_TYPES:
                seed_widget = "noise_seed" if s["type"] == "KSamplerAdvanced" else "seed"
                add("steps", "INT", s["id"], "steps")
                add("sampler_name", "COMBO", s["id"], "sampler_name")
                add("scheduler", "COMBO", s["id"], "scheduler")
                add("seed", "INT", s["id"], seed_widget)
                if not guidance_distilled:
                    add("cfg", "FLOAT", s["id"], "cfg")
        # Advanced pipeline spreads knobs across helper nodes.
        for s in selects:
            add("sampler_name", "COMBO", s["id"], "sampler_name")
        for s in scheds:
            add("scheduler", "COMBO", s["id"], "scheduler")
            add("steps", "INT", s["id"], "steps")
        for s in noises:
            add("seed", "INT", s["id"], "seed")
        for g in guids:
            add("guidance", "FLOAT", g["id"], "guidance")

        # Dedupe by param name (keep first); flag if multiple samplers collided.
        seen = set()
        deduped = []
        collided = False
        for e in expose:
            if e["param"] in seen:
                collided = True
                continue
            seen.add(e["param"])
            deduped.append(e)
        if len(samplers) > 1 or collided:
            warnings.append(
                f"Subgraph node {sg_id} has multiple samplers/sources for the same "
                f"knob — decide whether to share one param across them or expose "
                f"separate per-stage params (see SKILL 'Handling multiple samplers')."
            )
        if deduped:
            suggestions.append({
                "subgraph_node_id": sg_id,
                "definition_id": occ["definition_id"],
                "expose": deduped,
            })
    return suggestions, warnings


def analyze(workflow_path, comfy_url="http://localhost:8188"):
    """Main analysis function."""
    workflow = json.loads(Path(workflow_path).read_text())
    object_info = fetch_object_info(comfy_url)
    nodes, links = build_graph(workflow)

    result = {
        "file_path": str(Path(workflow_path).resolve()),
        "node_count": len(nodes),
        "link_count": len(links),
        "model_loaders": [],
        "clip_loaders": [],
        "vae_loaders": [],
        "samplers": [],
        "text_encoders": [],
        "save_image_nodes": [],
        "save_video_nodes": [],
        "latent_image_nodes": [],
        "lora_nodes": [],
        "guidance_nodes": [],
        "model_sampling_nodes": [],
        "noise_nodes": [],
        "scheduler_nodes": [],
        "sampler_select_nodes": [],
        "guider_nodes": [],
        "image_load_nodes": [],
        "video_load_nodes": [],
        "model_family": "unknown",
        "guidance_distilled": False,
        "task_type": "text-to-image",
        "has_existing_stimma_nodes": False,
        "subgraphs": [],
        "recommendations": {},
    }

    # Check for existing Stimma nodes
    stimma_types = {n["type"] for n in nodes.values() if n["type"].startswith("Stimma")}
    result["has_existing_stimma_nodes"] = bool(stimma_types)
    if stimma_types:
        result["existing_stimma_types"] = sorted(stimma_types)

    for nid, node in nodes.items():
        ntype = node["type"]
        is_muted = node.get("mode") == 4

        if ntype in MODEL_LOADERS:
            model_name = extract_model_name(node, ntype)
            outputs = get_output_links(node, links)
            entry = {
                "id": nid, "type": ntype, "model_name": model_name,
                "muted": is_muted, "outputs": {k: [l["id"] for l in v] for k, v in outputs.items()},
                "is_combined": ntype in COMBINED_LOADERS,
            }
            result["model_loaders"].append(entry)

        elif ntype in CLIP_LOADERS:
            wv = node.get("widgets_values", [])
            result["clip_loaders"].append({
                "id": nid, "type": ntype, "muted": is_muted,
                "widgets_values": wv,
            })

        elif ntype in VAE_LOADERS:
            wv = node.get("widgets_values", [])
            result["vae_loaders"].append({
                "id": nid, "type": ntype, "muted": is_muted,
                "vae_name": wv[0] if wv else "",
            })

        elif ntype in KSAMPLER_TYPES:
            if ntype == "KSampler":
                params = extract_widget_values_ksampler(node)
            else:
                params = extract_widget_values_ksampler_advanced(node)
            input_links = get_input_links(node, links)
            result["samplers"].append({
                "id": nid, "type": ntype, "muted": is_muted,
                "params": params,
                "input_links": {k: v["id"] for k, v in input_links.items()},
                "pipeline_type": "standard",
            })

        elif ntype in ADVANCED_SAMPLER_TYPES:
            input_links = get_input_links(node, links)
            result["samplers"].append({
                "id": nid, "type": ntype, "muted": is_muted,
                "params": {},  # Params are on sub-nodes
                "input_links": {k: v["id"] for k, v in input_links.items()},
                "pipeline_type": "advanced",
            })

        elif ntype in TEXT_ENCODERS:
            text = extract_text_from_encoder(node)
            input_links = get_input_links(node, links)
            # Guess role from title or connections
            title = node.get("title", "").lower()
            role = "unknown"
            if "negative" in title or "neg" in title:
                role = "negative"
            elif "positive" in title or "pos" in title:
                role = "positive"
            else:
                # Check if it connects to KSampler positive or negative input
                out_links = get_output_links(node, links)
                for slot_links in out_links.values():
                    for link in slot_links:
                        dst = nodes.get(link["dst_node"])
                        if dst and dst["type"] in ALL_SAMPLER_TYPES:
                            dst_input = dst.get("inputs", [])
                            if link["dst_slot"] < len(dst_input):
                                input_name = dst_input[link["dst_slot"]].get("name", "")
                                if "positive" in input_name:
                                    role = "positive"
                                elif "negative" in input_name:
                                    role = "negative"
                        # Check if it connects to a guider
                        if dst and dst["type"] in GUIDER_NODES:
                            role = "positive"  # Guiders take the main conditioning
                        # Check FluxGuidance
                        if dst and dst["type"] in GUIDANCE_NODES:
                            role = "positive"

            result["text_encoders"].append({
                "id": nid, "type": ntype, "muted": is_muted,
                "text_preview": text[:200] + ("..." if len(text) > 200 else ""),
                "full_text": text,
                "role": role,
            })

        elif ntype in SAVE_IMAGE_NODES:
            wv = node.get("widgets_values", [])
            result["save_image_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted,
                "filename_prefix": wv[0] if wv else "ComfyUI",
            })

        elif ntype in SAVE_VIDEO_NODES:
            result["save_video_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted,
            })

        elif ntype in LATENT_IMAGE_NODES:
            dims = extract_latent_dimensions(node, ntype)
            result["latent_image_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted, **dims,
            })

        elif ntype in LORA_NODES:
            wv = node.get("widgets_values", [])
            lora_name = wv[0] if wv else ""
            strength = wv[1] if len(wv) > 1 else 1.0
            result["lora_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted,
                "lora_name": lora_name, "strength": strength,
            })

        elif ntype in GUIDANCE_NODES:
            params = extract_widget_values_flux_guidance(node)
            result["guidance_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted, "params": params,
            })

        elif ntype in MODEL_SAMPLING_NODES:
            params = extract_widget_values_model_sampling(node, ntype)
            result["model_sampling_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted, "params": params,
            })

        elif ntype in NOISE_NODES:
            params = extract_widget_values_random_noise(node)
            result["noise_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted, "params": params,
            })

        elif ntype in SCHEDULER_NODES:
            if ntype == "BasicScheduler":
                params = extract_widget_values_basic_scheduler(node)
            else:
                params = {}
            result["scheduler_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted, "params": params,
            })

        elif ntype in SAMPLER_SELECT_NODES:
            params = extract_widget_values_sampler_select(node)
            result["sampler_select_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted, "params": params,
            })

        elif ntype in GUIDER_NODES:
            result["guider_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted,
            })

        elif ntype in IMAGE_LOAD_NODES:
            result["image_load_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted,
            })

        elif ntype in VIDEO_LOAD_NODES:
            result["video_load_nodes"].append({
                "id": nid, "type": ntype, "muted": is_muted,
            })

    # Descend into subgraph definitions for sampling-relevant inner nodes that
    # build_graph (top-level only) cannot see. Without this, a sampler buried in
    # a subgraph is invisible and steps/sampler/scheduler get silently dropped.
    definitions = workflow.get("definitions", {})
    subgraph_defs = {sg.get("id"): sg for sg in definitions.get("subgraphs", []) if sg.get("id")}
    subgraph_occurrences = collect_subgraphs(workflow.get("nodes", []), subgraph_defs)
    result["subgraphs"] = subgraph_occurrences
    inner_samplers = [s for occ in subgraph_occurrences for s in occ["samplers"]]

    # Detect model family
    result["model_family"] = detect_model_family(
        result["model_loaders"], result["guidance_nodes"],
        result["model_sampling_nodes"], nodes,
    )

    # Detect guidance-distilled — include inner subgraph samplers so a cfg=1.0
    # sampler hidden in a subgraph still trips the signal.
    result["guidance_distilled"] = detect_guidance_distilled(
        result["model_loaders"], result["samplers"] + inner_samplers,
        result["guidance_nodes"], result["model_family"],
    )

    # Detect task type
    result["task_type"] = detect_task_type(nodes, links)

    # Find CLIP source for LoRA loader wiring
    clip_source = find_clip_source(
        result["model_loaders"], result["clip_loaders"], nodes, links,
    )

    # Generate wiring recommendations
    active_model_loaders = [ml for ml in result["model_loaders"] if not ml["muted"]]
    active_samplers = [s for s in result["samplers"] if not s["muted"]]
    active_text_encoders = [te for te in result["text_encoders"] if not te["muted"]]
    active_save_image = [s for s in result["save_image_nodes"] if not s["muted"]]
    active_save_video = [s for s in result["save_video_nodes"] if not s["muted"]]
    active_latent = [l for l in result["latent_image_nodes"] if not l["muted"]]

    positive_encoders = [te for te in active_text_encoders if te["role"] == "positive"]
    negative_encoders = [te for te in active_text_encoders if te["role"] == "negative"]

    recs = {
        "expose_cfg": not result["guidance_distilled"],
        "expose_negative_prompt": not result["guidance_distilled"] and bool(negative_encoders),
        "expose_denoise": result["task_type"] in ("image-to-image", "inpaint", "outpaint"),
        "expose_guidance": bool(result["guidance_nodes"]) and result["model_family"] == "flux",
        "expose_shift": bool(result["model_sampling_nodes"]),
    }

    # Wiring targets
    wiring = {"targets": {}}

    # Prompt wiring
    if positive_encoders:
        wiring["targets"]["prompt"] = {
            "node_id": positive_encoders[0]["id"],
            "input_name": "text",
        }
    if negative_encoders and recs["expose_negative_prompt"]:
        wiring["targets"]["negative_prompt"] = {
            "node_id": negative_encoders[0]["id"],
            "input_name": "text",
        }

    # Sampler param wiring depends on pipeline type
    if active_samplers:
        sampler = active_samplers[0]
        if sampler["pipeline_type"] == "standard":
            wiring["targets"]["seed"] = {"node_id": sampler["id"], "input_name": "seed"}
            wiring["targets"]["steps"] = {"node_id": sampler["id"], "input_name": "steps"}
            wiring["targets"]["sampler_name"] = {"node_id": sampler["id"], "input_name": "sampler_name"}
            wiring["targets"]["scheduler"] = {"node_id": sampler["id"], "input_name": "scheduler"}
            if recs["expose_cfg"]:
                wiring["targets"]["cfg"] = {"node_id": sampler["id"], "input_name": "cfg"}
            if recs["expose_denoise"]:
                wiring["targets"]["denoise"] = {"node_id": sampler["id"], "input_name": "denoise"}
        else:
            # Advanced pipeline - params on sub-nodes
            if result["noise_nodes"]:
                wiring["targets"]["seed"] = {
                    "node_id": result["noise_nodes"][0]["id"],
                    "input_name": "seed",
                }
            if result["sampler_select_nodes"]:
                wiring["targets"]["sampler_name"] = {
                    "node_id": result["sampler_select_nodes"][0]["id"],
                    "input_name": "sampler_name",
                }
            if result["scheduler_nodes"]:
                sched = result["scheduler_nodes"][0]
                wiring["targets"]["scheduler"] = {"node_id": sched["id"], "input_name": "scheduler"}
                wiring["targets"]["steps"] = {"node_id": sched["id"], "input_name": "steps"}
                if recs["expose_denoise"]:
                    wiring["targets"]["denoise"] = {"node_id": sched["id"], "input_name": "denoise"}
            if result["guidance_nodes"] and recs["expose_guidance"]:
                wiring["targets"]["guidance"] = {
                    "node_id": result["guidance_nodes"][0]["id"],
                    "input_name": "guidance",
                }

    # Resolution wiring
    if active_latent:
        wiring["targets"]["resolution"] = {
            "node_id": active_latent[0]["id"],
            "width_input": "width",
            "height_input": "height",
        }

    # Model + CLIP sources for LoRA loader insertion
    if active_model_loaders:
        ml = active_model_loaders[0]
        wiring["lora_insertion"] = {
            "model_source": {"node_id": ml["id"], "slot": 0},
        }
        if ml["is_combined"]:
            wiring["lora_insertion"]["clip_source"] = {"node_id": ml["id"], "slot": 1}
        elif clip_source:
            wiring["lora_insertion"]["clip_source"] = clip_source

    # Output replacement
    wiring["output_replacements"] = []
    for save_node in active_save_image:
        if save_node["type"] != "PreviewImage":
            wiring["output_replacements"].append({
                "node_id": save_node["id"],
                "output_type": "image",
            })
    for save_node in active_save_video:
        wiring["output_replacements"].append({
            "node_id": save_node["id"],
            "output_type": "video",
        })

    # Default parameter values from current workflow
    defaults = {}
    if active_samplers and active_samplers[0]["params"]:
        defaults.update(active_samplers[0]["params"])
    if result["scheduler_nodes"]:
        defaults.update(result["scheduler_nodes"][0].get("params", {}))
    if result["noise_nodes"]:
        defaults.update(result["noise_nodes"][0].get("params", {}))
    if result["guidance_nodes"]:
        defaults.update(result["guidance_nodes"][0].get("params", {}))
    if result["model_sampling_nodes"]:
        defaults.update(result["model_sampling_nodes"][0].get("params", {}))
    if active_latent:
        lat = active_latent[0]
        if "width" in lat:
            defaults["width"] = lat["width"]
        if "height" in lat:
            defaults["height"] = lat["height"]

    recs["defaults"] = defaults
    recs["wiring"] = wiring

    # Where do the sampling knobs live? This tells the author whether the wiring
    # targets above are complete or whether subgraph_prep is required.
    active_inner_samplers = [s for s in inner_samplers if not s["muted"]]
    if active_samplers:
        recs["sampler_location"] = "top-level"
    elif active_inner_samplers:
        recs["sampler_location"] = "subgraph"
    else:
        recs["sampler_location"] = "none"

    prep_suggestions, prep_warnings = build_subgraph_prep_suggestions(
        subgraph_occurrences, result["guidance_distilled"],
    )
    recs["subgraph_prep_suggestions"] = prep_suggestions

    warnings = list(prep_warnings)
    if recs["sampler_location"] == "subgraph":
        warnings.insert(0,
            "Sampler(s) live INSIDE a subgraph, so steps/sampler/scheduler are "
            "absent from wiring.targets. Expose them via subgraph_prep using "
            "recommendations.subgraph_prep_suggestions, then wire Stimma params to "
            "the new subgraph boundary inputs. Do NOT ship without these knobs."
        )
    recs["warnings"] = warnings

    result["recommendations"] = recs

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze a ComfyUI workflow for Stimma-ification")
    parser.add_argument("workflow", help="Path to the ComfyUI workflow JSON file")
    parser.add_argument("--comfy-url", default="http://localhost:8188",
                        help="ComfyUI server URL (default: http://localhost:8188)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print output")
    args = parser.parse_args()

    result = analyze(args.workflow, args.comfy_url)
    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, default=str))


if __name__ == "__main__":
    main()
