#!/usr/bin/env python3
"""Build Stimma-Ideogram4-T2I.json from the stock ComfyUI image_ideogram4_t2i template.

Strategy: keep the stock "Text to Image (Ideogram v4)" subgraph (instance node 98)
and its `definitions`, strip the author-side helper nodes (ResolutionSelector, the
caption-template subgraph, PreviewAny, SaveImage, MarkdownNotes), and wire fresh
Stimma param nodes into the subgraph's promoted inputs.

Subgraph definition input slots (matched BY NAME via the instance inputs array):
  0 text       <- StimmaPromptParam
  1 value      (width)  <- StimmaResolutionParam.width
  2 value_1    (height) <- StimmaResolutionParam.height
  3 noise_seed <- StimmaSeedParam            (not surfaced by stock instance; we add it)
  4 unet_name  (cond unet)   - leave default
  5 clip_name                - leave default
  6 vae_name                 - leave default
  7 unet_name_1 (uncond unet)- leave default
  8 choice     (mode)  <- StimmaStringParam dropdown (Quality/Default/Turbo)
Unconnected promoted inputs fall back to the inner node widget defaults.
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "source_workflows", "image_ideogram4_t2i.json")
FLUX2 = os.path.join(HERE, "workflows", "Stimma-Flux2-Dev.json")
STRSRC = os.path.join(HERE, "workflows", "Stimma-Chroma-HD.json")
OUT = os.path.join(HERE, "workflows", "Stimma-Ideogram4-T2I.json")

RES_PRESETS = (
    "1024x1024\n1152x896\n896x1152\n1216x832\n832x1216\n"
    "1344x768\n768x1344\n1536x640\n640x1536"
)


def harvest(path, types):
    """Return one clean skeleton node per requested type from a workflow file."""
    d = json.load(open(path))
    found = {}
    for n in d["nodes"]:
        t = n.get("type")
        if t in types and t not in found:
            found[t] = copy.deepcopy(n)
    missing = set(types) - set(found)
    if missing:
        raise SystemExit(f"missing skeletons {missing} in {path}")
    return found


def clean_node(n, nid, pos, wv):
    n["id"] = nid
    n["pos"] = pos
    n["widgets_values"] = wv
    n["flags"] = {}
    n["order"] = nid
    n["mode"] = 0
    # reset all output links; clear input links (widget inputs keep their shape)
    for o in n.get("outputs", []) or []:
        o["links"] = []
    for i in n.get("inputs", []) or []:
        if "widget" not in i:
            i["link"] = None
    return n


def main():
    src = json.load(open(SRC))
    sk = harvest(
        FLUX2,
        {
            "StimmaToolInfo",
            "StimmaPromptParam",
            "StimmaResolutionParam",
            "StimmaSeedParam",
            "StimmaImageOutput",
            "StimmaLayoutGroup",
        },
    )
    sk.update(harvest(STRSRC, {"StimmaStringParam"}))

    # --- keep only the Ideogram subgraph instance (id 98) ---
    sub = next(n for n in src["nodes"] if n["id"] == 98)
    # Rebuild its promoted-input array with clean, correctly-named slots.
    # dst_slot (array index) -> name is what the executor maps by name.
    sub["inputs"] = [
        {"label": "prompt", "name": "text", "type": "STRING",
         "widget": {"name": "text"}, "link": None},
        {"label": "width", "name": "value", "type": "INT",
         "widget": {"name": "value"}, "link": None},
        {"label": "height", "name": "value_1", "type": "INT",
         "widget": {"name": "value_1"}, "link": None},
        {"label": "seed", "name": "noise_seed", "type": "INT",
         "widget": {"name": "noise_seed"}, "link": None},
        {"label": "mode", "name": "choice", "type": "COMBO",
         "widget": {"name": "choice"}, "link": None},
    ]
    sub["outputs"] = [{"name": "IMAGE", "type": "IMAGE", "links": []}]
    sub["pos"] = [600, 400]

    # --- assemble Stimma nodes ---
    tool = clean_node(sk["StimmaToolInfo"], 200, [40, 40], [
        "ideogram4-t2i", "Ideogram 4.0", "text-to-image", "Open Weights",
        "Generate images with Ideogram 4.0. Accepts plain natural-language "
        "prompts or a structured JSON prompt for precise control over layout, "
        "colors and style.", "ideogram", "ideogram-v4",
    ])
    prompt = clean_node(sk["StimmaPromptParam"], 201, [40, 260], [
        "prompt", "", True, 0,
        "Describe the image. Natural language works; or paste a structured "
        "JSON prompt (high_level_description / style_description / "
        "compositional_deconstruction) for precise control.",
    ])
    res = clean_node(sk["StimmaResolutionParam"], 202, [40, 560],
                     [1024, 1024, 512, 2048, 64, RES_PRESETS, 1])
    speed = clean_node(sk["StimmaStringParam"], 203, [40, 820], [
        "rendering_speed", "Default", "Quality\nDefault\nTurbo", "dropdown", 2,
        "Quality = best detail (48 steps), Default = balanced (20), "
        "Turbo = fast (12).",
    ])
    seed = clean_node(sk["StimmaSeedParam"], 204, [40, 1040], ["seed", 0, 3])
    out = clean_node(sk["StimmaImageOutput"], 205, [1120, 400],
                     ["Ideogram_4.0", ""])
    layout = clean_node(sk["StimmaLayoutGroup"], 206, [40, 1180],
                        ["Advanced", "rendering_speed\nseed", True, 10])

    nodes = [tool, prompt, res, speed, seed, out, layout, sub]

    # --- links: [id, src_node, src_slot, dst_node, dst_slot, type] ---
    L = []
    lid = [0]

    def link(src_node, src_slot, dst_node, dst_slot, typ):
        lid[0] += 1
        L.append([lid[0], src_node, src_slot, dst_node, dst_slot, typ])
        # register on src output
        sn = next(n for n in nodes if n["id"] == src_node)
        sn["outputs"][src_slot].setdefault("links", []).append(lid[0])
        # register on dst input link field
        dn = next(n for n in nodes if n["id"] == dst_node)
        dn["inputs"][dst_slot]["link"] = lid[0]
        return lid[0]

    link(201, 0, 98, 0, "STRING")   # prompt -> text
    link(202, 0, 98, 1, "INT")      # width  -> value
    link(202, 1, 98, 2, "INT")      # height -> value_1
    link(204, 0, 98, 3, "INT")      # seed   -> noise_seed
    link(203, 0, 98, 4, "COMBO")    # speed  -> choice
    link(98, 0, 205, 0, "IMAGE")    # subgraph IMAGE -> output.images

    wf = {
        "id": src.get("id", "stimma-ideogram4"),
        "revision": 0,
        "last_node_id": 206,
        "last_link_id": lid[0],
        "nodes": nodes,
        "links": L,
        "groups": [],
        "definitions": src.get("definitions", {}),
        "config": {},
        "extra": {},
        "version": src.get("version", 0.4),
    }
    json.dump(wf, open(OUT, "w"), indent=2)
    print("wrote", OUT)
    print("nodes:", [(n["id"], n["type"]) for n in nodes])
    print("links:", L)


if __name__ == "__main__":
    main()
