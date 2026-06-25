#!/usr/bin/env python3
"""Build Stimma-Krea2-Turbo-T2I.json from the stock image_krea2_turbo_t2i template.

Keeps the stock "Text to Image (Krea-2 Turbo)" subgraph and wires Stimma params
into its promoted inputs (matched BY NAME via the instance inputs array).

Subgraph definition input names -> Stimma source:
  value         (prompt)        <- StimmaPromptParam
  value_1       (prompt_enhance)<- StimmaBoolParam (default False; avoids LLM path)
  width / height                <- StimmaResolutionParam
  seed                          <- StimmaSeedParam
  value_2       (enable_lora?)  <- StimmaBoolParam (default False)
  strength_model(lora_strength) <- StimmaFloatParam
  max_length / lora_name / string_b / unet/clip/vae  -> internal defaults
"""
import copy
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "source_workflows", "image_krea2_turbo_t2i.json")
FLUX2 = os.path.join(HERE, "workflows", "Stimma-Flux2-Dev.json")
OUT = os.path.join(HERE, "workflows", "Stimma-Krea2-Turbo-T2I.json")

RES_PRESETS = (
    "1024x1024\n1152x896\n896x1152\n1216x832\n832x1216\n"
    "1344x768\n768x1344\n1536x1024\n1024x1536\n2048x2048"
)


def harvest(path, types):
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
    for o in n.get("outputs", []) or []:
        o["links"] = []
    for i in n.get("inputs", []) or []:
        if "widget" not in i:
            i["link"] = None
    return n


def make_bool(nid, pos, wv):
    """StimmaBoolParam skeleton (no instance exists in workflows/ to harvest)."""
    return {
        "id": nid, "type": "StimmaBoolParam", "pos": pos, "size": [320, 120],
        "flags": {}, "order": nid, "mode": 0,
        "inputs": [
            {"name": "name", "type": "STRING", "widget": {"name": "name"}, "link": None},
            {"name": "value", "type": "BOOLEAN", "widget": {"name": "value"}, "link": None},
            {"name": "ui_order", "type": "INT", "widget": {"name": "ui_order"}, "link": None},
            {"name": "ui_description", "type": "STRING", "widget": {"name": "ui_description"}, "link": None},
        ],
        "outputs": [{"name": "value", "type": "BOOLEAN", "links": [], "slot_index": 0}],
        "properties": {"Node name for S&R": "StimmaBoolParam"},
        "widgets_values": wv,
    }


def main():
    src = json.load(open(SRC))
    sk = harvest(FLUX2, {
        "StimmaToolInfo", "StimmaPromptParam", "StimmaResolutionParam",
        "StimmaSeedParam", "StimmaFloatParam", "StimmaImageOutput", "StimmaLayoutGroup",
    })

    # Patch inner LoRA path to where we actually placed the file (loras/krea2/)
    sub = next(n for n in src["nodes"] if n["id"] == 30)
    defn = src["definitions"]["subgraphs"][0]
    for n in defn["nodes"]:
        if n.get("type") == "LoraLoaderModelOnly":
            n["widgets_values"][0] = "krea2/krea2_darkbrush.safetensors"

    # Clean promoted-input array (names must match definition input names).
    sub["inputs"] = [
        {"label": "prompt", "name": "value", "type": "STRING", "widget": {"name": "value"}, "link": None},
        {"label": "prompt_enhance", "name": "value_1", "type": "BOOLEAN", "widget": {"name": "value_1"}, "link": None},
        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
        {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
        {"name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
        {"label": "enable_lora", "name": "value_2", "type": "BOOLEAN", "widget": {"name": "value_2"}, "link": None},
        {"label": "lora_strength", "name": "strength_model", "type": "FLOAT", "widget": {"name": "strength_model"}, "link": None},
    ]
    sub["outputs"] = [{"name": "IMAGE", "type": "IMAGE", "links": []}]
    sub["pos"] = [640, 400]

    tool = clean_node(sk["StimmaToolInfo"], 200, [40, 40], [
        "krea2-turbo-t2i", "Krea 2 Turbo", "text-to-image", "Open Weights",
        "Fast, high-quality text-to-image with Krea 2 Turbo (FP8). Optional "
        "Krea Darkbrush style LoRA and LLM prompt enhancement.",
        "krea", "krea-2-turbo",
    ])
    prompt = clean_node(sk["StimmaPromptParam"], 201, [40, 260],
                        ["prompt", "", True, 0, "Describe the image to generate."])
    res = clean_node(sk["StimmaResolutionParam"], 202, [40, 540],
                     [1024, 1024, 1024, 2048, 64, RES_PRESETS, 1])
    seed = clean_node(sk["StimmaSeedParam"], 204, [40, 800], ["seed", 0, 2])
    enhance = make_bool(205, [40, 980], [
        "prompt_enhance", False, 5,
        "Expand your prompt with an LLM before generating (requires an LLM "
        "backend configured in ComfyUI).",
    ])
    enable_lora = make_bool(206, [40, 1120], [
        "enable_lora", False, 6, "Apply the Krea Darkbrush style LoRA.",
    ])
    lora_str = clean_node(sk["StimmaFloatParam"], 207, [40, 1260], [
        "lora_strength", 0.8, 0.0, 1.5, 0.05, "slider", 7,
        "Strength of the Darkbrush style LoRA.",
    ])
    out = clean_node(sk["StimmaImageOutput"], 208, [1160, 400], ["Krea2_Turbo", ""])
    layout = clean_node(sk["StimmaLayoutGroup"], 209, [40, 1400], [
        "Advanced", "prompt_enhance\nenable_lora\nlora_strength\nseed", True, 10,
    ])

    nodes = [tool, prompt, res, seed, enhance, enable_lora, lora_str, out, layout, sub]
    L, lid = [], [0]

    def link(src_node, src_slot, dst_node, dst_slot, typ):
        lid[0] += 1
        L.append([lid[0], src_node, src_slot, dst_node, dst_slot, typ])
        next(n for n in nodes if n["id"] == src_node)["outputs"][src_slot].setdefault("links", []).append(lid[0])
        next(n for n in nodes if n["id"] == dst_node)["inputs"][dst_slot]["link"] = lid[0]

    link(201, 0, 30, 0, "STRING")    # prompt        -> value
    link(205, 0, 30, 1, "BOOLEAN")   # prompt_enhance-> value_1
    link(202, 0, 30, 2, "INT")       # width
    link(202, 1, 30, 3, "INT")       # height
    link(204, 0, 30, 4, "INT")       # seed
    link(206, 0, 30, 5, "BOOLEAN")   # enable_lora   -> value_2
    link(207, 0, 30, 6, "FLOAT")     # lora_strength -> strength_model
    link(30, 0, 208, 0, "IMAGE")     # IMAGE -> output

    wf = {
        "id": src.get("id", "stimma-krea2-turbo"), "revision": 0,
        "last_node_id": 209, "last_link_id": lid[0],
        "nodes": nodes, "links": L, "groups": [],
        "definitions": src.get("definitions", {}), "config": {}, "extra": {},
        "version": src.get("version", 0.4),
    }
    json.dump(wf, open(OUT, "w"), indent=2)
    print("wrote", OUT)
    print("nodes:", [(n["id"], n["type"]) for n in nodes])
    print("links:", L)


if __name__ == "__main__":
    main()
