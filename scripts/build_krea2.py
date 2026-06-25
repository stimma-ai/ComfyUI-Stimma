#!/usr/bin/env python3
"""Build Stimma-Krea2-Turbo-T2I.json from the stock image_krea2_turbo_t2i template.

Keeps the stock "Text to Image (Krea-2 Turbo)" subgraph and wires Stimma params
into its promoted inputs (matched BY NAME via the instance inputs array).

Subgraph definition input names -> Stimma source:
  value         (prompt)   <- StimmaPromptParam
  width / height           <- StimmaResolutionParam
  seed                     <- StimmaSeedParam
  model_in / clip_in       <- StimmaLoraLoader outputs (added boundary inputs)
  max_length / string_b / unet/clip/vae  -> internal defaults

LoRA handling (done right): instead of the stock fixed darkbrush toggle, a real
StimmaLoraLoader is inserted at the top level and intercepts the model + clip
through the subgraph boundary:
  - new subgraph OUTPUTS model_out (<- UNETLoader.MODEL) / clip_out (<- CLIPLoader.CLIP)
  - new subgraph INPUTS  model_in  (-> KSampler.model) / clip_in (-> CLIPTextEncode.clip)
  - StimmaLoraLoader(model_out, clip_out) -> model_in / clip_in
The stock inner LoraLoaderModelOnly + enable_lora switch are bypassed (orphaned and
stripped at execution). path_filter "krea2/**" exposes loras/krea2/* for selection.
The LLM prompt_enhance path is dropped (it needs an external LLM backend).
"""
import copy
import json
import os
import uuid

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


def defn_add_output(defn, name, type_str, inner_node_id, inner_output_slot):
    """Expose an inner node output as a new subgraph definition output (-20 side)."""
    st = defn.setdefault("state", {})
    lid = st.get("lastLinkId", 0) + 1
    st["lastLinkId"] = lid
    outs = defn.setdefault("outputs", [])
    tslot = len(outs)
    outs.append({"id": str(uuid.uuid4()), "name": name, "type": type_str,
                 "linkIds": [lid], "pos": [1560, 496 + tslot * 20]})
    defn["links"].append({"id": lid, "origin_id": inner_node_id,
                          "origin_slot": inner_output_slot, "target_id": -20,
                          "target_slot": tslot, "type": type_str})
    return tslot


def defn_add_input(defn, name, type_str, inner_node_id, inner_input_name):
    """Add a new subgraph definition input (-10 side) feeding an inner node's input.

    Appended last so it wins the last-link-wins resolution, overriding whatever
    previously fed inner_node_id.inner_input_name.
    """
    st = defn.setdefault("state", {})
    lid = st.get("lastLinkId", 0) + 1
    st["lastLinkId"] = lid
    inps = defn.setdefault("inputs", [])
    oslot = len(inps)
    inps.append({"id": str(uuid.uuid4()), "name": name, "type": type_str,
                 "linkIds": [lid], "pos": [-1230, 3890 + oslot * 20]})
    inner = next(n for n in defn["nodes"] if n["id"] == inner_node_id)
    islot = next(i for i, inp in enumerate(inner["inputs"])
                 if inp.get("name") == inner_input_name)
    inner["inputs"][islot]["link"] = lid
    defn["links"].append({"id": lid, "origin_id": -10, "origin_slot": oslot,
                          "target_id": inner_node_id, "target_slot": islot,
                          "type": type_str})
    return oslot


def main():
    src = json.load(open(SRC))
    sk = harvest(FLUX2, {
        "StimmaToolInfo", "StimmaPromptParam", "StimmaResolutionParam",
        "StimmaSeedParam", "StimmaImageOutput", "StimmaLayoutGroup",
        "StimmaLoraLoader",
    })

    sub = next(n for n in src["nodes"] if n["id"] == 30)
    defn = src["definitions"]["subgraphs"][0]

    # Force the inner prompt-enhance switch OFF. We no longer promote that input,
    # so without this the inner default (true) would route the prompt through the
    # LLM TextGenerate node (broken without an LLM backend). node 24 = enhance bool.
    for n in defn["nodes"]:
        if n["id"] == 24:                       # PrimitiveBoolean (prompt_enhance)
            n["widgets_values"] = [False]
        if n["id"] == 23:                       # PrimitiveBoolean (enable_lora switch)
            n["widgets_values"] = [False]
        if n.get("type") == "LoraLoaderModelOnly":
            # Orphaned once we bypass the model switch, but keep a valid path so
            # discovery doesn't flag a missing model on the dead node.
            n["widgets_values"][0] = "krea2/krea2_darkbrush.safetensors"

    # --- LoRA-through-boundary: expose UNET/CLIP out, feed model/clip back in ---
    # inner node ids: 10 UNETLoader, 11 CLIPLoader, 3 KSampler, 6 CLIPTextEncode
    defn_add_output(defn, "model_out", "MODEL", 10, 0)   # UNETLoader.MODEL
    defn_add_output(defn, "clip_out", "CLIP", 11, 0)     # CLIPLoader.CLIP
    defn_add_input(defn, "model_in", "MODEL", 3, "model")        # -> KSampler.model
    defn_add_input(defn, "clip_in", "CLIP", 6, "clip")          # -> CLIPTextEncode.clip

    # Promoted inputs (names match definition input names); link inputs have no widget.
    sub["inputs"] = [
        {"label": "prompt", "name": "value", "type": "STRING", "widget": {"name": "value"}, "link": None},
        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
        {"name": "height", "type": "INT", "widget": {"name": "height"}, "link": None},
        {"name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
        {"name": "model_in", "type": "MODEL", "link": None},
        {"name": "clip_in", "type": "CLIP", "link": None},
    ]
    sub["outputs"] = [
        {"name": "IMAGE", "type": "IMAGE", "links": []},
        {"name": "model_out", "type": "MODEL", "links": []},
        {"name": "clip_out", "type": "CLIP", "links": []},
    ]
    sub["pos"] = [640, 400]

    tool = clean_node(sk["StimmaToolInfo"], 200, [40, 40], [
        "krea2-turbo-t2i", "Krea 2 Turbo", "text-to-image", "Open Weights",
        "Fast, high-quality text-to-image with Krea 2 Turbo (FP8). Supports "
        "selectable Krea 2 style LoRAs (e.g. Darkbrush).",
        "krea", "krea-2-turbo",
    ])
    prompt = clean_node(sk["StimmaPromptParam"], 201, [40, 260],
                        ["prompt", "", True, 0, "Describe the image to generate."])
    res = clean_node(sk["StimmaResolutionParam"], 202, [40, 540],
                     [1024, 1024, 1024, 2048, 64, RES_PRESETS, 1])
    seed = clean_node(sk["StimmaSeedParam"], 204, [40, 800], ["seed", 0, 2])
    # StimmaLoraLoader: path_filter + ui_order + 10x (lora_name, strength) slots.
    lora = clean_node(sk["StimmaLoraLoader"], 207, [40, 1000],
                      ["krea2/**", 50] + ["None", 1] * 10)
    out = clean_node(sk["StimmaImageOutput"], 208, [1160, 400], ["Krea2_Turbo", ""])
    layout = clean_node(sk["StimmaLayoutGroup"], 209, [40, 1400], [
        "Advanced", "loras\nseed", True, 10,
    ])

    nodes = [tool, prompt, res, seed, lora, out, layout, sub]
    L, lid = [], [0]

    def link(src_node, src_slot, dst_node, dst_slot, typ):
        lid[0] += 1
        L.append([lid[0], src_node, src_slot, dst_node, dst_slot, typ])
        next(n for n in nodes if n["id"] == src_node)["outputs"][src_slot].setdefault("links", []).append(lid[0])
        next(n for n in nodes if n["id"] == dst_node)["inputs"][dst_slot]["link"] = lid[0]

    # sub instance slots: in [0 value,1 width,2 height,3 seed,4 model_in,5 clip_in]
    #                     out [0 IMAGE,1 model_out,2 clip_out]
    # lora slots:         in [0 model,1 clip,...]  out [0 model,1 clip]
    link(201, 0, 30, 0, "STRING")    # prompt  -> value
    link(202, 0, 30, 1, "INT")       # width
    link(202, 1, 30, 2, "INT")       # height
    link(204, 0, 30, 3, "INT")       # seed
    link(30, 1, 207, 0, "MODEL")     # subgraph model_out -> lora.model
    link(30, 2, 207, 1, "CLIP")      # subgraph clip_out  -> lora.clip
    link(207, 0, 30, 4, "MODEL")     # lora.model -> subgraph model_in
    link(207, 1, 30, 5, "CLIP")      # lora.clip  -> subgraph clip_in
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
