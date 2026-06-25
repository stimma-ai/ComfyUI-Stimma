#!/usr/bin/env python3
"""Headless render test for a Stimma workflow against the LOCAL ComfyUI.

Runs the plugin's REAL discovery + UI->API conversion + field/param injection,
then queues the resulting prompt on localhost:8188 and waits for the
StimmaImageOutput to write a file. Proves the stimmafied workflow actually
renders with chosen inputs.

Run inside the ComfyUI venv with a ComfyUI instance live on localhost:8188:
  python scripts/test_render.py \
      workflows/Stimma-Ideogram4-T2I.json \
      '{"prompt":"a red fox in snow","width":1024,"height":1024,"seed":42,"rendering_speed":"Turbo"}'
"""
import asyncio
import copy
import json
import os
import sys
import time
import urllib.request

# Plugin root = parent of this scripts/ directory. Keeps the test portable
# instead of hardcoding an install path.
PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PLUGIN)

COMFY = "http://127.0.0.1:8188"


def get_object_info():
    with urllib.request.urlopen(f"{COMFY}/object_info", timeout=30) as r:
        return json.load(r)


def post_prompt(prompt):
    data = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(f"{COMFY}/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def get_history(pid):
    with urllib.request.urlopen(f"{COMFY}/history/{pid}", timeout=30) as r:
        return json.load(r)


class StubCtx:
    async def report_progress(self, *a, **k):
        pass


class StubInstance:
    addr = "127.0.0.1:8188"


async def main():
    wf_path = sys.argv[1]
    params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}

    from stp_server import discovery, executor

    oi = get_object_info()
    workflows, errors = [], []
    discovery._scan_single_file(wf_path, os.path.basename(wf_path), oi, workflows, errors)
    if errors:
        print("SCAN ERRORS:", errors)
    if not workflows:
        print("FAIL: no Stimma workflow discovered (is StimmaToolInfo present?)")
        return 1
    dw = workflows[0]
    print("tool:", dw.tool_info.get("slug"), "| warnings:", dw.warnings)
    print("field params:", [(n["class_type"], n.get("name")) for n in dw.field_nodes])
    print("scalar params:", [(n["class_type"], n.get("name")) for n in dw.param_nodes])

    # Confirm the subgraph expanded into known node types only
    unknown = sorted({nd.get("class_type") for nd in dw.api_prompt.values()
                      if nd.get("class_type") not in oi})
    print("api_prompt nodes:", len(dw.api_prompt), "| unknown types:", unknown)

    prompt = copy.deepcopy(dw.api_prompt)
    outdir = "/tmp/stimma_test_out"
    os.makedirs(outdir, exist_ok=True)
    for f in os.listdir(outdir):
        os.remove(os.path.join(outdir, f))

    await executor._inject_fields(prompt, dw, params, StubCtx(), StubInstance())
    executor._inject_params(prompt, dw, params)
    discovery._resolve_stimma_links(prompt)
    executor._inject_output_dir(prompt, dw, outdir)

    # Sanity: show the key injected inner nodes
    for nid, nd in prompt.items():
        ct = nd.get("class_type")
        if ct in ("CLIPTextEncode", "RandomNoise", "CustomCombo", "EmptyFlux2LatentImage",
                  "PrimitiveInt", "Ideogram4Scheduler"):
            print(f"  {ct} #{nid} inputs:", json.dumps(nd.get("inputs"))[:160])

    resp = post_prompt(prompt)
    if resp.get("node_errors"):
        print("NODE ERRORS:", json.dumps(resp["node_errors"])[:2000])
        return 1
    pid = resp["prompt_id"]
    print("queued", pid, "...waiting")

    for _ in range(600):
        await asyncio.sleep(2)
        h = get_history(pid)
        if pid in h:
            status = h[pid].get("status", {})
            print("status:", status.get("status_str"), status.get("completed"))
            break
    files = sorted(os.listdir(outdir))
    print("OUTPUT FILES:", files, [os.path.getsize(os.path.join(outdir, f)) for f in files])
    if files:
        print("RENDER OK ->", os.path.join(outdir, files[0]))
        return 0
    print("FAIL: no output file written")
    # dump any execution messages
    if pid in h:
        msgs = h[pid].get("status", {}).get("messages", [])
        print("messages:", json.dumps(msgs)[:2000])
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
