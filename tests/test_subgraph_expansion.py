"""Tests for subgraph and group node expansion in _convert_ui_to_api.

Uses the real Stimma-Qwen-Image-2512.json workflow and real object_info
from a running ComfyUI instance.

Run: python tests/test_subgraph_expansion.py
"""
import copy
import json
import os
import sys
import types
import urllib.request

# Stub out dependencies before importing discovery
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
_stp_pkg = types.ModuleType("stp_server")
_stp_pkg.__path__ = [os.path.join(_root, "stp_server")]
_config_mod = types.ModuleType("stp_server.config")
_config_mod.Config = type("Config", (), {})
sys.modules["stp_server"] = _stp_pkg
sys.modules["stp_server.config"] = _config_mod

from stp_server.discovery import _convert_ui_to_api, _extract_stimma_nodes, _resolve_stimma_links

WORKFLOW_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "workflows", "Stimma-Qwen-Image-2512.json",
)
COMFYUI_URL = "http://127.0.0.1:8188"


def _load_workflow():
    with open(WORKFLOW_PATH) as f:
        return json.load(f)


def _fetch_object_info():
    resp = urllib.request.urlopen(f"{COMFYUI_URL}/object_info", timeout=10)
    return json.loads(resp.read())


def run_tests():
    wf = _load_workflow()
    object_info = _fetch_object_info()
    result = _convert_ui_to_api(wf, object_info)
    passed = 0
    failed = 0

    def check(name, condition, msg=""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} — {msg}")
            failed += 1

    # --- Basic conversion ---
    check("conversion_succeeds",
          result is not None and len(result) > 0,
          f"result={result}")

    if result is None:
        print(f"\nResults: {passed} passed, {failed} failed")
        return failed

    # --- No UUID class_types ---
    uuid_nodes = [
        (nid, nd.get("class_type"))
        for nid, nd in result.items()
        if len(nd.get("class_type", "")) == 36 and "-" in nd.get("class_type", "")
    ]
    check("no_uuid_class_types",
          len(uuid_nodes) == 0,
          f"Found UUID types: {uuid_nodes}")

    # --- Active subgraph expanded ---
    inner_86 = [nid for nid in result if nid.startswith("86:")]
    check("active_subgraph_expanded",
          len(inner_86) >= 6,
          f"Expected 6+ inner nodes for subgraph 86, got {len(inner_86)}: {inner_86}")

    # --- Muted subgraph skipped ---
    inner_92 = [nid for nid in result if nid.startswith("92:")]
    check("muted_subgraph_skipped",
          len(inner_92) == 0,
          f"Muted subgraph 92 should not be expanded, found: {inner_92}")

    # --- Muted nodes skipped ---
    check("muted_save_image_skipped",
          "90" not in result,
          "Muted SaveImage (90) should be skipped")

    # --- Stimma nodes present ---
    class_types = {nd["class_type"] for nd in result.values()}
    check("stimma_nodes_present",
          all(t in class_types for t in ["StimmaToolInfo", "StimmaPromptParam", "StimmaImageOutput"]),
          f"class_types: {class_types}")

    # --- Output wired to expanded inner node ---
    output_node = result.get("103")
    if output_node:
        images = output_node["inputs"].get("images")
        check("output_wired_to_inner_node",
              images and isinstance(images, list) and ":" in str(images[0]),
              f"images={images}")
        if images:
            check("output_target_exists",
                  str(images[0]) in result,
                  f"Referenced node {images[0]} not in result")
    else:
        check("output_node_exists", False, "StimmaImageOutput (103) missing")

    # --- KSampler inputs wired to Stimma param nodes ---
    ks = result.get("86:3")
    if ks:
        inp = ks["inputs"]
        check("ksampler_steps_linked",
              inp.get("steps") == ["110", 0],
              f"steps={inp.get('steps')}")
        check("ksampler_cfg_linked",
              inp.get("cfg") == ["111", 0],
              f"cfg={inp.get('cfg')}")
        check("ksampler_sampler_linked",
              inp.get("sampler_name") == ["112", 0],
              f"sampler_name={inp.get('sampler_name')}")
        check("ksampler_scheduler_linked",
              inp.get("scheduler") == ["113", 0],
              f"scheduler={inp.get('scheduler')}")
        check("ksampler_denoise_linked",
              inp.get("denoise") == ["114", 0],
              f"denoise={inp.get('denoise')}")
    else:
        check("ksampler_exists", False, "KSampler 86:3 missing")

    # --- Inner links wired correctly ---
    if ks:
        inp = ks["inputs"]
        check("inner_link_model",
              inp.get("model") == ["86:66", 0],
              f"model={inp.get('model')}")
        check("inner_link_latent",
              inp.get("latent_image") == ["86:58", 0],
              f"latent_image={inp.get('latent_image')}")

    # --- External input wired through ---
    clip_pos = result.get("86:81")
    if clip_pos:
        check("external_input_wired",
              clip_pos["inputs"].get("text") == ["104", 0],
              f"text={clip_pos['inputs'].get('text')}")
    else:
        check("clip_text_encode_exists", False, "CLIPTextEncode 86:81 missing")

    # --- Stimma nodes extractable ---
    stimma = _extract_stimma_nodes(result)
    check("stimma_extractable",
          stimma is not None and stimma["tool_info"]["slug"] == "qwen-image-2512",
          f"stimma={stimma}")
    if stimma:
        check("stimma_has_output",
              len(stimma["output_nodes"]) == 1,
              f"output_nodes={stimma['output_nodes']}")
        check("stimma_has_inputs",
              len(stimma["input_nodes"]) == 4,
              f"Expected 4 inputs (prompt, resolution, negative_prompt, seed), got {len(stimma['input_nodes'])}: "
              f"{[n['name'] for n in stimma['input_nodes']]}")
        check("stimma_has_params",
              len(stimma["param_nodes"]) >= 5,
              f"Expected 5+ params, got {len(stimma['param_nodes'])}: "
              f"{[n['name'] for n in stimma['param_nodes']]}")

    # --- All link targets exist ---
    broken = []
    for nid, nd in result.items():
        for inp_name, inp_val in nd.get("inputs", {}).items():
            if isinstance(inp_val, list) and len(inp_val) == 2:
                if str(inp_val[0]) not in result:
                    broken.append(f"{nid}.{inp_name} -> {inp_val[0]}")
    check("all_link_targets_exist",
          len(broken) == 0,
          f"broken refs: {broken}")

    # --- Resolve Stimma links and check literal values ---
    resolved = copy.deepcopy(result)
    _resolve_stimma_links(resolved)

    ks_resolved = resolved.get("86:3")
    if ks_resolved:
        ri = ks_resolved["inputs"]
        check("resolved_ksampler_steps", ri.get("steps") == 20, f"steps={ri.get('steps')}")
        check("resolved_ksampler_cfg", ri.get("cfg") == 4.0, f"cfg={ri.get('cfg')}")
        check("resolved_ksampler_sampler", ri.get("sampler_name") == "euler",
              f"sampler_name={ri.get('sampler_name')}")
        check("resolved_ksampler_scheduler", ri.get("scheduler") == "simple",
              f"scheduler={ri.get('scheduler')}")
        check("resolved_ksampler_denoise", ri.get("denoise") == 1.0,
              f"denoise={ri.get('denoise')}")

    # --- Validate prompt against ComfyUI (strip Stimma/unknown nodes first) ---
    # Use the resolved prompt so links to Stimma nodes are literal values
    known_types = set(object_info.keys())

    # Rewire pipe-through Stimma nodes (MODEL/CLIP passthrough) before stripping
    # so downstream ComfyUI nodes don't lose their connections
    _PIPE_THROUGH_TYPES = {"StimmaLoraLoader"}
    for nid, nd in list(resolved.items()):
        ct = nd.get("class_type", "")
        if ct not in _PIPE_THROUGH_TYPES or ct in known_types:
            continue
        model_src = nd["inputs"].get("model")
        clip_src = nd["inputs"].get("clip")
        for other_nid, other_nd in resolved.items():
            if other_nid == nid:
                continue
            for inp_name, inp_val in list(other_nd.get("inputs", {}).items()):
                if isinstance(inp_val, list) and len(inp_val) == 2 and inp_val[0] == nid:
                    if inp_val[1] == 0 and model_src:
                        other_nd["inputs"][inp_name] = model_src
                    elif inp_val[1] == 1 and clip_src:
                        other_nd["inputs"][inp_name] = clip_src

    prompt_to_validate = {}
    for nid, nd in resolved.items():
        ct = nd.get("class_type", "")
        if ct in known_types:
            prompt_to_validate[nid] = nd

    # Also strip nodes that reference removed nodes
    to_remove = set()
    changed = True
    while changed:
        changed = False
        for nid, nd in prompt_to_validate.items():
            if nid in to_remove:
                continue
            for inp_val in nd.get("inputs", {}).values():
                if isinstance(inp_val, list) and len(inp_val) == 2 and isinstance(inp_val[0], str):
                    if inp_val[0] not in prompt_to_validate or inp_val[0] in to_remove:
                        to_remove.add(nid)
                        changed = True
                        break
    for nid in to_remove:
        del prompt_to_validate[nid]

    if prompt_to_validate:
        try:
            import json as _json
            payload = _json.dumps({"prompt": prompt_to_validate}).encode()
            req = urllib.request.Request(
                f"{COMFYUI_URL}/prompt",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            resp_data = _json.loads(resp.read())
            has_error = "error" in resp_data or "node_errors" in resp_data and resp_data["node_errors"]
            if has_error:
                check("comfyui_validates_prompt", False, f"ComfyUI errors: {resp_data}")
            else:
                check("comfyui_validates_prompt", True, "")
                # Note: this actually queues it — cancel if we don't want execution
                pid = resp_data.get("prompt_id")
                if pid:
                    # Delete from queue
                    cancel = _json.dumps({"delete": [pid]}).encode()
                    cancel_req = urllib.request.Request(
                        f"{COMFYUI_URL}/queue",
                        data=cancel,
                        headers={"Content-Type": "application/json"},
                    )
                    try:
                        urllib.request.urlopen(cancel_req, timeout=5)
                    except Exception:
                        pass
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            check("comfyui_validates_prompt", False, f"HTTP {e.code}: {body[:500]}")
        except Exception as e:
            check("comfyui_validates_prompt", False, f"Exception: {e}")
    else:
        check("comfyui_validates_prompt", False, "No known nodes to validate")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(run_tests())
