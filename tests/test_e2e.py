#!/usr/bin/env python3
"""Schema-driven end-to-end tests for all Stimma workflows.

Discovers workflows in workflows/, inspects their Stimma nodes to generate
test scenarios, and executes against a live ComfyUI instance.

Usage:
    python3 tests/test_e2e.py [--comfy-url HOST:PORT] [--timeout SECS]
                              [--test-image PATH] [--workflow NAME.json] [--list]
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub stp_server.config before importing plugin modules
import types
config_mod = types.ModuleType("stp_server.config")
class Config: pass
config_mod.Config = Config
sys.modules["stp_server.config"] = config_mod

from stp_server.comfy_client import SingleComfy
from stp_server.discovery import _convert_ui_to_api, _resolve_stimma_links
from stp_server.executor import _strip_unprovided_input_chains

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WORKFLOWS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workflows"
)

# Stimma node class types that represent file-upload inputs
_FILE_INPUT_TYPES = {"StimmaImageParam", "StimmaImagesParam", "StimmaVideoParam", "StimmaVideosParam"}

GENERIC_PROMPT = "a beautiful landscape with mountains and a river, photorealistic"


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------

def _scan_file_inputs(api_prompt: dict) -> list[dict]:
    """Find all Stimma file-input nodes in an API prompt.

    Returns list of {node_id, class_type, name, required}.
    """
    results = []
    for nid, nd in api_prompt.items():
        ct = nd.get("class_type", "")
        if ct not in _FILE_INPUT_TYPES:
            continue
        inp = nd.get("inputs", {})
        results.append({
            "node_id": nid,
            "class_type": ct,
            "name": inp.get("name", ct),
            "required": (
                inp.get("min_images", 1) > 0
                if ct == "StimmaImagesParam"
                else inp.get("min_videos", 1) > 0
                if ct == "StimmaVideosParam"
                else True
            ),
        })
    return results


def generate_scenarios(api_prompt: dict) -> list[dict]:
    """Generate test scenarios from the Stimma nodes in an API prompt.

    Returns list of scenario dicts:
        {name, provide_node_ids: list[str], description: str}
    """
    file_inputs = _scan_file_inputs(api_prompt)
    required = [fi for fi in file_inputs if fi["required"]]
    optional = [fi for fi in file_inputs if not fi["required"]]

    scenarios = []

    # "defaults" — provide only required file inputs, use workflow defaults
    req_ids = [fi["node_id"] for fi in required]
    desc_parts = []
    if required:
        desc_parts.append(f"required: {', '.join(fi['name'] or fi['class_type'] for fi in required)}")
    if optional:
        desc_parts.append(f"optional stripped: {', '.join(fi['name'] or fi['class_type'] for fi in optional)}")
    scenarios.append({
        "name": "defaults",
        "provide_node_ids": req_ids,
        "strip_node_ids": [fi["node_id"] for fi in optional],
        "description": "; ".join(desc_parts) or "all defaults, no file inputs",
    })

    # "with-optional-images" — provide ALL file inputs
    if optional:
        all_ids = [fi["node_id"] for fi in file_inputs]
        scenarios.append({
            "name": "with-optional-images",
            "provide_node_ids": all_ids,
            "strip_node_ids": [],
            "description": f"all file inputs provided: {', '.join(fi['name'] or fi['class_type'] for fi in file_inputs)}",
        })

    return scenarios


# ---------------------------------------------------------------------------
# Value injection (generic)
# ---------------------------------------------------------------------------

def inject_values(api_prompt: dict, provide_node_ids: list[str], uploaded_name: str | None):
    """Inject test values into Stimma nodes. Modifies api_prompt in-place."""
    for nid, nd in api_prompt.items():
        ct = nd.get("class_type", "")
        inp = nd.get("inputs", {})

        if ct == "StimmaSeedParam":
            inp["value"] = random.randint(0, 2**64 - 1)

        elif ct == "StimmaPromptParam":
            # Use workflow default if present, otherwise generic
            if not inp.get("default_text", "").strip():
                inp["default_text"] = GENERIC_PROMPT

        elif ct in _FILE_INPUT_TYPES and nid in provide_node_ids:
            if uploaded_name:
                data_field = "video" if ct in {"StimmaVideoParam", "StimmaVideosParam"} else "image"
                inp[data_field] = uploaded_name


# ---------------------------------------------------------------------------
# Unknown-node stripping (same logic as executor Step 1)
# ---------------------------------------------------------------------------

def strip_unknown_nodes(api_prompt: dict, object_info: dict):
    """Remove nodes whose class_type isn't in object_info, plus dependents."""
    unknown = {nid for nid, nd in api_prompt.items()
               if nd.get("class_type") not in object_info}
    if not unknown:
        return

    to_remove = set(unknown)
    changed = True
    while changed:
        changed = False
        for nid, nd in list(api_prompt.items()):
            if nid in to_remove:
                continue
            for v in nd.get("inputs", {}).values():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    if v[0] in to_remove:
                        to_remove.add(nid)
                        changed = True
                        break

    for nid in sorted(to_remove):
        ct = api_prompt[nid].get("class_type", "?")
        logger.info(f"  Stripping: {ct} (#{nid})")
        del api_prompt[nid]


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

async def run_scenario(
    comfy: SingleComfy,
    object_info: dict,
    workflow_name: str,
    api_prompt_template: dict,
    scenario: dict,
    test_image: str | None,
    timeout: int,
) -> bool:
    """Run a single scenario against live ComfyUI. Returns True on success."""
    label = f"{workflow_name} [{scenario['name']}]"
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {label}")
    logger.info(f"  {scenario['description']}")
    logger.info(f"{'='*60}")

    import copy
    api_prompt = copy.deepcopy(api_prompt_template)

    # Upload test image if scenario needs file inputs
    uploaded_name = None
    if scenario["provide_node_ids"] and test_image:
        uploaded_name = await comfy.upload_image(test_image)
        logger.info(f"Uploaded test image: {uploaded_name}")

    # Inject values
    inject_values(api_prompt, scenario["provide_node_ids"], uploaded_name)

    # Strip unprovided optional input chains
    if scenario["strip_node_ids"]:
        _strip_unprovided_input_chains(api_prompt, scenario["strip_node_ids"], object_info)

    # Resolve Stimma links
    _resolve_stimma_links(api_prompt)

    # Strip unknown nodes
    strip_unknown_nodes(api_prompt, object_info)

    node_types = sorted({nd.get("class_type") for nd in api_prompt.values()})
    logger.info(f"Final prompt: {len(api_prompt)} nodes")
    logger.info(f"Node types: {node_types}")

    # Queue
    start = time.time()
    result = await comfy.queue_prompt(api_prompt)
    prompt_id = result["prompt_id"]
    logger.info(f"Queued: {prompt_id}")

    # Monitor via websocket
    import aiohttp
    ws_url = f"ws://{comfy.addr}/ws?clientId={comfy.client_id}"
    success = False

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url, timeout=aiohttp.ClientTimeout(total=timeout)) as ws:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        msg_type = data.get("type", "")

                        if msg_type == "executing":
                            exec_data = data.get("data", {})
                            if exec_data.get("prompt_id") == prompt_id:
                                node = exec_data.get("node")
                                if node is None:
                                    elapsed = time.time() - start
                                    logger.info(f"DONE in {elapsed:.1f}s")
                                    success = True
                                    break
                                else:
                                    ct = api_prompt.get(node, {}).get("class_type", "?")
                                    logger.info(f"  Executing: {ct} (#{node})")

                        elif msg_type == "execution_error":
                            err_data = data.get("data", {})
                            if err_data.get("prompt_id") == prompt_id:
                                error_msg = err_data.get("exception_message", "Unknown")
                                node = err_data.get("node_id", "?")
                                ct = api_prompt.get(str(node), {}).get("class_type", "?")
                                logger.error(f"FAILED at {ct} (#{node}): {error_msg}")
                                return False

                        elif msg_type == "execution_interrupted":
                            logger.error("Execution interrupted")
                            return False

                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.error("WebSocket closed unexpectedly")
                        return False
    except asyncio.TimeoutError:
        logger.error(f"Timeout after {timeout}s")
        return False

    if not success:
        logger.error("Did not receive completion signal")
        return False

    # Validate output exists in history
    history = await comfy.get_history(prompt_id)
    if prompt_id in history:
        outputs = history[prompt_id].get("outputs", {})
        for node_id, output in outputs.items():
            for img in output.get("images", []):
                logger.info(f"  Output image: {img.get('filename', '?')}")
                return True
            for vid in output.get("gifs", []):
                logger.info(f"  Output video: {vid.get('filename', '?')}")
                return True

    logger.info("No output files in history (execution succeeded)")
    return True


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_workflows(object_info: dict) -> list[dict]:
    """Discover all Stimma workflows and generate their scenarios.

    Returns list of {name, path, api_prompt, scenarios}.
    """
    results = []
    for fname in sorted(os.listdir(WORKFLOWS_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(WORKFLOWS_DIR, fname)
        try:
            with open(fpath) as f:
                ui_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {fname}: {e}")
            continue

        # Must be UI format with nodes
        if "nodes" not in ui_data:
            continue

        # Must have a StimmaToolInfo node
        has_tool_info = any(
            n.get("type") == "StimmaToolInfo" for n in ui_data.get("nodes", [])
        )
        if not has_tool_info:
            continue

        # Convert UI → API
        api_prompt = _convert_ui_to_api(ui_data, object_info)
        if api_prompt is None:
            logger.warning(f"Skipping {fname}: _convert_ui_to_api returned None")
            continue

        scenarios = generate_scenarios(api_prompt)
        results.append({
            "name": fname,
            "path": fpath,
            "api_prompt": api_prompt,
            "scenarios": scenarios,
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def create_test_image() -> str | None:
    """Create a simple test image, returns path or None."""
    try:
        from PIL import Image
        img = Image.new("RGB", (512, 512), color=(128, 64, 200))
        path = os.path.join(tempfile.gettempdir(), "stimma_e2e_test.png")
        img.save(path)
        logger.info(f"Created test image: {path}")
        return path
    except ImportError:
        logger.warning("PIL not available — scenarios requiring images will fail")
        return None


async def main():
    parser = argparse.ArgumentParser(description="Stimma workflow e2e tests")
    parser.add_argument("--comfy-url", default="localhost:8188")
    parser.add_argument("--timeout", type=int, default=600, help="Per-scenario timeout in seconds")
    parser.add_argument("--test-image", default=None, help="Path to test image")
    parser.add_argument("--workflow", default=None, help="Run only this workflow filename")
    parser.add_argument("--list", action="store_true", help="List workflows and scenarios, don't execute")
    args = parser.parse_args()

    comfy = SingleComfy(args.comfy_url)

    logger.info("Fetching object_info from ComfyUI...")
    object_info = await comfy.get_object_info()
    logger.info(f"Got {len(object_info)} node types")

    # Discover workflows
    workflows = discover_workflows(object_info)
    if not workflows:
        logger.error("No Stimma workflows found in workflows/")
        sys.exit(1)

    # Filter by --workflow
    if args.workflow:
        workflows = [w for w in workflows if w["name"] == args.workflow]
        if not workflows:
            logger.error(f"Workflow not found: {args.workflow}")
            sys.exit(1)

    # --list mode
    if args.list:
        for wf in workflows:
            print(f"\n{wf['name']}:")
            for sc in wf["scenarios"]:
                print(f"  [{sc['name']}] {sc['description']}")
        sys.exit(0)

    # Create/use test image
    test_image = args.test_image or create_test_image()

    # Run all scenarios
    results = {}
    for wf in workflows:
        for scenario in wf["scenarios"]:
            label = f"{wf['name']} [{scenario['name']}]"

            # Check if this scenario needs a file input but we have no image
            if scenario["provide_node_ids"] and not test_image:
                logger.warning(f"Skipping {label} — no test image available")
                results[label] = None
                continue

            passed = await run_scenario(
                comfy=comfy,
                object_info=object_info,
                workflow_name=wf["name"],
                api_prompt_template=wf["api_prompt"],
                scenario=scenario,
                test_image=test_image,
                timeout=args.timeout,
            )
            results[label] = passed

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS:")
    all_passed = True
    for label, passed in results.items():
        if passed is None:
            print(f"  SKIP: {label}")
        elif passed:
            print(f"  PASS: {label}")
        else:
            print(f"  FAIL: {label}")
            all_passed = False
    print(f"{'='*60}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
