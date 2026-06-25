#!/usr/bin/env python3
"""Test a Stimma-ified workflow through the full executor pipeline.

Uses the same _convert_ui_to_api + _resolve_stimma_links code path as the
real executor, so if this test passes the workflow will work in production.

Usage:
    python3 test_workflow.py <workflow.json> [--comfy-url localhost:8188] [--timeout 120]
    python3 test_workflow.py <workflow.json> --test-image /path/to/image.jpg
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# Add plugin root to sys.path so stp_server modules can be imported
_SCRIPT_DIR = Path(__file__).parent
# scripts -> stimmafy -> commands -> .claude -> ComfyUI-Stimma (plugin/repo root)
_PLUGIN_ROOT = _SCRIPT_DIR.parent.parent.parent.parent
sys.path.insert(0, str(_PLUGIN_ROOT))

# Stub stp_server.config before importing anything else (avoids ComfyUI __init__ imports)
import types
_config_mod = types.ModuleType("stp_server.config")
class Config: pass
_config_mod.Config = Config
sys.modules["stp_server.config"] = _config_mod

from stp_server.comfy_client import SingleComfy
from stp_server.discovery import _convert_ui_to_api, _resolve_stimma_links

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _is_api_format(data: dict) -> bool:
    """Check if data is already in ComfyUI API format (flat dict of {class_type, inputs})."""
    return (
        isinstance(data, dict)
        and any(isinstance(v, dict) and "class_type" in v for v in data.values())
        and "nodes" not in data
    )


def _inject_test_defaults(
    api_prompt: dict,
    uploaded_image: str | None = None,
    uploaded_video: str | None = None,
) -> None:
    """Inject test values into Stimma nodes.

    - Seeds: randomized for each run
    - Text prompts: use the workflow default (already in the node)
    - Image inputs: use uploaded_image if provided
    - Video inputs: use uploaded_video if provided
    - All other params: use workflow defaults (already set from widget values)
    """
    for nid, nd in api_prompt.items():
        ct = nd.get("class_type", "")
        inp = nd.get("inputs", {})

        if ct == "StimmaSeedParam":
            inp["value"] = random.randint(0, 2**32 - 1)

        elif ct == "StimmaImageParam" and uploaded_image:
            inp["image"] = uploaded_image

        elif ct == "StimmaVideoParam" and uploaded_video:
            inp["video"] = uploaded_video

        elif ct in ("StimmaLoraLoader", "StimmaPairedLoraLoader"):
            # Clear all LoRA slots so ComfyUI validation passes (no LoRAs selected)
            for i in range(1, 11):
                inp[f"lora_{i}"] = "None"
                inp[f"strength_{i}"] = 1.0
            if ct == "StimmaPairedLoraLoader":
                for i in range(1, 11):
                    inp[f"lora_low_{i}"] = "None"


def _strip_unknown_nodes(api_prompt: dict, object_info: dict) -> dict:
    """Remove nodes unknown to ComfyUI plus their transitive required dependents.

    Mirrors the executor's stripping logic. After _resolve_stimma_links, the
    ComfyUI nodes already have literal values in place of Stimma node refs,
    so stripping Stimma nodes doesn't cascade-remove the real workflow nodes.
    """
    unknown = {nid for nid, nd in api_prompt.items() if nd.get("class_type") not in object_info}
    if not unknown:
        return api_prompt

    unknown_types = sorted({api_prompt[nid].get("class_type", "?") for nid in unknown})
    logger.info(f"Stripping {len(unknown)} unknown node type(s): {unknown_types}")

    to_remove = set(unknown)
    changed = True
    while changed:
        changed = False
        for nid, nd in list(api_prompt.items()):
            if nid in to_remove:
                continue
            for v in nd.get("inputs", {}).values():
                if (isinstance(v, list) and len(v) == 2
                        and isinstance(v[0], str) and v[0] in to_remove):
                    to_remove.add(nid)
                    changed = True
                    break

    for nid in sorted(to_remove):
        ct = api_prompt[nid].get("class_type", "?")
        if nid in unknown:
            logger.info(f"  Stripped: {ct} (#{nid})")
        else:
            logger.info(f"  Stripped dependent: {ct} (#{nid})")
        del api_prompt[nid]

    return api_prompt


async def run_test(
    workflow_path: str,
    comfy_url: str = "localhost:8188",
    timeout: int = 120,
    test_image: str | None = None,
    test_video: str | None = None,
) -> dict:
    """Run a workflow through the full executor pipeline and return results."""
    import aiohttp

    comfy = SingleComfy(comfy_url)

    # Load workflow
    workflow = json.loads(Path(workflow_path).read_text())
    logger.info(f"Testing: {Path(workflow_path).name}")

    # Fetch object_info — needed for proper UI→API conversion
    logger.info("Fetching object_info from ComfyUI...")
    object_info = await comfy.get_object_info()
    logger.info(f"Got {len(object_info)} node types")

    # Convert UI → API (or use directly if already API format)
    if _is_api_format(workflow):
        logger.info("Detected API format — using directly")
        api_prompt = dict(workflow)
    else:
        logger.info("Converting UI → API (subgraph expansion, group node expansion)...")
        api_prompt = _convert_ui_to_api(workflow, object_info)
        if api_prompt is None:
            return {"status": "error", "error": "UI→API conversion returned None"}
        logger.info(f"Converted: {len(api_prompt)} nodes")

    # Upload test image if provided
    uploaded_image = None
    if test_image:
        logger.info(f"Uploading test image: {test_image}")
        uploaded_image = await comfy.upload_image(test_image)
        logger.info(f"Uploaded: {uploaded_image}")
    else:
        has_required_image = any(
            nd.get("class_type") == "StimmaImageParam"
            and nd.get("inputs", {}).get("required", True)
            for nd in api_prompt.values()
        )
        if has_required_image:
            logger.warning(
                "Workflow has required image input(s) but no --test-image provided — "
                "image chain will be stripped"
            )

    # Upload test video if provided (videos are uploaded via the same image endpoint)
    uploaded_video = None
    if test_video:
        logger.info(f"Uploading test video: {test_video}")
        uploaded_video = await comfy.upload_image(test_video)
        logger.info(f"Uploaded: {uploaded_video}")

    # Inject defaults (randomize seeds, set uploaded image/video)
    _inject_test_defaults(api_prompt, uploaded_image=uploaded_image, uploaded_video=uploaded_video)

    # Resolve Stimma links: replace link refs to Stimma nodes with literal values.
    # This is the same step the executor does before stripping unknown nodes, so
    # the real ComfyUI nodes get literal values instead of dead link references.
    _resolve_stimma_links(api_prompt)
    logger.info("Stimma links resolved")

    # Strip Stimma nodes and any other nodes unknown to this ComfyUI instance
    api_prompt = _strip_unknown_nodes(api_prompt, object_info)

    node_types = sorted({nd.get("class_type") for nd in api_prompt.values()})
    logger.info(f"Final prompt: {len(api_prompt)} nodes")
    logger.info(f"Node types: {node_types}")

    if not api_prompt:
        return {"status": "error", "error": "No nodes remaining after stripping unknown types"}

    # Queue prompt to ComfyUI
    logger.info("Queueing prompt...")
    start = time.time()
    queue_result = await comfy.queue_prompt(api_prompt)
    prompt_id = queue_result["prompt_id"]
    logger.info(f"Queued prompt_id: {prompt_id}")

    # Fail fast on node_errors — these cause silent output skips
    node_errors = queue_result.get("node_errors", {})
    if node_errors:
        lines = []
        for nid, errs in node_errors.items():
            ct = api_prompt.get(str(nid), {}).get("class_type", "?")
            for e in errs.get("errors", []):
                lines.append(f"  {ct} (#{nid}): {e.get('message', e)}")
        return {
            "status": "error",
            "error": "node_errors:\n" + "\n".join(lines),
            "elapsed": time.time() - start,
        }

    # Monitor execution via WebSocket
    ws_url = f"ws://{comfy_url}/ws?clientId={comfy.client_id}"
    success = False

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(
                ws_url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as ws:
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
                                    logger.info(f"✓ Execution complete in {elapsed:.1f}s")
                                    success = True
                                    break
                                else:
                                    ct = api_prompt.get(node, {}).get("class_type", "?")
                                    logger.info(f"  Executing: {ct} (#{node})")

                        elif msg_type == "execution_error":
                            err_data = data.get("data", {})
                            if err_data.get("prompt_id") == prompt_id:
                                error_msg = err_data.get("exception_message", "Unknown error")
                                node = err_data.get("node_id", "?")
                                ct = api_prompt.get(str(node), {}).get("class_type", "?")
                                logger.error(f"✗ Error at {ct} (#{node}): {error_msg}")
                                return {
                                    "status": "error",
                                    "error": error_msg,
                                    "elapsed": time.time() - start,
                                }

                        elif msg_type == "execution_interrupted":
                            logger.error("✗ Execution interrupted")
                            return {"status": "interrupted", "elapsed": time.time() - start}

                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.error("WebSocket closed unexpectedly")
                        break

    except asyncio.TimeoutError:
        return {"status": "timeout", "elapsed": timeout}

    if not success:
        return {
            "status": "error",
            "error": "Did not receive completion signal",
            "elapsed": time.time() - start,
        }

    # Check history for outputs
    history = await comfy.get_history(prompt_id)
    outputs = history.get(prompt_id, {}).get("outputs", {})
    output_count = sum(
        len(files)
        for node_outputs in outputs.values()
        for files in node_outputs.values()
        if isinstance(files, list)
    )

    elapsed = time.time() - start
    logger.info(f"Output files: {output_count}")

    if output_count == 0:
        return {
            "status": "error",
            "error": "No output files produced (check node_errors or output node wiring)",
            "elapsed": elapsed,
        }

    return {
        "status": "success",
        "output_count": output_count,
        "outputs": outputs,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test a Stimma workflow through the executor pipeline"
    )
    parser.add_argument("workflow", help="Workflow JSON file to test")
    parser.add_argument("--comfy-url", default="localhost:8188",
                        help="ComfyUI server address (default: localhost:8188)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout in seconds (default: 120)")
    parser.add_argument("--test-image", default=None,
                        help="Test image path for image-input workflows")
    parser.add_argument("--test-video", default=None,
                        help="Test video path for video-input workflows")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    result = asyncio.run(run_test(
        args.workflow,
        comfy_url=args.comfy_url,
        timeout=args.timeout,
        test_image=args.test_image,
        test_video=args.test_video,
    ))

    if args.json:
        print(json.dumps(result, indent=2, default=str))

    if result["status"] == "success":
        logger.info("PASSED")
        sys.exit(0)
    else:
        logger.error(f"FAILED: {result.get('error', result['status'])}")
        sys.exit(1)


if __name__ == "__main__":
    main()
