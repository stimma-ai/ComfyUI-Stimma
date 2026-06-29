"""Workflow executor — injects values, handles LoRAs, captures output."""

import asyncio
import copy
import json
import logging
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from stimma_tools_protocol.provider import ExecutionContext
    from .discovery import DiscoveredWorkflow
    from .provider import StimmaPluginProvider

logger = logging.getLogger(__name__)

# Map Stimma node types to their "data" input field name
_INPUT_DATA_FIELDS = {
    "StimmaPromptParam": "default_text",
    "StimmaImageParam": "image",
    "StimmaMaskParam": "image",
    "StimmaImagesParam": "image",
    "StimmaVideoParam": "video",
    "StimmaVideosParam": "video",
    "StimmaSeedParam": "value",
}

_PARAM_DATA_FIELDS = {
    "StimmaIntParam": "value",
    "StimmaFloatParam": "value",
    "StimmaStringParam": "value",
    "StimmaDropdownParam": "value",
    "StimmaBoolParam": "value",
}


def _raise_preflight_error(workflow: "DiscoveredWorkflow") -> None:
    """Raise a clear, actionable error when a workflow has missing dependencies.

    Called at execution time so the user gets specific instructions about
    what to install/download before the workflow can run.
    """
    missing_nodes = []
    missing_models = []
    for w in workflow.warnings:
        if w.startswith("missing node: "):
            missing_nodes.append(w[14:])
        elif w.startswith("missing model: "):
            missing_models.append(w[15:])

    slug = workflow.tool_info.get("slug", "unknown")
    display_name = workflow.tool_info.get("display_name", slug)
    filename = os.path.basename(workflow.file_path)

    parts = [
        f'Cannot run "{display_name}" — this workflow has missing dependencies.',
        "",
    ]

    if missing_nodes:
        parts.append(f"Missing custom nodes ({len(missing_nodes)}):")
        for node_type in missing_nodes:
            parts.append(f"  - {node_type}")
        parts.append("")
        parts.append(
            "Install these custom nodes in ComfyUI Manager or manually place them "
            "in ComfyUI/custom_nodes/. Then restart ComfyUI."
        )
        parts.append("")

    if missing_models:
        parts.append(f"Missing models ({len(missing_models)}):")
        for model_info in missing_models:
            parts.append(f"  - {model_info}")
        parts.append("")
        parts.append(
            "Download these model files and place them in the appropriate ComfyUI "
            "models/ subfolder (e.g. models/checkpoints/, models/unet/, models/loras/)."
        )
        parts.append("")

    parts.append(
        f"To diagnose: open ComfyUI, load the workflow file ({filename}), "
        f"and check what nodes show as red/missing. ComfyUI Manager can help "
        f"find and install the required custom nodes and models."
    )

    raise RuntimeError("\n".join(parts))


def _summarize_queue_node_errors(
    node_errors: Dict[str, Any],
    prompt: Dict[str, Any],
    max_items: int = 6,
) -> str:
    """Create a readable one-line summary from ComfyUI queue-time node_errors."""
    details: List[str] = []
    for nid, nerr in node_errors.items():
        class_type = prompt.get(str(nid), {}).get("class_type", "?")
        if isinstance(nerr, dict):
            errs = nerr.get("errors", [])
            if errs and isinstance(errs, list):
                first = errs[0]
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("details") or str(first)
                else:
                    msg = str(first)
            else:
                msg = nerr.get("error", str(nerr))
        else:
            msg = str(nerr)
        details.append(f"#{nid} ({class_type}): {msg}")
        if len(details) >= max_items:
            break

    more = len(node_errors) - len(details)
    if more > 0:
        details.append(f"... plus {more} more node error(s)")
    return "; ".join(details)


async def execute_workflow(
    context: "ExecutionContext",
    workflow: "DiscoveredWorkflow",
    parameters: dict,
    provider: "StimmaPluginProvider",
) -> dict:
    """Execute a discovered workflow via the STP protocol.

    Steps:
    1. Deep-copy the API prompt
    2. Inject Stimma node values (media inputs + scalar params) into the prompt
    3. Handle image/video/mask uploads to ComfyUI
    4. Inject LoRA loader nodes if needed
    5. Set up output capture directory
    6. Queue prompt to ComfyUI
    7. Monitor progress via websocket
    8. Capture output and upload to STP asset manager
    """
    start_time = time.time()

    # Step 0: Pre-flight validation — give excellent error messages for missing deps
    if workflow.warnings:
        _raise_preflight_error(workflow)

    # Step 1: Deep-copy prompt and strip nodes unknown to this ComfyUI instance.
    # Unknown nodes that other nodes depend on (e.g. group/component nodes)
    # are also stripped, along with all their transitive dependents.
    prompt = copy.deepcopy(workflow.api_prompt)
    object_info = provider.object_info or {}
    if object_info:
        unknown = {nid for nid, nd in prompt.items()
                   if nd.get("class_type") not in object_info}
        if unknown:
            unknown_types = sorted({prompt[nid].get("class_type", "?") for nid in unknown})

            # Find nodes that transitively depend on unknown nodes
            to_remove = set(unknown)
            changed = True
            while changed:
                changed = False
                for nid, nd in list(prompt.items()):
                    if nid in to_remove:
                        continue
                    for inp_val in nd.get("inputs", {}).values():
                        if isinstance(inp_val, list) and len(inp_val) == 2 and isinstance(inp_val[0], str):
                            if inp_val[0] in to_remove:
                                to_remove.add(nid)
                                changed = True
                                break

            for nid in sorted(to_remove):
                ct = prompt[nid].get("class_type", "?")
                if nid in unknown:
                    logger.info(f"Stripping unknown node '{ct}' (#{nid}) from prompt")
                else:
                    logger.info(f"Stripping dependent node '{ct}' (#{nid}) — depends on missing node")
                del prompt[nid]

            if not prompt:
                raise RuntimeError(
                    f"Workflow cannot execute: all nodes depend on missing types: "
                    f"{unknown_types}. These may be ComfyUI component/group nodes "
                    f"that need to be expanded. Try re-saving the workflow without group nodes."
                )

    # Step 2: Create temp output directory
    output_dir = tempfile.mkdtemp(prefix="stimma_output_")

    try:
        # Acquire a ComfyUI instance for the full job lifecycle. The worker
        # holds the instance until output capture completes, so the next job
        # only lands here once the GPU is actually idle.
        async with provider.comfy_client.acquire() as instance:
            logger.info(f"Acquired ComfyUI instance {instance.addr} for job")

            # Step 3: Inject media/prompt/seed/resolution values (uploads go to the acquired instance)
            unprovided = await _inject_fields(
                prompt, workflow, parameters, context, instance
            )

            # Step 3.5: Strip unprovided optional input chains
            # When optional inputs (e.g. reference image for i2i) aren't provided,
            # remove them and cascade through required dependents so the downstream
            # nodes that expect those inputs are also removed, while nodes with
            # optional references to them survive.
            if unprovided:
                _strip_unprovided_input_chains(prompt, unprovided, object_info)

            # Step 4: Inject parameter values
            _inject_params(prompt, workflow, parameters)

            # Step 4.5: Resolve Stimma node link references to literal values.
            # Stimma nodes are unknown to ComfyUI and will be stripped — this
            # prevents dependent ComfyUI nodes from being stripped too, and
            # fixes STRING→COMBO type mismatches (e.g. sampler_name, scheduler).
            from .discovery import _resolve_stimma_links
            _resolve_stimma_links(prompt)

            # Step 5: Inject checkpoints
            if workflow.checkpoint_nodes:
                _inject_checkpoints(prompt, workflow, parameters)

            # Step 5.5: Inject LoRAs
            if workflow.lora_nodes:
                _inject_loras(prompt, workflow, parameters, provider)

            # Step 6: Inject output directory
            _inject_output_dir(prompt, workflow, output_dir)
            if workflow.output_nodes:
                logger.info(f"Injected output dir into {len(workflow.output_nodes)} output nodes")
            else:
                logger.warning("No Stimma output nodes — will fall back to history capture")

            # Step 7: Connect the monitoring websocket BEFORE queueing.
            # ComfyUI broadcasts execution events live; a prompt that finishes
            # before we connect would have its "complete" signal sent to nobody,
            # leaving the monitor to hang. Connecting first closes that race and
            # also overlaps the (localhost) handshake with prompt validation.
            await context.report_progress(0.1)

            node_types = {nd.get("class_type") for nd in prompt.values()}
            logger.info(f"Prompt has {len(prompt)} nodes, class_types: {sorted(node_types)}")

            ws = await instance.connect_ws()
            try:
                queue_response = await instance.queue_prompt(prompt)
                queue_node_errors = queue_response.get("node_errors", {}) if isinstance(queue_response, dict) else {}
                if queue_node_errors:
                    summary = _summarize_queue_node_errors(queue_node_errors, prompt)
                    logger.error("ComfyUI accepted prompt with node_errors: %s", summary)
                    raise RuntimeError(
                        "ComfyUI prompt has node validation errors (execution may skip outputs): "
                        f"{summary}"
                    )
                prompt_id = queue_response["prompt_id"]
                logger.info(f"Queued prompt {prompt_id} on {instance.addr}")

                # Step 8: Monitor via websocket
                gen_start = time.time()
                await _monitor_execution(ws, prompt_id, context)
                t_gen = time.time() - gen_start
            finally:
                # Tear the monitoring socket down in the background. A graceful
                # ws close handshake blocks until ComfyUI acks it, and ComfyUI's
                # event loop is busy with post-execution work for ~0.5s right
                # after a job finishes — so awaiting the close here would tack
                # that latency onto every generation. We already have the
                # completion signal and the output file, so let it close async.
                _schedule_ws_close(ws)

            await context.report_progress(0.9)

            # Step 9: Capture output
            t_cap0 = time.time()
            expected_output_node_ids = [
                nid for nid, nd in prompt.items()
                if nd.get("class_type") in {"StimmaImageOutput", "StimmaVideoOutput"}
            ]
            result = await _capture_output(
                output_dir,
                prompt,
                context,
                instance,
                prompt_id,
                expected_output_node_ids=expected_output_node_ids,
            )
            t_cap = time.time() - t_cap0

            await context.report_progress(1.0)

            generation_time = time.time() - start_time
            # gen = ComfyUI queue->complete; cap = output read+upload; the rest
            # (inject/queue/ws) is provider overhead and should be ~0.
            overhead = generation_time - t_gen - t_cap
            logger.info(
                f"Workflow {workflow.tool_info['slug']} completed in {generation_time:.2f}s "
                f"(gen {t_gen:.2f}s, capture {t_cap:.2f}s, overhead {overhead:.2f}s)"
            )

            result["generation_time"] = generation_time
            return result

    finally:
        # Cleanup output directory
        shutil.rmtree(output_dir, ignore_errors=True)


async def _inject_fields(
    prompt: Dict[str, Any],
    workflow: "DiscoveredWorkflow",
    input_data: dict,
    context: "ExecutionContext",
    instance,
) -> List[str]:
    """Inject Stimma field values (media/prompt/seed/resolution) into the prompt dict.

    Returns list of node IDs for optional fields that weren't provided.
    """
    unprovided_node_ids = []

    image_values = input_data.get("input_images")
    if image_values is None:
        image_values = []
    elif not isinstance(image_values, list):
        image_values = [image_values]

    video_values = input_data.get("input_videos")
    if video_values is None:
        video_values = []
    elif not isinstance(video_values, list):
        video_values = [video_values]

    logger.warning(
        "Stimma input ingest: keys=%s input_images=%d input_media_ids=%d input_videos=%d input_video_media_ids=%d",
        sorted(list(input_data.keys())),
        len(input_data.get("input_images") or []) if isinstance(input_data.get("input_images"), list) else (1 if input_data.get("input_images") else 0),
        len(input_data.get("input_media_ids") or []) if isinstance(input_data.get("input_media_ids"), list) else (1 if input_data.get("input_media_ids") else 0),
        len(input_data.get("input_videos") or []) if isinstance(input_data.get("input_videos"), list) else (1 if input_data.get("input_videos") else 0),
        len(input_data.get("input_video_media_ids") or []) if isinstance(input_data.get("input_video_media_ids"), list) else (1 if input_data.get("input_video_media_ids") else 0),
    )
    logger.warning(
        "Stimma normalized media: image_values=%d video_values=%d image_sample=%s",
        len(image_values),
        len(video_values),
        image_values[:3],
    )

    image_cursor = 0
    video_cursor = 0

    for node in workflow.field_nodes:
        node_id = node["node_id"]
        class_type = node["class_type"]
        data_field = _INPUT_DATA_FIELDS.get(class_type)

        if not data_field or node_id not in prompt:
            continue

        if class_type == "StimmaPromptParam":
            field_name = node.get("name", "")
            value = input_data.get(field_name)
            if value is not None:
                prompt[node_id]["inputs"][data_field] = value
            continue

        if class_type == "StimmaMaskParam":
            field_name = node.get("name", "")
            value = input_data.get(field_name)
            if value is None or (isinstance(value, list) and not value):
                # Keep existing optional-chain behavior for mask nodes.
                is_required = node.get("inputs", {}).get("required", True)
                if not is_required:
                    unprovided_node_ids.append(node_id)
                continue
            asset_id = value[0] if isinstance(value, list) else value
            uploaded_name = await _download_and_upload_image(asset_id, context, instance)
            prompt[node_id]["inputs"][data_field] = uploaded_name
            continue

        if class_type == "StimmaImageParam":
            max_items = 1
            consumed = image_values[image_cursor:image_cursor + max_items]
            image_cursor += len(consumed)
            logger.warning(
                "Node %s StimmaImageParam consume: requested=%d consumed=%d cursor=%d",
                node_id,
                max_items,
                len(consumed),
                image_cursor,
            )

            if not consumed:
                if image_cursor == 0:
                    # First node, no images at all — truly required
                    raise RuntimeError("Missing required input_images (expected at least 1 image)")
                # Subsequent node with no images left — treat as optional
                unprovided_node_ids.append(node_id)
                continue

            uploaded_name = await _download_and_upload_image(consumed[0], context, instance)
            prompt[node_id]["inputs"][data_field] = uploaded_name
            continue

        if class_type == "StimmaImagesParam":
            min_items = int(node.get("inputs", {}).get("min_images", 1))
            max_items = int(node.get("inputs", {}).get("max_images", 3))
            consumed = image_values[image_cursor:image_cursor + max_items]
            image_cursor += len(consumed)
            logger.warning(
                "Node %s StimmaImagesParam consume: min=%d max=%d consumed=%d cursor=%d values=%s",
                node_id,
                min_items,
                max_items,
                len(consumed),
                image_cursor,
                consumed[:10],
            )

            if not consumed:
                if min_items == 0:
                    unprovided_node_ids.append(node_id)
                    continue
                raise RuntimeError(
                    f"Missing required input_images (expected at least {min_items} images)"
                )
            if len(consumed) < min_items:
                raise RuntimeError(
                    f"Missing required input_images (got {len(consumed)}, expected at least {min_items})"
                )

            uploaded_names = []
            for asset_id in consumed:
                uploaded_names.append(await _download_and_upload_image(asset_id, context, instance))
            prompt[node_id]["inputs"][data_field] = uploaded_names[0]
            prompt[node_id]["inputs"].pop("_stimma_images", None)
            expanded = _expand_stimma_images_reference_chains(prompt, node_id, uploaded_names)
            # Fallback to IMAGE batching for workflows that don't use ReferenceLatent chaining.
            if not expanded:
                prompt[node_id]["inputs"]["_stimma_images"] = json.dumps(uploaded_names)
            logger.warning(
                "Node %s StimmaImagesParam uploaded=%d first=%s expanded_ref_chain=%s",
                node_id,
                len(uploaded_names),
                uploaded_names[0] if uploaded_names else None,
                expanded,
            )
            continue

        if class_type == "StimmaVideoParam":
            max_items = 1
            consumed = video_values[video_cursor:video_cursor + max_items]
            video_cursor += len(consumed)
            logger.warning(
                "Node %s StimmaVideoParam consume: requested=%d consumed=%d cursor=%d",
                node_id,
                max_items,
                len(consumed),
                video_cursor,
            )

            if not consumed:
                if video_cursor == 0:
                    raise RuntimeError("Missing required input_videos (expected at least 1 video)")
                unprovided_node_ids.append(node_id)
                continue

            uploaded_name = await _download_and_upload_video(consumed[0], context, instance)
            prompt[node_id]["inputs"][data_field] = uploaded_name
            continue

        if class_type == "StimmaVideosParam":
            min_items = int(node.get("inputs", {}).get("min_videos", 1))
            max_items = int(node.get("inputs", {}).get("max_videos", 3))
            consumed = video_values[video_cursor:video_cursor + max_items]
            video_cursor += len(consumed)
            logger.warning(
                "Node %s StimmaVideosParam consume: min=%d max=%d consumed=%d cursor=%d",
                node_id,
                min_items,
                max_items,
                len(consumed),
                video_cursor,
            )

            if len(consumed) < min_items:
                if min_items == 0:
                    unprovided_node_ids.append(node_id)
                    continue
                raise RuntimeError(
                    f"Missing required input_videos (expected at least {min_items} videos)"
                )

            uploaded_names = []
            for asset_id in consumed:
                uploaded_names.append(await _download_and_upload_video(asset_id, context, instance))
            prompt[node_id]["inputs"][data_field] = uploaded_names[0]
            if len(uploaded_names) > 1:
                logger.warning(
                    "StimmaVideosParam consumed %d videos but only the first can be injected; "
                    "multi-video chaining must be handled in the workflow graph",
                    len(uploaded_names),
                )
            continue

        if class_type == "StimmaSeedParam":
            field_name = node.get("name", "")
            value = input_data.get(field_name)
            if isinstance(value, int):
                prompt[node_id]["inputs"][data_field] = value
            elif value is not None:
                prompt[node_id]["inputs"][data_field] = int(value)
            else:
                prompt[node_id]["inputs"][data_field] = random.randint(0, 0xFFFFFFFF)

    # Always randomize seeds that weren't explicitly provided
    for node in workflow.field_nodes:
        node_id = node["node_id"]
        class_type = node["class_type"]
        if class_type != "StimmaSeedParam" or node_id not in prompt:
            continue
        field_name = node["name"]
        if input_data.get(field_name) is not None:
            continue  # Explicitly provided, already handled above
        data_field = _INPUT_DATA_FIELDS.get(class_type)
        if data_field:
            prompt[node_id]["inputs"][data_field] = random.randint(0, 0xFFFFFFFF)

    # Inject resolution inputs (width/height come from input_data, not per-node name)
    for node in workflow.field_nodes:
        node_id = node["node_id"]
        if node["class_type"] != "StimmaResolutionParam" or node_id not in prompt:
            continue
        width = input_data.get("width")
        if width is not None:
            prompt[node_id]["inputs"]["width"] = int(width)
        height = input_data.get("height")
        if height is not None:
            prompt[node_id]["inputs"]["height"] = int(height)

    return unprovided_node_ids


def _collect_children(prompt: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build a forward adjacency list from node input links."""
    children: Dict[str, List[str]] = {}
    for nid, nd in prompt.items():
        for inp_val in nd.get("inputs", {}).values():
            if (
                isinstance(inp_val, list)
                and len(inp_val) == 2
                and isinstance(inp_val[0], str)
                and inp_val[0] in prompt
            ):
                children.setdefault(inp_val[0], []).append(nid)
    return children


def _find_path(prompt: Dict[str, Any], start: str, goal: str) -> Optional[List[str]]:
    """Find a directed path from start to goal using input-link edges."""
    if start == goal:
        return [start]

    children = _collect_children(prompt)
    queue: List[List[str]] = [[start]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        cur = path[-1]
        for nxt in children.get(cur, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            new_path = path + [nxt]
            if nxt == goal:
                return new_path
            queue.append(new_path)
    return None


def _next_generated_node_id(prompt: Dict[str, Any], used: set[str]) -> str:
    """Generate a fresh numeric node id for runtime prompt mutation."""
    max_numeric = 0
    for nid in list(prompt.keys()) + list(used):
        if nid.isdigit():
            max_numeric = max(max_numeric, int(nid))
    candidate = str(max_numeric + 1)
    while candidate in prompt or candidate in used:
        candidate = str(int(candidate) + 1)
    used.add(candidate)
    return candidate


def _capture_consumers(prompt: Dict[str, Any], source_node_id: str) -> List[tuple[str, str]]:
    """Return (node_id, input_name) pairs that consume source_node_id output."""
    refs: List[tuple[str, str]] = []
    for nid, nd in prompt.items():
        for inp_name, inp_val in nd.get("inputs", {}).items():
            if (
                isinstance(inp_val, list)
                and len(inp_val) == 2
                and isinstance(inp_val[0], str)
                and inp_val[0] == source_node_id
            ):
                refs.append((nid, inp_name))
    return refs


def _expand_stimma_images_reference_chains(
    prompt: Dict[str, Any],
    stimma_node_id: str,
    uploaded_names: List[str],
) -> bool:
    """Expand multi-image refs into chained ReferenceLatent nodes.

    This preserves historical Klein-style behavior where each extra reference image
    is encoded separately and appended via additional ReferenceLatent nodes, instead
    of relying on a batched IMAGE tensor.
    """
    if len(uploaded_names) <= 1 or stimma_node_id not in prompt:
        return False

    children = _collect_children(prompt)
    downstream = set()
    stack = [stimma_node_id]
    while stack:
        cur = stack.pop()
        for child in children.get(cur, []):
            if child in downstream:
                continue
            downstream.add(child)
            stack.append(child)

    vae_nodes = [
        nid for nid in downstream
        if prompt.get(nid, {}).get("class_type") == "VAEEncode"
    ]
    if not vae_nodes:
        return False

    used_ids: set[str] = set()
    any_expanded = False
    for vae_id in vae_nodes:
        ref_nodes = []
        for nid, nd in prompt.items():
            if nd.get("class_type") != "ReferenceLatent":
                continue
            latent_in = nd.get("inputs", {}).get("latent")
            if (
                isinstance(latent_in, list)
                and len(latent_in) == 2
                and latent_in[0] == vae_id
            ):
                ref_nodes.append(nid)
        if not ref_nodes:
            continue

        path = _find_path(prompt, stimma_node_id, vae_id)
        if not path:
            continue

        external_consumers = {
            ref_id: _capture_consumers(prompt, ref_id)
            for ref_id in ref_nodes
        }
        branch_tail = {ref_id: ref_id for ref_id in ref_nodes}

        for uploaded_name in uploaded_names[1:]:
            clone_map: Dict[str, str] = {}

            # Clone source input node with the extra uploaded image.
            new_source = _next_generated_node_id(prompt, used_ids)
            new_source_node = copy.deepcopy(prompt[stimma_node_id])
            new_source_node["inputs"]["image"] = uploaded_name
            new_source_node["inputs"].pop("_stimma_images", None)
            prompt[new_source] = new_source_node
            clone_map[stimma_node_id] = new_source

            # Clone transform path up to (and including) VAEEncode.
            for orig_id in path[1:]:
                if orig_id not in prompt:
                    clone_map = {}
                    break
                new_id = _next_generated_node_id(prompt, used_ids)
                cloned = copy.deepcopy(prompt[orig_id])
                for inp_name, inp_val in cloned.get("inputs", {}).items():
                    if (
                        isinstance(inp_val, list)
                        and len(inp_val) == 2
                        and isinstance(inp_val[0], str)
                        and inp_val[0] in clone_map
                    ):
                        cloned["inputs"][inp_name] = [clone_map[inp_val[0]], inp_val[1]]
                prompt[new_id] = cloned
                clone_map[orig_id] = new_id

            if vae_id not in clone_map:
                continue
            new_vae = clone_map[vae_id]

            # Append one ReferenceLatent per branch so refs are chained.
            for ref_id in ref_nodes:
                new_ref = _next_generated_node_id(prompt, used_ids)
                ref_clone = copy.deepcopy(prompt[ref_id])
                ref_clone["inputs"]["latent"] = [new_vae, 0]
                ref_clone["inputs"]["conditioning"] = [branch_tail[ref_id], 0]
                prompt[new_ref] = ref_clone
                branch_tail[ref_id] = new_ref

        for ref_id in ref_nodes:
            final_ref = branch_tail[ref_id]
            if final_ref == ref_id:
                continue
            for consumer_id, inp_name in external_consumers[ref_id]:
                if consumer_id not in prompt:
                    continue
                current = prompt[consumer_id]["inputs"].get(inp_name)
                if (
                    isinstance(current, list)
                    and len(current) == 2
                    and current[0] == ref_id
                ):
                    prompt[consumer_id]["inputs"][inp_name] = [final_ref, current[1]]

        any_expanded = True
        logger.warning(
            "Expanded StimmaImagesParam node %s into %d serial reference image branches via VAE node %s",
            stimma_node_id,
            len(uploaded_names),
            vae_id,
        )

    return any_expanded


async def _download_and_upload_image(
    asset_id: str,
    context: "ExecutionContext",
    instance,
) -> str:
    """Download an STP asset and upload it to ComfyUI's input directory."""
    candidates = asset_id if isinstance(asset_id, (list, tuple)) else [asset_id]
    last_err = None
    tried: List[str] = []

    for candidate in candidates:
        asset_ref = str(candidate)
        tried.append(asset_ref)
        image_data = None
        try:
            image_data = await context.assets.download(asset_ref)
        except FileNotFoundError as err:
            last_err = err
            local_path = Path(asset_ref).expanduser()
            if local_path.exists() and local_path.is_file():
                image_data = local_path.read_bytes()
            else:
                continue

        ext = Path(asset_ref).suffix or ".png"
        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"stimma_upload_{context.request_id}_{os.urandom(4).hex()}{ext}",
        )

        try:
            with open(temp_path, "wb") as f:
                f.write(image_data)
            uploaded_name = await instance.upload_image(temp_path)
            return uploaded_name
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if last_err is not None:
        raise FileNotFoundError(f"Asset not found for any candidate: {tried}") from last_err
    raise FileNotFoundError(f"No valid image input candidates: {tried}")


async def _download_and_upload_video(
    asset_id: str,
    context: "ExecutionContext",
    instance,
) -> str:
    """Download an STP video asset and upload it to ComfyUI."""
    candidates = asset_id if isinstance(asset_id, (list, tuple)) else [asset_id]
    last_err = None
    tried: List[str] = []

    for candidate in candidates:
        asset_ref = str(candidate)
        tried.append(asset_ref)
        video_data = None
        try:
            video_data = await context.assets.download(asset_ref)
        except FileNotFoundError as err:
            last_err = err
            local_path = Path(asset_ref).expanduser()
            if local_path.exists() and local_path.is_file():
                video_data = local_path.read_bytes()
            else:
                continue

        ext = Path(asset_ref).suffix or ".mp4"
        temp_path = os.path.join(
            tempfile.gettempdir(),
            f"stimma_upload_{context.request_id}_{os.urandom(4).hex()}{ext}",
        )

        try:
            with open(temp_path, "wb") as f:
                f.write(video_data)
            uploaded_name = await instance.upload_video(temp_path)
            return uploaded_name
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if last_err is not None:
        raise FileNotFoundError(f"Asset not found for any candidate: {tried}") from last_err
    raise FileNotFoundError(f"No valid video input candidates: {tried}")


def _inject_params(
    prompt: Dict[str, Any],
    workflow: "DiscoveredWorkflow",
    parameters: dict,
):
    """Inject parameter values into the prompt dict."""
    for node in workflow.param_nodes:
        node_id = node["node_id"]
        class_type = node["class_type"]

        if node_id not in prompt:
            continue

        field_name = node["name"]
        data_field = _PARAM_DATA_FIELDS.get(class_type)

        if not data_field:
            continue

        value = parameters.get(field_name)
        if value is None:
            continue

        # Type coercion
        if class_type == "StimmaIntParam":
            value = int(value)
        elif class_type == "StimmaFloatParam":
            value = float(value)
        elif class_type == "StimmaBoolParam":
            value = bool(value)

        prompt[node_id]["inputs"][data_field] = value


def _inject_checkpoints(
    prompt: Dict[str, Any],
    workflow: "DiscoveredWorkflow",
    parameters: dict,
):
    """Inject checkpoint selections into StimmaCheckpointLoader nodes."""
    for ckpt_node in workflow.checkpoint_nodes:
        node_id = ckpt_node["node_id"]
        field_name = ckpt_node.get("name", "checkpoint")

        selected = parameters.get(field_name)
        if selected is None:
            continue

        if node_id not in prompt:
            logger.warning(f"  Checkpoint node {node_id} not found in prompt")
            continue

        prompt[node_id]["inputs"]["ckpt_name"] = selected
        logger.info(f"Injected checkpoint {selected!r} into node {node_id}")


def _inject_loras(
    prompt: Dict[str, Any],
    workflow: "DiscoveredWorkflow",
    parameters: dict,
    provider: "StimmaPluginProvider",
):
    """Inject LoRA selections into StimmaLoraLoader/StimmaPairedLoraLoader slots."""
    for lora_node in workflow.lora_nodes:
        node_id = lora_node["node_id"]
        class_type = lora_node.get("class_type", "StimmaLoraLoader")
        field_name = lora_node["name"]

        loras_raw = parameters.get(field_name, [])
        logger.info(f"LoRA node {node_id} ({class_type}): "
                     f"field={field_name!r}, loras_raw={loras_raw!r}")

        if node_id not in prompt:
            logger.warning(f"  LoRA node {node_id} not found in prompt — was it stripped?")
            continue

        if class_type == "StimmaPairedLoraLoader":
            _inject_paired_loras(prompt, node_id, loras_raw, provider)
        else:
            _inject_standard_loras(prompt, node_id, loras_raw)


def _inject_standard_loras(
    prompt: Dict[str, Any],
    node_id: str,
    loras_raw: list,
):
    """Fill StimmaLoraLoader slots with selected LoRAs."""
    for i in range(1, 11):
        prompt[node_id]["inputs"][f"lora_{i}"] = "None"
        prompt[node_id]["inputs"][f"strength_{i}"] = 1.0
    for i, entry in enumerate(loras_raw[:10], 1):
        lora_path = entry.get("path") or entry.get("lora") or entry.get("name", "")
        weight = entry.get("weight", 1.0)
        if lora_path:
            prompt[node_id]["inputs"][f"lora_{i}"] = lora_path
            prompt[node_id]["inputs"][f"strength_{i}"] = weight
            logger.info(f"  Filled slot {i}: {lora_path} @ {weight}")


def _inject_paired_loras(
    prompt: Dict[str, Any],
    node_id: str,
    loras_raw: list,
    provider: "StimmaPluginProvider",
):
    """Fill StimmaPairedLoraLoader slots with paired LoRA high/low variants.

    The client sends {"path": "<display_name>", "weight": 1.0}. We rebuild
    the pairing map to find the actual high/low filenames.
    """
    from .tool_builder import _get_lora_list, _find_lora_pair, _match_lora_filter

    # Build pairing map: display_name → (high, low)
    path_filter = prompt[node_id]["inputs"].get("path_filter", "")
    all_loras = _get_lora_list(provider.object_info)
    if path_filter:
        filtered = [l for l in all_loras if _match_lora_filter(l, path_filter)]
    else:
        filtered = list(all_loras)

    pair_map = {}
    for name in filtered:
        pair = _find_lora_pair(name, all_loras)
        if pair:
            pair_map[pair["display"]] = pair

    # Clear all slots
    for i in range(1, 11):
        prompt[node_id]["inputs"][f"lora_{i}"] = "None"
        prompt[node_id]["inputs"][f"lora_low_{i}"] = "None"
        prompt[node_id]["inputs"][f"strength_{i}"] = 1.0

    # Fill slots from selections
    for i, entry in enumerate(loras_raw[:10], 1):
        display_name = entry.get("path") or entry.get("lora") or entry.get("name", "")
        weight = entry.get("weight", 1.0)
        if not display_name:
            continue

        pair = pair_map.get(display_name)
        if pair:
            prompt[node_id]["inputs"][f"lora_{i}"] = pair["high"]
            prompt[node_id]["inputs"][f"lora_low_{i}"] = pair["low"]
            prompt[node_id]["inputs"][f"strength_{i}"] = weight
            logger.info(f"  Filled paired slot {i}: {pair['high']} / {pair['low']} @ {weight}")
        else:
            # Fallback: use as-is for both (may not match, but log warning)
            logger.warning(f"  No pair found for display_name={display_name!r}, using as-is")
            prompt[node_id]["inputs"][f"lora_{i}"] = display_name
            prompt[node_id]["inputs"][f"lora_low_{i}"] = display_name
            prompt[node_id]["inputs"][f"strength_{i}"] = weight


def _is_input_required(
    class_type: str,
    input_name: str,
    object_info: Dict[str, Any],
) -> bool:
    """Check if a node input is required according to object_info.

    Returns True if the input is in the 'required' section (or unknown).
    """
    node_info = object_info.get(class_type, {})
    input_def = node_info.get("input", {})

    if input_name in input_def.get("optional", {}):
        return False
    # If in required, or if we can't determine, assume required
    return True


# Guide nodes that degrade to identity (pass-through) when an optional guide
# input is absent, instead of being cascade-removed. This lets a workflow keep
# an optional conditioning guide (e.g. an end frame for first-last-frame video)
# wired into the critical path: when the guide media isn't provided, the node is
# bypassed and its main inputs flow straight to its outputs.
#
# {class_type: {guide_input_name: [(output_slot, passthrough_input_name), ...]}}
# When guide_input_name's source is stripped, each consumer of output_slot is
# rewired to the source of passthrough_input_name.
_BYPASS_ON_MISSING_GUIDE: Dict[str, Dict[str, List[tuple]]] = {
    # LTXVAddGuide(positive, negative, vae, latent, image, ...) -> (positive, negative, latent).
    # With no guide image it is identity over (positive, negative, latent).
    "LTXVAddGuide": {"image": [(0, "positive"), (1, "negative"), (2, "latent")]},
}


def _bypass_guide_node(
    prompt: Dict[str, Any], node_id: str, mapping: List[tuple],
) -> bool:
    """Rewire consumers of node_id's outputs to its pass-through input sources, then drop it.

    Returns False (and changes nothing) if a pass-through source is unavailable,
    so the caller can fall back to cascade removal.
    """
    inputs = prompt[node_id].get("inputs", {})
    rewire = {}
    for out_slot, in_name in mapping:
        src = inputs.get(in_name)
        if not (isinstance(src, list) and len(src) == 2):
            return False  # no clean pass-through source — let the caller cascade-remove
        rewire[out_slot] = src
    for other_id, other in prompt.items():
        if other_id == node_id:
            continue
        for k, v in other.get("inputs", {}).items():
            if isinstance(v, list) and len(v) == 2 and v[0] == node_id and v[1] in rewire:
                other["inputs"][k] = rewire[v[1]]
    del prompt[node_id]
    return True


def _strip_unprovided_input_chains(
    prompt: Dict[str, Any],
    unprovided_node_ids: List[str],
    object_info: Dict[str, Any],
) -> None:
    """Remove unprovided optional input nodes and cascade-remove dependents.

    For each node that references a removed node:
    - If the node is a registered guide node and the missing input is its guide
      input → bypass it (pass main inputs through to outputs) instead of removing
    - If the referencing input is required (per object_info) → cascade-remove
    - If the referencing input is optional → just delete that input entry

    Repeats until stable (no more removals).
    """
    removed = set(unprovided_node_ids)

    # Remove the unprovided Stimma input nodes themselves
    for nid in unprovided_node_ids:
        if nid in prompt:
            ct = prompt[nid].get("class_type", "?")
            logger.info(f"Stripping unprovided optional input '{ct}' (#{nid})")
            del prompt[nid]

    # Cascade: remove nodes whose required inputs reference removed nodes
    changed = True
    while changed:
        changed = False
        for nid in list(prompt.keys()):
            nd = prompt[nid]
            ct = nd.get("class_type", "")
            inputs = nd.get("inputs", {})

            for inp_name, inp_val in list(inputs.items()):
                # Check if this input is a link to a removed node
                if not (isinstance(inp_val, list) and len(inp_val) == 2
                        and isinstance(inp_val[0], str)):
                    continue
                if inp_val[0] not in removed:
                    continue

                # This input references a removed node
                if _is_input_required(ct, inp_name, object_info):
                    # Guide node whose optional guide input vanished → bypass
                    # (pass through) rather than cascade-remove the pipeline.
                    bypass_map = _BYPASS_ON_MISSING_GUIDE.get(ct, {})
                    if inp_name in bypass_map and _bypass_guide_node(
                        prompt, nid, bypass_map[inp_name]
                    ):
                        logger.info(
                            f"Bypassing guide node '{ct}' (#{nid}) — guide input "
                            f"'{inp_name}' not provided; passing inputs through"
                        )
                        removed.add(nid)
                        changed = True
                        break
                    # Required input → cascade: remove this node too
                    logger.info(
                        f"Cascade-removing '{ct}' (#{nid}) — "
                        f"required input '{inp_name}' references removed node #{inp_val[0]}"
                    )
                    removed.add(nid)
                    del prompt[nid]
                    changed = True
                    break  # node deleted, move on
                else:
                    # Optional input → just remove the reference
                    logger.info(
                        f"Removing optional input '{inp_name}' from '{ct}' (#{nid}) — "
                        f"references removed node #{inp_val[0]}"
                    )
                    del inputs[inp_name]


def _inject_output_dir(
    prompt: Dict[str, Any],
    workflow: "DiscoveredWorkflow",
    output_dir: str,
):
    """Inject _stimma_output_dir into all output nodes."""
    for node in workflow.output_nodes:
        node_id = node["node_id"]
        if node_id in prompt:
            prompt[node_id]["inputs"]["_stimma_output_dir"] = output_dir


def _schedule_ws_close(ws) -> None:
    """Close a monitoring websocket + its session without blocking the caller.

    The graceful aiohttp close handshake can wait hundreds of ms on ComfyUI's
    post-execution event loop. We don't need a clean close — fire it off as a
    background task so the result path returns immediately.
    """
    async def _close():
        try:
            await ws.close()
        except Exception:
            pass
        session = getattr(ws, "_session", None)
        if session is not None:
            try:
                await session.close()
            except Exception:
                pass

    try:
        asyncio.create_task(_close())
    except RuntimeError:
        # No running loop (shouldn't happen in the provider) — best effort.
        pass


async def _monitor_execution(ws, prompt_id: str, context: "ExecutionContext"):
    """Monitor ComfyUI execution progress via websocket."""
    import aiohttp

    # Throttle progress forwarding. ComfyUI emits progress dozens of times per
    # second; forwarding every one (each an awaited ws send to the client) lets
    # the monitor fall behind, so a backlog of progress messages sits in the
    # receive buffer ahead of the completion signal — delaying when we see the
    # job finish by hundreds of ms after the GPU is actually done. Coalescing to
    # ~10/s keeps the monitor current so completion is detected immediately.
    _last_progress_t = 0.0
    _PROGRESS_MIN_INTERVAL = 0.1

    async for message in ws:
        if message.type == aiohttp.WSMsgType.TEXT:
            data = json.loads(message.data)
            msg_type = data.get("type", "")

            if msg_type == "progress":
                prog_data = data.get("data", {})
                if prog_data.get("prompt_id") == prompt_id:
                    value = prog_data.get("value", 0)
                    max_val = prog_data.get("max", 1)
                    now = time.time()
                    # Always forward the terminal tick; throttle the rest.
                    if max_val > 0 and (
                        value >= max_val or now - _last_progress_t >= _PROGRESS_MIN_INTERVAL
                    ):
                        _last_progress_t = now
                        # Map ComfyUI progress (0-max) to our range (0.1-0.9)
                        progress = 0.1 + (value / max_val) * 0.8
                        await context.report_progress(progress)

            elif msg_type == "execution_success":
                if data.get("data", {}).get("prompt_id") == prompt_id:
                    # ComfyUI signals success here, ~0.5s before the queue-idle
                    # "executing: null" — and the output file is already written
                    # by this point, so we can proceed to capture immediately.
                    logger.info(f"Execution complete for prompt {prompt_id}")
                    return

            elif msg_type == "executing":
                exec_data = data.get("data", {})
                if exec_data.get("prompt_id") == prompt_id:
                    if exec_data.get("node") is None:
                        # Fallback completion signal (queue went idle).
                        logger.info(f"Execution complete for prompt {prompt_id}")
                        return

            elif msg_type == "execution_error":
                err_data = data.get("data", {})
                if err_data.get("prompt_id") == prompt_id:
                    error_msg = err_data.get("exception_message", "Unknown error")
                    node_type = err_data.get("node_type", "")
                    traceback_lines = err_data.get("traceback", [])
                    # Log full traceback for debugging
                    if traceback_lines:
                        logger.error(
                            "ComfyUI traceback for prompt %s:\n%s",
                            prompt_id,
                            "".join(traceback_lines),
                        )
                    label = f"[{node_type}] " if node_type else ""
                    raise RuntimeError(f"ComfyUI execution error: {label}{error_msg}")

            elif msg_type == "execution_interrupted":
                int_data = data.get("data", {})
                if not int_data or int_data.get("prompt_id") == prompt_id:
                    raise RuntimeError(
                        f"ComfyUI execution interrupted for prompt {prompt_id}"
                    )

        elif message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
            raise RuntimeError(
                f"ComfyUI websocket closed unexpectedly while monitoring prompt {prompt_id}"
            )


async def _capture_output(
    output_dir: str,
    prompt: Dict[str, Any],
    context: "ExecutionContext",
    instance,
    prompt_id: str,
    expected_output_node_ids: Optional[List[str]] = None,
) -> dict:
    """Capture output files from the temp directory and upload to STP assets."""
    output_files = sorted(Path(output_dir).glob("stimma_output_*"))

    if not output_files:
        # Fallback: try to get output from ComfyUI history
        logger.info(f"No stimma_output_* files in {output_dir}, falling back to history")
        return await _capture_from_history(
            instance,
            prompt_id,
            context,
            expected_output_node_ids=expected_output_node_ids or [],
        )

    # Upload the first output file. The Stimma output node (StimmaImageOutput)
    # already writes ComfyUI's prompt + workflow into the PNG's text chunks when
    # it saves, so we must NOT decode + re-encode here to "add" metadata — that
    # re-encode cost hundreds of ms (and over a second under event-loop
    # contention) to duplicate metadata the file already has. Just stream the
    # bytes straight through. The read runs in a worker thread so a large file
    # never blocks the shared STP event loop / asset serving.
    output_path = output_files[0]
    ext = output_path.suffix.lower()

    output_bytes = await asyncio.to_thread(output_path.read_bytes)
    asset_id = await context.assets.upload(output_bytes, ext)
    logger.info(f"Uploaded output: {asset_id} ({len(output_bytes)} bytes)")

    return {"asset_id": asset_id}


async def _capture_from_history(
    instance,
    prompt_id: str,
    context: "ExecutionContext",
    expected_output_node_ids: Optional[List[str]] = None,
) -> dict:
    """Fallback: capture output from ComfyUI history (for non-Stimma output nodes)."""
    history = await instance.get_history(prompt_id)
    logger.debug(f"History keys for {prompt_id}: {list(history.keys()) if isinstance(history, dict) else type(history)}")

    if prompt_id not in history:
        raise RuntimeError(f"No output found for prompt {prompt_id}")

    prompt_history = history[prompt_id]
    outputs = prompt_history.get("outputs", {})
    status = prompt_history.get("status", {})
    logger.info(
        f"History output node IDs: {list(outputs.keys())}, "
        f"status: {status}"
    )

    for node_id, output in outputs.items():
        # Check for images
        images = output.get("images", [])
        for img_info in images:
            filename = img_info.get("filename", "")
            if filename:
                # Fetch image from ComfyUI
                params = {"filename": filename, "type": img_info.get("type", "output")}
                subfolder = img_info.get("subfolder", "")
                if subfolder:
                    params["subfolder"] = subfolder

                import aiohttp
                async with aiohttp.ClientSession(auto_decompress=False) as session:
                    async with session.get(
                        f"http://{instance.addr}/view", params=params
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            ext = Path(filename).suffix or ".png"
                            asset_id = await context.assets.upload(data, ext)
                            return {"asset_id": asset_id}

        # Check for videos
        videos = output.get("videos", output.get("gifs", []))
        for vid_info in videos:
            filename = vid_info.get("filename", "")
            if filename:
                subfolder = vid_info.get("subfolder", "")
                params = {"filename": filename, "type": "output"}
                if subfolder:
                    params["subfolder"] = subfolder

                import aiohttp
                async with aiohttp.ClientSession(auto_decompress=False) as session:
                    async with session.get(
                        f"http://{instance.addr}/view", params=params
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.read()
                            ext = Path(filename).suffix or ".mp4"
                            asset_id = await context.assets.upload(data, ext)
                            return {"asset_id": asset_id}

    # Check history status messages for error details before giving up
    status_messages = status.get("messages", [])
    for msg_type, msg_data in status_messages:
        if msg_type == "execution_error" and isinstance(msg_data, dict):
            node_type = msg_data.get("node_type", "")
            error_msg = msg_data.get("exception_message", "")
            traceback_lines = msg_data.get("traceback", [])
            if traceback_lines:
                logger.error(
                    "ComfyUI traceback (from history) for prompt %s:\n%s",
                    prompt_id,
                    "".join(traceback_lines),
                )
            label = f"[{node_type}] " if node_type else ""
            raise RuntimeError(f"ComfyUI execution error: {label}{error_msg}")

    # Generic fallback with whatever diagnostic info we have
    status_str = status.get("status_str", "unknown")
    n_output_nodes = len(outputs)
    expected_output_node_ids = expected_output_node_ids or []
    expected_str = ",".join(expected_output_node_ids) if expected_output_node_ids else "none"
    raise RuntimeError(
        f"Workflow produced no output files (status={status_str}, "
        f"output nodes={n_output_nodes}, expected_stimma_output_nodes={expected_str}). "
        f"Check that your workflow has a StimmaImageOutput or StimmaVideoOutput node."
    )


