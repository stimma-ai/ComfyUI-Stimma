"""Workflow cleanup utilities for placing Stimma nodes in a left-side lane.

This module only mutates UI layout fields (`nodes[*].pos`, top-level `groups`).
It never rewires links or changes execution semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


Rect = Tuple[float, float, float, float]


@dataclass(frozen=True)
class CleanupConfig:
    """Tunable constants for Stimma layout cleanup."""

    lane_margin_left: float = 260.0
    lane_padding: float = 40.0
    lane_top_offset: float = 36.0
    row_gap: float = 24.0
    section_gap: float = 170.0
    collision_padding: float = 20.0
    shift_step: float = 320.0
    max_shifts: int = 40
    group_padding: float = 50.0
    group_top_padding: float = 86.0
    group_color: str = "#3f789e"
    group_font_size: int = 24


DEFAULT_CONFIG = CleanupConfig()

_TALL_INPUT_TYPES = {
    "StimmaImageParam",
    "StimmaImagesParam",
    "StimmaVideoParam",
    "StimmaVideosParam",
    "StimmaMaskParam",
}

_MEDIA_INPUT_TYPES = {
    "StimmaImageParam",
    "StimmaImagesParam",
    "StimmaVideoParam",
    "StimmaVideosParam",
}

_MEDIA_MIN_LAYOUT_HEIGHT = {
    # Image picker nodes expand unpredictably in UI due to preview pane,
    # so use conservative floor heights to prevent vertical overlap.
    "StimmaImageParam": 560.0,
    "StimmaImagesParam": 620.0,
    "StimmaVideoParam": 420.0,
    "StimmaVideosParam": 460.0,
}

_NODE_MIN_LAYOUT_HEIGHT = {
    # Prompt editor can render taller than persisted size due to controls/help text.
    "StimmaPromptParam": 390.0,
}

_PARAM_LAYOUT_EXTRA = 10.0


def _layout_height(node: Dict[str, Any], default: float = 160.0) -> float:
    """Return height used for vertical stacking/group bounds."""
    size = node.get("size") or [320.0, default]
    h = _num(size[1] if len(size) > 1 else default, default)
    if h <= 0:
        h = default
    ntype = str(node.get("type", ""))
    if ntype in _MEDIA_INPUT_TYPES:
        # UI render height is often larger than persisted size for media inputs.
        h = max(h, _MEDIA_MIN_LAYOUT_HEIGHT.get(ntype, h))
    elif ntype in _TALL_INPUT_TYPES:
        h += 28.0
    if ntype in _NODE_MIN_LAYOUT_HEIGHT:
        h = max(h, _NODE_MIN_LAYOUT_HEIGHT[ntype])
    if "Param" in ntype:
        h += _PARAM_LAYOUT_EXTRA
    return h


def is_ui_workflow(data: Any) -> bool:
    return isinstance(data, dict) and isinstance(data.get("nodes"), list)


def is_stimma_node(node: Dict[str, Any]) -> bool:
    ntype = str(node.get("type", ""))
    return ntype.startswith("Stimma")


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def node_rect(node: Dict[str, Any]) -> Rect:
    """Return [x1, y1, x2, y2] with defensive fallbacks."""
    pos = node.get("pos") or [0.0, 0.0]
    size = node.get("size") or [320.0, 160.0]

    x = _num(pos[0] if len(pos) > 0 else 0.0, 0.0)
    y = _num(pos[1] if len(pos) > 1 else 0.0, 0.0)

    width = _num(size[0] if len(size) > 0 else 320.0, 320.0)
    height = _num(size[1] if len(size) > 1 else 160.0, 160.0)
    if width <= 0:
        width = 320.0
    if height <= 0:
        height = 160.0

    return (x, y, x + width, y + height)


def _bbox(rects: Sequence[Rect]) -> Optional[Rect]:
    if not rects:
        return None
    x1 = min(r[0] for r in rects)
    y1 = min(r[1] for r in rects)
    x2 = max(r[2] for r in rects)
    y2 = max(r[3] for r in rects)
    return (x1, y1, x2, y2)


def _intersects(a: Rect, b: Rect, padding: float = 0.0) -> bool:
    return not (
        a[2] + padding <= b[0] - padding
        or a[0] - padding >= b[2] + padding
        or a[3] + padding <= b[1] - padding
        or a[1] - padding >= b[3] + padding
    )


def _bucket_name(node_type: str) -> str:
    if node_type == "StimmaToolInfo":
        return "metadata"
    if node_type == "StimmaLayoutGroup":
        return "layout"
    if node_type.endswith("Output"):
        return "outputs"
    if "Lora" in node_type:
        return "loras"
    if node_type.endswith("Input"):
        return "inputs"
    if "Param" in node_type:
        return "params"
    return "other"


def _bucket_order(name: str) -> int:
    order = {
        "metadata": 0,
        "inputs": 1,
        "params": 2,
        "loras": 3,
        "outputs": 4,
        "layout": 5,
        "other": 6,
    }
    return order.get(name, 99)


def _sort_key(node: Dict[str, Any]) -> Tuple[int, int, float, int]:
    ntype = str(node.get("type", ""))
    bucket = _bucket_name(ntype)
    order = int(node.get("order", 0) or 0)
    y = _num((node.get("pos") or [0, 0])[1] if len(node.get("pos") or []) > 1 else 0.0, 0.0)
    nid = int(node.get("id", 0) or 0)
    return (_bucket_order(bucket), order, y, nid)


def _proposed_positions(stimma_nodes: List[Dict[str, Any]], lane_x: float, lane_y: float, cfg: CleanupConfig) -> Dict[int, List[float]]:
    y = lane_y
    current_bucket = None
    positions: Dict[int, List[float]] = {}

    for node in sorted(stimma_nodes, key=_sort_key):
        ntype = str(node.get("type", ""))
        bucket = _bucket_name(ntype)
        if current_bucket is None:
            current_bucket = bucket
        elif bucket != current_bucket:
            y += cfg.section_gap
            current_bucket = bucket

        nid = int(node.get("id"))
        positions[nid] = [lane_x, y]
        y += _layout_height(node) + cfg.row_gap

    return positions


def _new_rects(stimma_nodes: List[Dict[str, Any]], new_pos: Dict[int, List[float]]) -> List[Rect]:
    rects: List[Rect] = []
    for node in stimma_nodes:
        nid = int(node.get("id"))
        x, y = new_pos[nid]
        size = node.get("size") or [320.0, 160.0]
        w = _num(size[0] if len(size) > 0 else 320.0, 320.0)
        h = _num(size[1] if len(size) > 1 else 160.0, 160.0)
        if w <= 0:
            w = 320.0
        if h <= 0:
            h = 160.0
        rects.append((x, y, x + w, y + h))
    return rects


def _count_collisions(candidate_rects: Sequence[Rect], fixed_rects: Sequence[Rect], padding: float) -> int:
    collisions = 0
    for r in candidate_rects:
        if any(_intersects(r, f, padding=padding) for f in fixed_rects):
            collisions += 1
    return collisions


def _collect_bucket_groups(stimma_nodes: List[Dict[str, Any]], positions: Dict[int, List[float]]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {
        "metadata": [],
        "inputs": [],
        "params": [],
        "loras": [],
        "outputs": [],
        "layout": [],
    }
    for node in stimma_nodes:
        nid = int(node.get("id"))
        if nid not in positions:
            continue
        bucket = _bucket_name(str(node.get("type", "")))
        if bucket in groups:
            groups[bucket].append(nid)
    return groups


def _next_group_id(groups: List[Dict[str, Any]]) -> int:
    existing = [int(g.get("id", 0) or 0) for g in groups if isinstance(g, dict)]
    return (max(existing) + 1) if existing else 1


def refresh_stimma_canvas_groups(
    workflow: Dict[str, Any],
    stimma_nodes: List[Dict[str, Any]],
    positions: Dict[int, List[float]],
    cfg: CleanupConfig = DEFAULT_CONFIG,
) -> Dict[str, int]:
    """Replace auto-managed Stimma groups and return stats."""
    existing_groups = workflow.get("groups")
    if not isinstance(existing_groups, list):
        existing_groups = []

    kept_groups: List[Dict[str, Any]] = []
    removed = 0
    for g in existing_groups:
        flags = g.get("flags") if isinstance(g, dict) else None
        auto = isinstance(flags, dict) and bool(flags.get("stimma_auto_layout"))
        if auto:
            removed += 1
        else:
            kept_groups.append(g)

    id_cursor = _next_group_id(kept_groups)
    added = 0

    title_map = {
        "metadata": "Stimma: Metadata",
        "inputs": "Stimma: Inputs",
        "params": "Stimma: Parameters",
        "loras": "Stimma: LoRAs",
        "outputs": "Stimma: Outputs",
        "layout": "Stimma: Layout",
    }

    nodes_by_id = {int(n.get("id")): n for n in stimma_nodes}
    bucket_groups = _collect_bucket_groups(stimma_nodes, positions)

    for bucket in ["metadata", "inputs", "params", "loras", "outputs", "layout"]:
        ids = bucket_groups.get(bucket) or []
        if not ids:
            continue
        rects: List[Rect] = []
        for nid in ids:
            node = nodes_by_id.get(nid)
            if not node:
                continue
            x, y = positions[nid]
            size = node.get("size") or [320.0, 160.0]
            w = _num(size[0] if len(size) > 0 else 320.0, 320.0)
            h = _layout_height(node)
            if w <= 0:
                w = 320.0
            rects.append((x, y, x + w, y + h))

        box = _bbox(rects)
        if not box:
            continue
        group = {
            "id": id_cursor,
            "title": title_map[bucket],
            "bounding": [
                box[0] - cfg.group_padding,
                box[1] - cfg.group_top_padding,
                (box[2] - box[0]) + (2 * cfg.group_padding),
                (box[3] - box[1]) + cfg.group_top_padding + cfg.group_padding,
            ],
            "color": cfg.group_color,
            "font_size": cfg.group_font_size,
            "flags": {"stimma_auto_layout": True},
        }
        kept_groups.append(group)
        id_cursor += 1
        added += 1

    workflow["groups"] = kept_groups
    return {"groups_added": added, "groups_removed": removed}


def layout_stimma_nodes(workflow: Dict[str, Any], cfg: CleanupConfig = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Place Stimma nodes on left side without overlapping non-Stimma nodes."""
    nodes = workflow.get("nodes", [])
    stimma_nodes = [n for n in nodes if isinstance(n, dict) and is_stimma_node(n)]
    non_stimma_nodes = [n for n in nodes if isinstance(n, dict) and not is_stimma_node(n)]

    stats = {
        "stimma_nodes": len(stimma_nodes),
        "moved_nodes": 0,
        "shifts": 0,
        "collision_warnings": 0,
        "groups_added": 0,
        "groups_removed": 0,
    }

    if not stimma_nodes:
        return stats

    non_rects = [node_rect(n) for n in non_stimma_nodes]
    stimma_rects = [node_rect(n) for n in stimma_nodes]

    non_box = _bbox(non_rects)
    stimma_box = _bbox(stimma_rects)

    lane_y = (non_box[1] if non_box else (stimma_box[1] if stimma_box else -400.0)) + cfg.lane_top_offset

    max_w = 0.0
    for n in stimma_nodes:
        size = n.get("size") or [320.0, 160.0]
        w = _num(size[0] if len(size) > 0 else 320.0, 320.0)
        if w <= 0:
            w = 320.0
        max_w = max(max_w, w)
    lane_width = max_w + (2 * cfg.lane_padding)

    if non_box:
        lane_x = non_box[0] - lane_width - cfg.lane_margin_left
    elif stimma_box:
        lane_x = stimma_box[0]
    else:
        lane_x = -1400.0

    best_positions = _proposed_positions(stimma_nodes, lane_x, lane_y, cfg)
    best_rects = _new_rects(stimma_nodes, best_positions)
    best_collisions = _count_collisions(best_rects, non_rects, cfg.collision_padding)

    if best_collisions > 0:
        for i in range(1, cfg.max_shifts + 1):
            trial_x = lane_x - (cfg.shift_step * i)
            trial_positions = _proposed_positions(stimma_nodes, trial_x, lane_y, cfg)
            trial_rects = _new_rects(stimma_nodes, trial_positions)
            trial_collisions = _count_collisions(trial_rects, non_rects, cfg.collision_padding)
            if trial_collisions < best_collisions:
                best_collisions = trial_collisions
                best_positions = trial_positions
                best_rects = trial_rects
                stats["shifts"] = i
            if trial_collisions == 0:
                break

    for node in stimma_nodes:
        nid = int(node.get("id"))
        new_pos = best_positions[nid]
        old_pos = node.get("pos") or [0.0, 0.0]
        old_x = _num(old_pos[0] if len(old_pos) > 0 else 0.0, 0.0)
        old_y = _num(old_pos[1] if len(old_pos) > 1 else 0.0, 0.0)
        if abs(old_x - new_pos[0]) > 0.001 or abs(old_y - new_pos[1]) > 0.001:
            stats["moved_nodes"] += 1
        node["pos"] = [new_pos[0], new_pos[1]]

    gstats = refresh_stimma_canvas_groups(workflow, stimma_nodes, best_positions, cfg)
    stats.update(gstats)

    if best_collisions > 0:
        stats["collision_warnings"] = best_collisions

    return stats


def cleanup_workflow(workflow: Dict[str, Any], cfg: CleanupConfig = DEFAULT_CONFIG) -> Dict[str, int]:
    if not is_ui_workflow(workflow):
        return {
            "stimma_nodes": 0,
            "moved_nodes": 0,
            "shifts": 0,
            "collision_warnings": 0,
            "groups_added": 0,
            "groups_removed": 0,
            "skipped": 1,
        }
    stats = layout_stimma_nodes(workflow, cfg)
    stats["skipped"] = 0
    return stats
