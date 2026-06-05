#!/usr/bin/env python3
"""Cleanup Stimma node layout in ComfyUI UI-format workflows."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import List

from workflow_layout_cleanup import cleanup_workflow


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GLOB = "workflows/*.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup Stimma node canvas layout")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check whether files need changes")
    mode.add_argument("--write", action="store_true", help="Write changes in-place")

    parser.add_argument("--file", action="append", default=[], help="Specific file path (repeatable)")
    parser.add_argument("--glob", dest="glob_pattern", default=DEFAULT_GLOB, help=f"File glob (default: {DEFAULT_GLOB})")
    return parser.parse_args()


def _iter_targets(args: argparse.Namespace) -> List[Path]:
    targets: List[Path] = []
    for path in args.file:
        targets.append((REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path))
    if not targets:
        matches = sorted(glob.glob(str(REPO_ROOT / args.glob_pattern)))
        targets = [Path(m) for m in matches]
    return targets


def _format_stats(stats: dict) -> str:
    return (
        f"stimma={stats.get('stimma_nodes',0)} moved={stats.get('moved_nodes',0)} "
        f"shifts={stats.get('shifts',0)} groups(+{stats.get('groups_added',0)}/-{stats.get('groups_removed',0)}) "
        f"collisions={stats.get('collision_warnings',0)}"
    )


def main() -> int:
    args = _parse_args()
    write = bool(args.write)
    check = bool(args.check) or not write

    targets = _iter_targets(args)
    if not targets:
        print("No workflow files matched.")
        return 0

    changed = 0
    skipped = 0

    for path in targets:
        if not path.exists():
            print(f"SKIP {path}: not found")
            skipped += 1
            continue

        before = path.read_text()
        try:
            workflow = json.loads(before)
        except json.JSONDecodeError as exc:
            print(f"SKIP {path}: invalid json ({exc})")
            skipped += 1
            continue

        stats = cleanup_workflow(workflow)
        after = json.dumps(workflow, indent=2)
        needs_change = before != after

        if stats.get("skipped"):
            print(f"SKIP {path}: not UI format")
            skipped += 1
            continue

        prefix = "CHG" if needs_change else "OK "
        print(f"{prefix} {path}: {_format_stats(stats)}")

        if needs_change:
            changed += 1
            if write:
                path.write_text(after)

    if check and changed:
        print(f"\n{changed} file(s) require cleanup.")
        return 1

    print(f"\nDone. changed={changed} skipped={skipped} write={write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
