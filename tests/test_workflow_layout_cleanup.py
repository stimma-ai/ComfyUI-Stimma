"""Tests for Stimma workflow canvas cleanup utilities."""

import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from workflow_layout_cleanup import cleanup_workflow, node_rect


def _intersects(a, b, padding=0):
    return not (
        a[2] + padding <= b[0] - padding
        or a[0] - padding >= b[2] + padding
        or a[3] + padding <= b[1] - padding
        or a[1] - padding >= b[3] + padding
    )


class TestWorkflowLayoutCleanup(unittest.TestCase):
    def _sample_workflow(self):
        return {
            "nodes": [
                {
                    "id": 1,
                    "type": "KSampler",
                    "pos": [0, 0],
                    "size": [350, 250],
                    "order": 1,
                },
                {
                    "id": 2,
                    "type": "SaveImage",
                    "pos": [420, 120],
                    "size": [360, 320],
                    "order": 2,
                },
                {
                    "id": 10,
                    "type": "StimmaToolInfo",
                    "pos": [20, 20],
                    "size": [480, 276],
                    "order": 3,
                },
                {
                    "id": 11,
                    "type": "StimmaPromptParam",
                    "pos": [80, 80],
                    "size": [480, 360],
                    "order": 4,
                },
                {
                    "id": 12,
                    "type": "StimmaIntParam",
                    "pos": [120, 120],
                    "size": [384, 355],
                    "order": 5,
                },
                {
                    "id": 13,
                    "type": "StimmaImageOutput",
                    "pos": [180, 180],
                    "size": [360, 300],
                    "order": 6,
                },
                {
                    "id": 14,
                    "type": "StimmaLayoutGroup",
                    "pos": [240, 240],
                    "size": [365, 240],
                    "order": 7,
                },
            ],
            "groups": [
                {
                    "id": 1,
                    "title": "User Group",
                    "bounding": [0, 0, 100, 100],
                    "flags": {},
                },
                {
                    "id": 2,
                    "title": "Old Stimma Group",
                    "bounding": [0, 0, 100, 100],
                    "flags": {"stimma_auto_layout": True},
                },
            ],
        }

    def test_moves_only_stimma_nodes_and_avoids_overlaps(self):
        wf = self._sample_workflow()
        non_stimma_before = {
            n["id"]: list(n["pos"]) for n in wf["nodes"] if not str(n["type"]).startswith("Stimma")
        }

        stats = cleanup_workflow(wf)
        self.assertEqual(stats["skipped"], 0)
        self.assertGreater(stats["moved_nodes"], 0)

        non_stimma_after = {
            n["id"]: list(n["pos"]) for n in wf["nodes"] if not str(n["type"]).startswith("Stimma")
        }
        self.assertEqual(non_stimma_before, non_stimma_after)

        non_stimma_rects = [node_rect(n) for n in wf["nodes"] if not str(n["type"]).startswith("Stimma")]
        stimma_nodes = [n for n in wf["nodes"] if str(n["type"]).startswith("Stimma")]
        stimma_rects = [node_rect(n) for n in stimma_nodes]
        self.assertTrue(all(r[2] < min(x[0] for x in non_stimma_rects) for r in stimma_rects))

        for srect in stimma_rects:
            self.assertFalse(any(_intersects(srect, nrect, padding=20) for nrect in non_stimma_rects))

    def test_group_refresh_and_idempotence(self):
        wf = self._sample_workflow()
        cleanup_workflow(wf)

        groups = wf.get("groups", [])
        user_groups = [g for g in groups if not g.get("flags", {}).get("stimma_auto_layout")]
        stimma_groups = [g for g in groups if g.get("flags", {}).get("stimma_auto_layout")]
        self.assertEqual(len(user_groups), 1)
        self.assertGreaterEqual(len(stimma_groups), 3)

        first = json.dumps(wf, sort_keys=True)
        cleanup_workflow(wf)
        second = json.dumps(wf, sort_keys=True)
        self.assertEqual(first, second)

    def test_non_ui_workflow_is_skipped(self):
        wf = {"1": {"class_type": "KSampler", "inputs": {}}}
        stats = cleanup_workflow(wf)
        self.assertEqual(stats["skipped"], 1)
        self.assertEqual(stats["moved_nodes"], 0)


if __name__ == "__main__":
    unittest.main()
