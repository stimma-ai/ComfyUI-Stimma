"""H3-owned binding checks for the bundled state-producing workflows."""
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stp_server.discovery import _convert_ui_to_api
ROOT = Path(__file__).resolve().parents[1]


class H3WorkflowTests(unittest.TestCase):
    def test_every_variant_exports_the_exact_decoded_sample_and_vaes(self):
        for path in sorted((ROOT / "workflows").glob("Stimma-MiniMax-H3-*.json")):
            with self.subTest(workflow=path.name):
                api = _convert_ui_to_api(json.loads(path.read_text()), {})
                state_id, state = next((k, n) for k, n in api.items() if n["class_type"] == "StimmaH3GenerationState")
                output = next(n for n in api.values() if n["class_type"] == "StimmaVideoOutput")
                assert output["inputs"]["sidecars"] == [state_id, 0]
                video = api[output["inputs"]["frames"][0]]["inputs"]
                audio = api[output["inputs"]["audio"][0]]["inputs"]
                assert video["samples"] == audio["samples"] == state["inputs"]["samples"]
                assert video["vae"] == state["inputs"]["vae"]
                assert audio["vae"] == state["inputs"]["audio_vae"]
                assert api[state["inputs"]["samples"][0]]["class_type"] == "SamplerCustomAdvanced"
                assert all(not n["class_type"].count("-") == 4 for n in api.values())


if __name__ == "__main__":
    unittest.main()
