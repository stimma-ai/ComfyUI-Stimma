"""Paths and pinned sources for the optional isolated MOSS audio runtime."""

import os
from pathlib import Path

MOSS_RUNTIME = "moss-soundeffect-v2"

PLUGIN_ROOT = Path(__file__).resolve().parent
MOSS_SOURCE = "934d6826b084c46a0d033402174d5f8ac4ed2519"
MOSS_REPO = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
MOSS_REVISION = "e35df4d82fbe87fcd5d14e5d100e349c0c3c076d"
SA3_REPO = "Comfy-Org/stable-audio-3"
SA3_REVISION = "75aad8836271cb5a5123fb58304fae5d3bb4b0f9"
SA3_FILES = (
    "checkpoints/stable_audio_3_medium.safetensors",
    "checkpoints/stable_audio_3_medium_base.safetensors",
    "text_encoders/t5gemma_b_b_ul2.safetensors",
)
MOSS_FILES = (
    "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/merges.txt", "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/vae_128d_48k.pth",
)


def runtime_python():
    override = os.environ.get("STIMMA_MOSS_PYTHON")
    return Path(override) if override else PLUGIN_ROOT / ".runtimes/moss-soundeffect-v2" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_ready():
    python = runtime_python()
    if not python.is_file():
        return False
    if os.environ.get("STIMMA_MOSS_PYTHON"):
        return True  # Explicit externally managed runtime.
    marker = python.parent.parent / ".stimma-moss-source"
    try:
        return marker.read_text().strip() == MOSS_SOURCE
    except OSError:
        return False


def model_directory(models_root):
    return Path(models_root) / "moss_soundeffect_v2/MOSS-SoundEffect-v2.0"


def missing_moss_files(models_root):
    root = model_directory(models_root)
    return [name for name in MOSS_FILES if not (root / name).is_file()]


def moss_dependencies(models_root):
    """File dependencies use the same names and folders as the download catalog."""
    root = model_directory(models_root)
    return [{"filename": f"{root.name}/{name}", "folder": "moss_soundeffect_v2",
             "installed": (root / name).is_file()} for name in MOSS_FILES]


def runtime_status(runtime_id):
    if runtime_id != MOSS_RUNTIME:
        raise KeyError(runtime_id)
    installed = runtime_ready()
    return {"id": runtime_id, "title": "MOSS SoundEffect runtime", "installed": installed,
            "installable": installed or not bool(os.environ.get("STIMMA_MOSS_PYTHON")),
            "detail": "The configured external runtime is missing." if not installed and os.environ.get("STIMMA_MOSS_PYTHON") else None}
