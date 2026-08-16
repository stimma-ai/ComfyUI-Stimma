"""Credentials for model downloads: Hugging Face token, Civitai key.

Stored in the plugin's config.yaml under `credentials:`; env / hf-cli login
are honored as fallbacks so people who already logged in don't have to paste.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = _PLUGIN_DIR / "config.yaml"


def _read_yaml() -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        logger.warning("could not parse config.yaml", exc_info=True)
        return {}


def _write_credentials(creds: dict) -> None:
    import yaml
    data = _read_yaml()
    data["credentials"] = {k: v for k, v in creds.items() if v}
    if not data["credentials"]:
        data.pop("credentials", None)
    tmp = CONFIG_PATH.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    os.replace(tmp, CONFIG_PATH)


def _configured() -> dict:
    return (_read_yaml().get("credentials") or {}) if CONFIG_PATH.exists() else {}


def hf_token() -> Optional[str]:
    """Token precedence: config.yaml → env → huggingface-cli login."""
    tok = _configured().get("huggingface_token")
    if tok:
        return str(tok).strip()
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    for path in (Path.home() / ".cache" / "huggingface" / "token",
                 Path(os.environ.get("HF_HOME", "")) / "token" if os.environ.get("HF_HOME") else None):
        if path and path.exists():
            try:
                t = path.read_text().strip()
                if t:
                    return t
            except OSError:
                pass
    return None


def hf_token_source() -> Optional[str]:
    if _configured().get("huggingface_token"):
        return "config"
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return "environment"
    if (Path.home() / ".cache" / "huggingface" / "token").exists():
        return "huggingface-cli"
    return None


def civitai_key() -> Optional[str]:
    k = _configured().get("civitai_api_key")
    if k:
        return str(k).strip()
    return os.environ.get("CIVITAI_API_KEY") or None


def set_hf_token(token: Optional[str]) -> None:
    creds = dict(_configured())
    creds["huggingface_token"] = (token or "").strip() or None
    _write_credentials(creds)


def set_civitai_key(key: Optional[str]) -> None:
    creds = dict(_configured())
    creds["civitai_api_key"] = (key or "").strip() or None
    _write_credentials(creds)


def mask(secret: Optional[str]) -> Optional[str]:
    if not secret:
        return None
    if len(secret) <= 8:
        return "••••"
    return f"{secret[:3]}••••{secret[-3:]}"


def summary() -> dict:
    hf = hf_token()
    cv = civitai_key()
    return {
        "huggingface": {"set": bool(hf), "masked": mask(hf), "source": hf_token_source()},
        "civitai": {"set": bool(cv), "masked": mask(cv),
                    "source": "config" if _configured().get("civitai_api_key") else ("environment" if cv else None)},
    }
