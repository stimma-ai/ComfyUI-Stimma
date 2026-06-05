"""Plugin configuration for ComfyUI-Stimma."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """STP provider identity."""
    id: str = "comfyui"
    name: str = "ComfyUI Workflows"


@dataclass
class ComfyUIConfig:
    """ComfyUI instance addresses for multi-GPU load balancing."""
    addresses: List[str] = field(default_factory=list)


@dataclass
class DiscoveryConfig:
    """Workflow discovery settings."""
    extra_workflow_dirs: List[str] = field(default_factory=list)
    watch_interval: float = 2.0  # seconds; 0 to disable


@dataclass
class Config:
    """Root configuration for ComfyUI-Stimma plugin."""
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    comfyui: ComfyUIConfig = field(default_factory=ComfyUIConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to config file. If None, looks for config.yaml
                     in the plugin directory.

    Returns:
        Parsed Config object
    """
    if config_path is None:
        plugin_dir = Path(__file__).parent.parent
        config_path = str(plugin_dir / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        logger.info(f"No config file at {config_path}, using defaults")
        return Config()

    logger.info(f"Loading config from {config_path}")

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, using default config")
        return Config()

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    provider_data = data.get("provider", {})
    provider = ProviderConfig(
        id=provider_data.get("id", "comfyui"),
        name=provider_data.get("name", "ComfyUI Workflows"),
    )

    comfyui_data = data.get("comfyui", {})
    comfyui = ComfyUIConfig(
        addresses=comfyui_data.get("addresses", []),
    )

    discovery_data = data.get("discovery", {})
    discovery = DiscoveryConfig(
        extra_workflow_dirs=discovery_data.get("extra_workflow_dirs", []),
        watch_interval=discovery_data.get("watch_interval", 2.0),
    )

    config = Config(
        provider=provider,
        comfyui=comfyui,
        discovery=discovery,
    )

    logger.info(
        f"Config loaded: "
        f"addresses={config.comfyui.addresses or 'auto-detect'}"
    )
    return config
