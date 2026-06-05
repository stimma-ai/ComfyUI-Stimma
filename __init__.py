"""ComfyUI-Stimma — expose ComfyUI workflows as Stimma tools.

This plugin:
1. Defines custom ComfyUI nodes (Stimma inputs, outputs, parameters)
2. Hooks an STP WebSocket endpoint into ComfyUI's HTTP server
3. Auto-discovers saved workflows containing Stimma nodes
4. Exposes them as tools that Stimma can execute
"""

import os
import sys
import logging

# Add plugin directory to sys.path so the embedded stimma_tools_protocol package is importable
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

logger = logging.getLogger(__name__)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Hook STP routes into ComfyUI's PromptServer during import.
# Routes must be added before the aiohttp app starts (router freezes on startup).
try:
    from .stp_server.startup import setup_stp_server
    setup_stp_server()
except Exception as e:
    logger.error(f"Failed to set up STP server: {e}", exc_info=True)
