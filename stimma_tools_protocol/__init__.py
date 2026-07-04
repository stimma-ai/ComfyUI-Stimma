"""
Stimma Tools Protocol (STP) Framework — embedded in ComfyUI-Stimma.
"""

from stimma_tools_protocol.protocol import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcNotification,
    JsonRpcError,
    STPError,
)
from stimma_tools_protocol.provider import Provider, ProviderConfig
from stimma_tools_protocol.tool import (
    STANDARD_TASK_TYPES,
    tool,
    Tool,
    ToolParameter,
    ToolDescriptor,
    Group,
    Param,
)
from stimma_tools_protocol.transport import Transport, StdioTransport, WebSocketTransport
from stimma_tools_protocol.assets import AssetManager, LocalAssetServer, FilesystemAssetManager

__version__ = "0.1.0"

__all__ = [
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcNotification",
    "JsonRpcError",
    "STPError",
    "Provider",
    "ProviderConfig",
    "tool",
    "STANDARD_TASK_TYPES",
    "Tool",
    "ToolParameter",
    "ToolDescriptor",
    "Group",
    "Param",
    "Transport",
    "StdioTransport",
    "WebSocketTransport",
    "AssetManager",
    "LocalAssetServer",
    "FilesystemAssetManager",
]
