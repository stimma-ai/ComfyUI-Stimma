"""
JSON-RPC 2.0 and Stimma Tools Protocol message types.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
from enum import Enum
import json


# JSON-RPC 2.0 Error Codes
class JsonRpcErrorCode(Enum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# STP Application Error Codes
class STPErrorCode(Enum):
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    ASSET_NOT_FOUND = "ASSET_NOT_FOUND"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    MODEL_NOT_LOADED = "MODEL_NOT_LOADED"


@dataclass
class JsonRpcError:
    """JSON-RPC 2.0 error object."""
    code: int
    message: str
    data: Optional[Any] = None

    def to_dict(self) -> dict:
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result

    @classmethod
    def from_code(cls, code: JsonRpcErrorCode, data: Optional[Any] = None) -> "JsonRpcError":
        messages = {
            JsonRpcErrorCode.PARSE_ERROR: "Parse error",
            JsonRpcErrorCode.INVALID_REQUEST: "Invalid Request",
            JsonRpcErrorCode.METHOD_NOT_FOUND: "Method not found",
            JsonRpcErrorCode.INVALID_PARAMS: "Invalid params",
            JsonRpcErrorCode.INTERNAL_ERROR: "Internal error",
        }
        return cls(code=code.value, message=messages[code], data=data)


@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 request message."""
    method: str
    id: Union[str, int]
    params: Optional[dict] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        result = {"jsonrpc": self.jsonrpc, "method": self.method, "id": self.id}
        if self.params is not None:
            result["params"] = self.params
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> "JsonRpcRequest":
        return cls(
            method=data["method"],
            id=data["id"],
            params=data.get("params"),
            jsonrpc=data.get("jsonrpc", "2.0"),
        )


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 response message."""
    id: Union[str, int, None]
    result: Optional[Any] = None
    error: Optional[JsonRpcError] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        result = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.error is not None:
            result["error"] = self.error.to_dict()
        else:
            result["result"] = self.result
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def success(cls, id: Union[str, int], result: Any) -> "JsonRpcResponse":
        return cls(id=id, result=result)

    @classmethod
    def failure(cls, id: Union[str, int, None], error: JsonRpcError) -> "JsonRpcResponse":
        return cls(id=id, error=error)


@dataclass
class JsonRpcNotification:
    """JSON-RPC 2.0 notification (no id, no response expected)."""
    method: str
    params: Optional[dict] = None
    jsonrpc: str = "2.0"

    def to_dict(self) -> dict:
        result = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            result["params"] = self.params
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class STPError(Exception):
    """Exception for STP-specific errors."""
    def __init__(self, code: STPErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(message)

    def to_dict(self) -> dict:
        return {"code": self.code.value, "message": self.message}


# STP Message Types

STP_VERSION = "1.0"


@dataclass
class ProviderRegistration:
    """Provider registration message params."""
    provider_id: str
    provider_name: str
    server: Optional[str] = None  # Software identifier, e.g. "ComfyUI-Stimma/1.2.3"
    max_concurrent: int = 1
    capabilities: dict = field(default_factory=dict)
    asset_endpoint: Optional[str] = None  # Provider's asset endpoint (relative or absolute URL)
    stp_version: str = STP_VERSION

    def to_dict(self) -> dict:
        result = {
            "stp_version": self.stp_version,
            "provider_id": self.provider_id,
            "provider_name": self.provider_name,
            "max_concurrent": self.max_concurrent,
            "capabilities": self.capabilities,
        }
        if self.server:
            result["server"] = self.server
        if self.asset_endpoint:
            result["asset_endpoint"] = self.asset_endpoint
        return result


@dataclass
class RegistrationResponse:
    """Response to provider registration."""
    session_id: str
    stp_version: Optional[str] = None
    host_version: Optional[str] = None
    capabilities: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RegistrationResponse":
        return cls(
            session_id=data["session_id"],
            stp_version=data.get("stp_version"),
            host_version=data.get("host_version"),
            capabilities=data.get("capabilities") or {},
        )


@dataclass
class ToolDescriptor:
    """Descriptor for a tool exposed by the provider."""
    id: str
    name: str
    task_type: str
    parameter_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    task_types: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.task_types and self.task_type:
            self.task_types = [self.task_type]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "task_types": self.task_types,
            "parameter_schema": self.parameter_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata,
        }


@dataclass
class ExecuteRequest:
    """Request to execute a tool."""
    request_id: str
    tool_id: str
    parameters: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ExecuteRequest":
        return cls(
            request_id=data["request_id"],
            tool_id=data["tool_id"],
            parameters=data.get("parameters", {}),
        )


@dataclass
class QueueStatus:
    """Queue status notification params."""
    queued: int
    running: int
    capacity: int

    def to_dict(self) -> dict:
        return {
            "queued": self.queued,
            "running": self.running,
            "capacity": self.capacity,
        }


@dataclass
class ProgressNotification:
    """Progress notification for a running job."""
    request_id: str
    progress: float  # 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "progress": self.progress,
        }


@dataclass
class ExecutionResult:
    """Result of tool execution."""
    request_id: str
    success: bool
    output: Optional[dict] = None
    error: Optional[dict] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "request_id": self.request_id,
            "success": self.success,
        }
        if self.success:
            result["output"] = self.output or {}
            result["metadata"] = self.metadata
        else:
            result["error"] = self.error or {}
        return result


def parse_message(data: str) -> Union[JsonRpcRequest, JsonRpcNotification, JsonRpcResponse]:
    """Parse a JSON-RPC message from a string."""
    try:
        msg = json.loads(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(msg, dict):
        raise ValueError("Message must be a JSON object")

    if "jsonrpc" not in msg or msg["jsonrpc"] != "2.0":
        raise ValueError("Invalid JSON-RPC version")

    # Response (has result or error)
    if "result" in msg or "error" in msg:
        error = None
        if "error" in msg:
            err = msg["error"]
            error = JsonRpcError(code=err["code"], message=err["message"], data=err.get("data"))
        return JsonRpcResponse(
            id=msg.get("id"),
            result=msg.get("result"),
            error=error,
        )

    # Request or Notification
    if "method" not in msg:
        raise ValueError("Missing method field")

    if "id" in msg:
        return JsonRpcRequest.from_dict(msg)
    else:
        return JsonRpcNotification(method=msg["method"], params=msg.get("params"))
