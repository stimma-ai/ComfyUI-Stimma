"""
Tool decorator and registry for defining STP tools.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List, Dict, Awaitable
from functools import wraps
import inspect

from stimma_tools_protocol.protocol import ToolDescriptor


# --- Layout DSL ---

@dataclass
class Param:
    """Reference to a parameter in a layout group.

    Used to specify which params belong in a group and any layout overrides.
    """
    name: str
    label: Optional[str] = None  # Override param's display label
    full_width: bool = False     # Span all columns in the UI
    visible_when: Optional[tuple] = None  # (param_name, value) - show only when condition met

    def to_dict(self) -> dict:
        """Convert to serializable dict for metadata."""
        d = {"name": self.name}
        if self.label is not None:
            d["label"] = self.label
        if self.full_width:
            d["full_width"] = True
        if self.visible_when is not None:
            d["visible_when"] = {"param": self.visible_when[0], "value": self.visible_when[1]}
        return d


@dataclass
class Group:
    """A group of parameters with a label.

    Used in tool layout to organize params visually.
    """
    label: str  # Title Case - UI applies styling (e.g., ALLCAPS)
    params: List["Param"] = field(default_factory=list)
    collapsed: bool = False  # Start collapsed (user can expand)

    def to_dict(self) -> dict:
        """Convert to serializable dict for metadata."""
        d = {
            "label": self.label,
            "params": [p.to_dict() for p in self.params],
        }
        if self.collapsed:
            d["collapsed"] = True
        return d


# --- Tool Parameter Definition ---

@dataclass
class ToolParameter:
    """Definition of a tool parameter for schema generation."""
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str = ""
    required: bool = False
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    items: Optional[dict] = None  # For array types
    # UI hints that pass through to x-* properties in the schema
    # e.g., {"control": "slider", "group": "sampling", "order": 1, "step": 0.1}
    ui_hints: Optional[Dict[str, Any]] = None

    def to_schema(self) -> dict:
        """Convert to JSON Schema property."""
        schema: dict = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.items is not None:
            schema["items"] = self.items
        # Add UI hints as x-* properties
        if self.ui_hints:
            for key, value in self.ui_hints.items():
                schema[f"x-{key}"] = value
        return schema


@dataclass
class Tool:
    """A registered tool that can be executed."""
    slug: str
    display_name: str
    task_type: str
    function: Callable[..., Awaitable[Any]]
    description: str = ""
    parameters: List[ToolParameter] = field(default_factory=list)
    layout: Optional[List[Group]] = None  # Tool-level layout (grouping, order)
    metadata: dict = field(default_factory=dict)
    task_types: List[str] = field(default_factory=list)
    model_vendor: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self):
        if not self.task_types and self.task_type:
            self.task_types = [self.task_type]

    def to_descriptor(self) -> ToolDescriptor:
        """Convert to a ToolDescriptor for protocol transmission."""
        # Build parameter schema
        param_properties = {}
        param_required = []
        for p in self.parameters:
            param_properties[p.name] = p.to_schema()
            if p.required:
                param_required.append(p.name)

        parameter_schema = {"type": "object", "properties": param_properties}
        if param_required:
            parameter_schema["required"] = param_required

        # Output schema (standard: list of produced assets)
        output_schema = {
            "type": "object",
            "required": ["assets"],
            "properties": {
                "assets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["asset_id", "type"],
                        "properties": {
                            "asset_id": {"type": "string"},
                            "type": {"type": "string", "enum": ["image", "video", "audio", "document"]},
                            "role": {"type": "string"},
                        },
                    },
                },
            },
        }

        # Filter metadata to only include JSON-serializable values
        # (e.g., exclude ToolRequirements which is only used internally)
        serializable_metadata = {
            k: v for k, v in self.metadata.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }

        # Add layout to metadata if present
        if self.layout:
            serializable_metadata["layout"] = [g.to_dict() for g in self.layout]

        return ToolDescriptor(
            id=self.slug,
            name=self.display_name,
            task_type=self.task_type,
            parameter_schema=parameter_schema,
            output_schema=output_schema,
            metadata=serializable_metadata,
            task_types=self.task_types,
            model_vendor=self.model_vendor or None,
            model=self.model or None,
        )


class ToolRegistry:
    """Registry of all available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.slug] = tool

    def get(self, slug: str) -> Optional[Tool]:
        """Get a tool by slug."""
        return self._tools.get(slug)

    def list(self) -> List[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_descriptors(self) -> List[ToolDescriptor]:
        """Get descriptors for all registered tools."""
        return [t.to_descriptor() for t in self._tools.values()]

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


def tool(
    slug: str,
    display_name: str,
    task_type: str,
    description: str = "",
    parameters: Optional[List[ToolParameter]] = None,
    layout: Optional[List[Group]] = None,
    metadata: Optional[dict] = None,
    register: bool = True,
    task_types: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator to register a function as a tool.

    Example:
        @tool(
            slug="flux-t2i",
            display_name="Flux Dev T2I",
            task_type="text-to-image",
            parameters=[
                ToolParameter(name="prompt", type="string", required=True),
                ToolParameter(name="steps", type="integer", default=30),
            ],
        )
        async def flux_t2i(context, parameters):
            prompt = parameters["prompt"]
            steps = parameters.get("steps", 30)
            # ... generate image ...
            return {"asset_id": "result.png"}
    """

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(f"Tool function {func.__name__} must be async")

        tool_obj = Tool(
            slug=slug,
            display_name=display_name,
            task_type=task_type,
            function=func,
            description=description or func.__doc__ or "",
            parameters=parameters or [],
            layout=layout,
            metadata=metadata or {},
            task_types=task_types or [],
        )

        if register:
            _registry.register(tool_obj)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Attach tool metadata to the wrapper
        wrapper._tool = tool_obj
        return wrapper

    return decorator
