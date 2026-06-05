"""StimmaToolInfo node — metadata-only node that identifies a workflow as a Stimma tool."""


class StimmaToolInfo:
    """Metadata node that marks a workflow as a Stimma tool.

    Place one of these in your workflow to make it discoverable by the
    ComfyUI-Stimma plugin. The slug must be unique across all workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slug": ("STRING", {"default": "my-tool"}),
                "display_name": ("STRING", {"default": "My Tool"}),
                "task_types": ("STRING", {"default": "text-to-image"}),
                "badges": ("STRING", {"default": "", "multiline": True}),
                "description": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Stimma"

    def execute(self, slug, display_name, task_types, badges, description):
        return {}
