"""StimmaLayoutGroup node — optional UI grouping for parameters."""


class StimmaLayoutGroup:
    """Groups parameters together in the Stimma UI.

    Metadata-only node — no data connections needed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "group_label": ("STRING", {"default": "Settings"}),
                "param_names": ("STRING", {"default": "", "multiline": True}),
                "collapsed": ("BOOLEAN", {"default": False}),
                "ui_order": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "Stimma"

    def execute(self, group_label, param_names, collapsed, ui_order):
        return {}
