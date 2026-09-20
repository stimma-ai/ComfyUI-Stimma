"""Canvas selection for unified generation and editing workflows."""


class StimmaImageModeLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"generation": ("LATENT",), "edit": ("LATENT",)},
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Utils"
    DESCRIPTION = (
        "Use the generation canvas when no image is supplied, or the "
        "reference-aligned edit canvas when an image is supplied."
    )

    def execute(self, generation, edit, image=None):
        return (generation if image is None else edit,)
