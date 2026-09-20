"""Deterministic noise seeds for image editing."""

import hashlib
import struct


class StimmaImageSeed:
    """Keep edit noise independent of the noise used to create its reference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "seed": ("INT", {"forceInput": True, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Utils"
    DESCRIPTION = (
        "Derive edit noise from the requested seed and reference pixels. "
        "The same seed and image remain reproducible, while changing the image "
        "changes the noise. This avoids artifacts from reusing generation noise "
        "when editing a generated image or applying successive edits."
    )

    def execute(self, seed, image):
        # Canonical CPU float32 bytes make this independent of device, strides,
        # tensor precision, and host byte order. No global random state changes.
        pixels = image.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
        digest = hashlib.blake2b(digest_size=8, person=b"stimma-edit-v1")
        digest.update(struct.pack("<Q", seed))
        digest.update(struct.pack("<I", pixels.ndim))
        digest.update(struct.pack("<" + "Q" * pixels.ndim, *pixels.shape))
        digest.update(memoryview(pixels).cast("B"))
        return (int.from_bytes(digest.digest(), "little"),)
