"""FLUX.3 Klein reference preparation, matching Black Forest Labs' reference pipeline.

A single-image edit keeps the image's own size (up to 4 MP including padding),
reflect-pads it to the 16 px latent grid and crops the padding off the result, so
the output lines up with the input pixel for pixel. ComfyUI's generic path
instead resamples to a megapixel budget and center-crops to the grid, which
shifts scale and framing. A multi-reference edit fits each image under 1 MP and
center-crops it to the grid; the output follows the first prepared image.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

GRID = 16
SINGLE_MAX_PIXELS = 4 * 1024 * 1024  # first reference, padded canvas included
MULTI_MAX_PIXELS = 1024 * 1024  # each reference of a multi-reference edit
MIN_SIDE, MAX_SIDE = 64, 4096
MAX_REFERENCES = 10


def _ceil_grid(v: int) -> int:
    return -(-v // GRID) * GRID


def _lanczos(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize a [1, H, W, C] image the way the reference pipeline does (PIL Lanczos on 8-bit)."""
    from PIL import Image

    pixels = (image[0, ..., :3].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    resized = Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(resized)).float().div(255)[None].to(image.device)


def single_canvas(width: int, height: int):
    """(fitted size, canvas size) for a single reference: canvas is the fit rounded up to the grid,
    at most 4 MP, sides 64-4096."""
    scale = min(1.0, math.sqrt(SINGLE_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, _ceil_grid(v))) for v in fit)
        if canvas[0] * canvas[1] <= SINGLE_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_single(image: torch.Tensor):
    """Fit and reflect-pad one reference to its canvas, centred.

    Returns the prepared image and the (left, top, right, bottom) box of the
    fitted image inside it, or None when no padding was needed.
    """
    image = image[:1, ..., :3]
    height, width = int(image.shape[1]), int(image.shape[2])
    (fit_w, fit_h), (canvas_w, canvas_h) = single_canvas(width, height)
    if (fit_w, fit_h) != (width, height):
        image = _lanczos(image, fit_w, fit_h)
    if (canvas_w, canvas_h) == (fit_w, fit_h):
        return image, None
    left, top = (canvas_w - fit_w) // 2, (canvas_h - fit_h) // 2
    pad = (left, canvas_w - fit_w - left, top, canvas_h - fit_h - top)
    # Reflect needs each pad shorter than the side; fall back to edge replication.
    mode = "reflect" if min(fit_w, fit_h) > GRID else "replicate"
    padded = F.pad(image.movedim(-1, 1), pad, mode=mode).movedim(1, -1)
    return padded, (left, top, left + fit_w, top + fit_h)


def prepare_multi(image: torch.Tensor) -> torch.Tensor:
    """Fit one reference of a multi-reference edit under 1 MP (sides floored) and centre-crop it to the grid."""
    image = image[:1, ..., :3]
    height, width = int(image.shape[1]), int(image.shape[2])
    if width * height > MULTI_MAX_PIXELS:
        scale = math.sqrt(MULTI_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = _lanczos(image, width, height)
    w, h = width // GRID * GRID, height // GRID * GRID
    left, top = (width - w) // 2, (height - h) // 2
    return image[:, top:top + h, left:left + w, :]


class StimmaFlux3References:
    """Encode FLUX.3 Klein edit references and size the edit canvas from the first."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {"negative": ("CONDITIONING",)}
        optional.update({f"image_{i}": ("IMAGE",) for i in range(1, MAX_REFERENCES + 1)})
        return {"required": {"positive": ("CONDITIONING",), "vae": ("VAE",)}, "optional": optional}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STIMMA_CROP")
    RETURN_NAMES = ("positive", "negative", "edit_latent", "crop")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Utils"
    DESCRIPTION = (
        "FLUX.3 Klein reference preparation from Black Forest Labs' pipeline. One image keeps "
        "its size (up to 4 MP) and is padded to the latent grid, so Crop To Reference returns "
        "an output aligned with the input. Several images are each fitted under 1 MP."
    )

    def execute(self, positive, vae, negative=None, **images):
        import node_helpers

        refs = [images[f"image_{i}"] for i in range(1, MAX_REFERENCES + 1) if images.get(f"image_{i}") is not None]
        crop = None
        if len(refs) == 1:
            prepared, crop = prepare_single(refs[0])
            prepared = [prepared]
        else:
            prepared = [prepare_multi(r) for r in refs]
        latents = [vae.encode(p) for p in prepared]
        if latents:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": latents}, append=True)
            if negative is not None:
                negative = node_helpers.conditioning_set_values(negative, {"reference_latents": latents}, append=True)
            edit_latent = {"samples": latents[0] * 0.0}
        else:
            from nodes import EmptyLatentImage
            edit_latent = EmptyLatentImage().generate(1024, 1024)[0]
        return (positive, negative if negative is not None else positive, edit_latent, crop)


class StimmaCropToReference:
    """Remove the padding added by StimmaFlux3References."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "optional": {"crop": ("STIMMA_CROP",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Utils"

    def execute(self, image, crop=None):
        if crop is None:
            return (image,)
        left, top, right, bottom = crop
        return (image[:, top:bottom, left:right, :],)


def preserve_unchanged(source: torch.Tensor, edited: torch.Tensor, threshold: float = 0.06,
                       grow: int = 16, feather: int = 8, speck: int = 4,
                       max_changed: float = 0.6) -> torch.Tensor:
    """Put the source's own pixels back wherever the edit left the image alone.

    Every output pixel passes through the autoencoder and the sampler, so even
    untouched regions come back slightly re-rendered. Pixels whose locally
    averaged difference stays under ``threshold`` count as unchanged; specks
    narrower than about ``2 * speck`` px are dropped, and the changed region is grown by ``grow`` px and feathered by ``feather`` px so
    edits keep their soft knock-on effects. When more than ``max_changed`` of
    the frame changed (a restyle or relight), the edit is returned untouched.
    """
    if source.shape[1:3] != edited.shape[1:3]:
        return edited
    source = source[..., :3].to(edited.device, edited.dtype)
    diff = (source - edited[..., :3]).abs().amax(-1, keepdim=True).movedim(-1, 1)
    diff = F.avg_pool2d(diff, 9, stride=1, padding=4, count_include_pad=False)
    changed = (diff > threshold).float()
    if changed.mean().item() > max_changed:
        return edited
    # Opening drops isolated specks of re-rendered texture before the grow
    # would turn each into a patch.
    changed = -F.max_pool2d(-changed, 2 * speck + 1, stride=1, padding=speck)
    changed = F.max_pool2d(changed, 2 * speck + 1, stride=1, padding=speck)
    if grow:
        changed = F.max_pool2d(changed, 2 * grow + 1, stride=1, padding=grow)
    alpha = changed
    for _ in range(2):
        alpha = F.avg_pool2d(alpha, 2 * feather + 1, stride=1, padding=feather, count_include_pad=False)
    alpha = alpha.movedim(1, -1)
    out = edited.clone()
    out[..., :3] = source * (1 - alpha) + edited[..., :3] * alpha
    return out


class StimmaPreserveUnchanged:
    """Restore the source's exact pixels outside the edited region."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"edited": ("IMAGE",), "enabled": ("BOOLEAN", {"default": True})},
            "optional": {"source": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "Stimma/Utils"
    DESCRIPTION = (
        "Copy the source image's own pixels back wherever an edit left it unchanged, so untouched "
        "regions are bit-exact instead of re-rendered. Skipped for whole-image edits and when the "
        "sizes differ (multi-reference edits)."
    )

    def execute(self, edited, enabled=True, source=None):
        if not enabled or source is None:
            return (edited,)
        return (preserve_unchanged(source, edited),)
