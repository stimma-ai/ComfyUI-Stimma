# Model Detection & Parameter Heuristics

## Canonical Parameter Name Requirements

A Stimma tool exposes a **single `parameter_schema`** (no separate `input_schema`). The backend
and generation agent look the media/prompt parameters below up **by name**, so they must be named
exactly as shown per task_type.

| Task type | Required parameter name(s) | Node |
|-----------|---------------------------|------|
| `text-to-image` | `prompt` | `StimmaPromptParam` (auto) |
| `image-to-image` | `prompt`, **`input_images`** (plural!) | `StimmaImageParam(name="input_images")` |
| `inpaint-image` | `input_image`, `mask` | `StimmaImageParam(name="input_image")` + `StimmaMaskParam` |
| `outpaint-image` | `input_image` | `StimmaImageParam(name="input_image")` |
| `upscale-image` | `input_image` | `StimmaImageParam(name="input_image")` |
| `text-to-video` | `prompt` | `StimmaPromptParam` (auto) |
| `image-to-video` | `input_image` (or `start_image`) | `StimmaImageParam(name="input_image")` |
| `video-extend` | `input_video` | `StimmaVideoParam(name="input_video")` |

**Key rule**: `image-to-image` → `input_images` (plural). All other image tasks → `input_image` (singular).

See full details in `SKILL.md` → "Canonical Parameter Names".

---

## Model Family Detection

The analysis script detects model family from checkpoint filenames and node types:

| Family | Filename Keywords | Node Type Signals | Typical Resolution |
|--------|------------------|-------------------|--------------------|
| flux | flux | ModelSamplingFlux, FluxGuidance, CLIPTextEncodeFlux | 1024x1024 |
| sdxl | sdxl, sd_xl | — | 1024x1024 |
| sd15 | sd1.5, v1-5 | — | 512x512 |
| sd3 | sd3 | ModelSamplingSD3, EmptySD3LatentImage | 1024x1024 |
| qwen | qwen, q-image | — | 1328x1328 |
| z-image | z_image, zimage, lumina | ModelSamplingAuraFlow | 1024x1024 |
| wan | wan, wan2 | — | varies |
| hunyuan | hunyuan | — | varies |

## Guidance-Distilled Detection

A model is "guidance-distilled" when it was trained to work without classifier-free guidance (CFG). For these models, cfg=1.0 effectively disables the guidance mechanism. Exposing CFG as a parameter would mislead users — changing it won't improve results and may hurt them.

**Detection signals (any match = guidance-distilled):**

1. **CFG locked to 1.0** in KSampler widget values — strongest signal
2. **Filename keywords**: lightning, turbo, schnell, klein, hyper
3. **Model family**: flux (all Flux variants use FluxGuidance instead of CFG)
4. **Z-Image Turbo**: always guidance-distilled

**When guidance-distilled:**
- Do NOT expose CFG parameter
- Do NOT expose negative prompt (CFG=1 means negative conditioning has no effect)
- The model may have guidance via FluxGuidance node instead — that's a different parameter to expose

## Parameter Decisions by Model Family

### Flux (all variants)
- Guidance-distilled: YES (uses FluxGuidance instead of CFG)
- Expose: prompt, resolution, seed, sampler, scheduler, steps
- Expose guidance (FluxGuidance value) — typically 3.0-3.5 for flux-dev, 1.0 for flux-schnell
- Expose shift if ModelSamplingFlux present (max_shift, base_shift)
- Skip: CFG, negative prompt
- Notes: Uses SamplerCustomAdvanced pipeline (BasicGuider, not CFGGuider)

### SDXL
- Guidance-distilled: check filename for "lightning", "turbo"
- If standard: expose prompt, negative_prompt, resolution, seed, sampler, scheduler, steps, cfg, denoise (if img2img)
- If lightning/turbo: skip CFG, skip negative_prompt, reduce step range (4-8 typical)
- Default resolution: 1024x1024, supported: 1024x1024, 1152x896, 896x1152, etc.

### SD 1.5
- Guidance-distilled: rarely
- Expose: prompt, negative_prompt, resolution, seed, sampler, scheduler, steps, cfg, denoise
- Default resolution: 512x512
- Full parameter set usually appropriate

### Qwen Image
- Guidance-distilled: check for lightning/turbo variants (Qwen-2512-Lightning uses cfg=1)
- Standard: expose full set including CFG, negative prompt
- Resolution: 1328x1328 base, supported aspect ratios vary by model
- May have shift parameter (ModelSamplingAuraFlow)

### Z-Image Turbo
- Guidance-distilled: YES always
- Skip: CFG, negative prompt
- Steps: typically 8-12 (low)
- May have muted ModelSamplingAuraFlow — the shift param is available but rarely needed

### WAN (video models)
- Often have 2 KSamplers sharing params
- First KSampler may be for structure, second for detail
- Ask user which params to share vs. separate
- Video output: use StimmaVideoOutput
- May need StimmaVideoParam for video-to-video

## Sampler Pipeline Types

### Standard (KSampler)
All params on one node: seed, steps, cfg, sampler_name, scheduler, denoise.
Widget values order: `[seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise]`

### Standard Advanced (KSamplerAdvanced)
Similar to KSampler but with start/end step control:
`[add_noise, seed, control_after_generate, steps, cfg, sampler_name, scheduler, start_at_step, end_at_step, return_with_leftover_noise]`

### Advanced Pipeline (SamplerCustomAdvanced)
Params spread across helper nodes:
- **RandomNoise**: seed — `[seed, control_after_generate]`
- **KSamplerSelect**: sampler_name — `[sampler_name]`
- **BasicScheduler**: scheduler, steps, denoise — `[scheduler, steps, denoise]`
- **FluxGuidance**: guidance — `[guidance]`
- **ModelSamplingFlux**: max_shift, base_shift, width, height — `[max_shift, base_shift, width, height]`
- **BasicGuider**: no widget values (just connects model + conditioning)

## Typical Default Ranges

| Parameter | Typical Default | Min | Max | Step |
|-----------|----------------|-----|-----|------|
| steps (normal) | 20 | 1 | 50 | 1 |
| steps (turbo) | 4-8 | 1 | 20 | 1 |
| cfg | 7.0 | 1.0 | 20.0 | 0.5 |
| denoise | 1.0 (t2i) / 0.7 (i2i) | 0.0 | 1.0 | 0.05 |
| guidance (Flux) | 3.5 | 1.0 | 10.0 | 0.1 |
| shift (AuraFlow) | 3.0 | 0.0 | 10.0 | 0.1 |
| max_shift (Flux) | 1.15 | 0.0 | 5.0 | 0.05 |
| base_shift (Flux) | 0.5 | 0.0 | 5.0 | 0.05 |

## Supported Resolutions by Family

### Flux / SDXL / SD3
```
1024x1024
1152x896
896x1152
1216x832
832x1216
1344x768
768x1344
1536x640
640x1536
```

### Qwen Image 2512
```
1328x1328
1664x928
928x1664
1472x1104
1104x1472
1584x1056
1056x1584
```

### SD 1.5
```
512x512
768x512
512x768
640x480
480x640
```
