"""BFL's updated FLUX.3 sampling and local prompt upsampling defaults."""
import json
import math
from pathlib import Path

DEFAULT_MODEL = 'qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors'


class StimmaFlux3Sampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('MODEL',), 'alpha': ('FLOAT', {'default': 16.0, 'min': 1.0, 'max': 100.0})}}

    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'execute'
    CATEGORY = 'Stimma/Utils'

    def execute(self, model, alpha=16.0):
        from comfy_extras.nodes_model_advanced import ModelSamplingFlux
        shift = math.log(alpha)
        # Equal shifts make this independent of canvas size, as in BFL's graphs.
        return ModelSamplingFlux().patch(model, shift, shift, 2048, 2048)


class StimmaFlux3PromptUpsample:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        models = folder_paths.get_filename_list('text_encoders')
        if DEFAULT_MODEL not in models:
            models = [DEFAULT_MODEL] + models
        return {'required': {
            'prompt': ('STRING', {'multiline': True}),
            'enabled': ('BOOLEAN', {'default': True}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'model_name': (models, {'default': DEFAULT_MODEL}),
        }, 'optional': {f'image_{i}': ('IMAGE',) for i in range(1, 11)}}

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'execute'
    CATEGORY = 'Stimma/Utils'

    def execute(self, prompt, enabled=True, seed=0, model_name=DEFAULT_MODEL, **images):
        if not enabled:
            return (prompt,)
        import nodes
        refs = [images[f'image_{i}'] for i in range(1, 11) if images.get(f'image_{i}') is not None]
        # The VLM accepts a list of differently sized images; don't rescale them
        # into an artificial contact sheet or change their attachment order.
        refs = [frame[None] for batch in refs for frame in batch][:10]
        prompts = json.loads(Path(__file__).with_name('flux3_pup_prompts.json').read_text())
        system = prompts['i2i' if refs else 't2i']
        request = ('Reference images follow in order (the first image, the second image, ...).\n'
                   f'User request: {prompt}\nOutput the caption.') if refs else prompt
        clip = nodes.CLIPLoader().load_clip(model_name, 'flux3')[0]
        tokens = clip.tokenize(request, images=refs, min_length=1,
                               thinking=False, system_prompt=system)
        generated = clip.generate(
            tokens, do_sample=True, max_length=3584, temperature=1.0,
            top_k=1, top_p=0.95, min_p=0.0, repetition_penalty=1.0,
            seed=seed, presence_penalty=0.0 if refs else 1.5, mtp=False,
        )
        text = clip.decode(generated).strip()
        if text.startswith('<think>'):
            text = text.partition('</think>')[2].strip()
        if not text:
            raise RuntimeError('FLUX.3 prompt upsampler returned an empty caption')
        return (text,)
