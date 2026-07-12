"""StimmaVideoStitchAssembler — assemble an N-clip stitch from clips + morphs.

The workflow generates one FLW transition ("morph") per seam; unused clip slots
and their morph pipelines strip away, so this node receives the provided clips
(1..k, contiguous) and morphs (1..k-1) as its non-None optional inputs. It does
the position-dependent trimming in Python, where it is trivial:

    final = clip1[:-trimA] + morph1 + clip2[trimB:-trimA] + ... + clipK[trimB:]

Each seam discards trimA frames off the left clip's tail and trimB off the right
clip's head (trimA+trimB == the morph length), so total length is unchanged and
the audio stays synced when we simply concatenate every clip's full audio track.
"""

MAX_CLIPS = 10


class StimmaVideoStitchAssembler:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, MAX_CLIPS + 1):
            optional[f"clip_{i}"] = ("IMAGE",)
            optional[f"audio_{i}"] = ("AUDIO",)
        for i in range(1, MAX_CLIPS):
            optional[f"morph_{i}"] = ("IMAGE",)
        return {
            "required": {
                "trimA": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "trimB": ("INT", {"default": 0, "min": 0, "max": 100000}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "execute"
    CATEGORY = "Stimma/Params"

    def execute(self, trimA, trimB, **kw):
        import torch

        clips = [kw.get(f"clip_{i}") for i in range(1, MAX_CLIPS + 1)]
        clips = [c for c in clips if c is not None]
        morphs = [kw.get(f"morph_{i}") for i in range(1, MAX_CLIPS)]
        morphs = [m for m in morphs if m is not None]
        audios = [kw.get(f"audio_{i}") for i in range(1, MAX_CLIPS + 1)]
        audios = [a for a in audios if a is not None]

        k = len(clips)
        if k == 0:
            raise RuntimeError("StimmaVideoStitchAssembler: no clips provided")

        # Common resolution: the morphs' size (generated at clip 1's /32 dims) so the
        # original clip frames line up with the generated transitions. Clips of other
        # sizes are scaled to match; clip 1 is usually already the target.
        ref = morphs[0] if morphs else clips[0]
        th, tw = int(ref.shape[1]), int(ref.shape[2])

        def fit(img):
            if img.shape[1] == th and img.shape[2] == tw:
                return img
            import torch.nn.functional as F
            x = img.movedim(-1, 1)
            x = F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)
            return x.movedim(1, -1).contiguous()

        segments = []
        for i, clip in enumerate(clips):
            head = trimB if i > 0 else 0          # head-trim every clip but the first
            tail = trimA if i < k - 1 else 0      # tail-trim every clip but the last
            n = clip.shape[0]
            kept = clip[head:n - tail] if (head or tail) else clip
            segments.append(fit(kept))
            if i < k - 1 and i < len(morphs):
                segments.append(fit(morphs[i]))
        frames = torch.cat(segments, dim=0)

        audio = self._concat_audio(audios)
        return (frames, audio)

    @staticmethod
    def _concat_audio(audios):
        import torch

        tracks = [a for a in audios if a and a.get("waveform") is not None]
        if not tracks:
            return {"waveform": torch.zeros((1, 2, 1), dtype=torch.float32), "sample_rate": 44100}

        rate = int(tracks[0]["sample_rate"])
        chans = max(int(a["waveform"].shape[1]) for a in tracks)
        segs = []
        for a in tracks:
            w = a["waveform"]  # (batch, channels, samples)
            if w.dim() == 2:
                w = w.unsqueeze(0)
            sr = int(a["sample_rate"])
            if sr != rate:
                try:
                    import torchaudio
                    w = torchaudio.functional.resample(w, sr, rate)
                except Exception:
                    pass
            if w.shape[1] == 1 and chans > 1:
                w = w.repeat(1, chans, 1)
            elif w.shape[1] > chans:
                w = w[:, :chans, :]
            segs.append(w)
        return {"waveform": torch.cat(segs, dim=-1), "sample_rate": rate}


NODE_CLASS_MAPPINGS = {"StimmaVideoStitchAssembler": StimmaVideoStitchAssembler}
NODE_DISPLAY_NAME_MAPPINGS = {"StimmaVideoStitchAssembler": "Stimma Video Stitch Assembler"}
