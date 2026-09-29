"""One isolated MOSS job. GPU memory is released when this process exits."""

import json
import os
from pathlib import Path
import sys

# Match upstream's inference fallback; compilation is fragile across GPU types.
os.environ["TORCHDYNAMO_DISABLE"] = "1"


def main():
    import numpy as np
    import torch
    from moss_soundeffect_v2 import MossSoundEffectPipeline

    job_dir = Path(sys.argv[1])
    request = json.loads((job_dir / "request.json").read_text())

    def progress(iterable):
        total = len(iterable)
        for index, item in enumerate(iterable):
            yield item
            temp = job_dir / "progress.tmp"
            temp.write_text(json.dumps([index + 1, total]))
            temp.replace(job_dir / "progress.json")

    pipeline = MossSoundEffectPipeline.from_pretrained(
        request.pop("model_directory"), device=request.pop("device"),
        torch_dtype=torch.bfloat16, local_files_only=True,
    )
    with torch.inference_mode():
        audio = pipeline(**request, progress_bar_cmd=progress)
    np.save(job_dir / "audio.npy", audio.detach().float().cpu().numpy(), allow_pickle=False)
    (job_dir / "result.json").write_text(json.dumps({"sample_rate": pipeline.sample_rate}))


if __name__ == "__main__":
    main()
