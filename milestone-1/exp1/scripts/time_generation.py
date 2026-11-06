"""Time end-to-end generation on this machine and save the result next to the samples.

Run from the project root, plugged in:  python scripts/time_generation.py
"""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
from w2w import encode, generate  # noqa: E402
from w2w.paths import RESULTS  # noqa: E402

cache = encode.load_cache()
pipe = generate.load_pipeline()
caption_rows = [3730, 8367, 10240, 16931, 24712]   # the five Milestone 1 sample captions

generate.generate_from_states(pipe, cache["states"][caption_rows[0]], cache["null"], seed=999, steps=5)  # warm-up
times = []
for ci in caption_rows:
    for seed in (0, 1):
        _, _, dt = generate.generate_from_states(pipe, cache["states"][ci], cache["null"], seed=seed)
        times.append(round(dt, 3))

out = {
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "python": platform.python_version(),
    "steps": 30, "guidance": 7.5, "resolution": 512, "dtype": "float16",
    "includes": "UNet denoising, VAE decode and safety checker; one warm-up call excluded",
    "n_images": len(times),
    "seconds_per_image": times,
    "mean_seconds_per_image": round(sum(times) / len(times), 2),
}
(RESULTS / "m1_pipeline" / "timing.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
