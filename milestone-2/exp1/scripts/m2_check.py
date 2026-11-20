"""Milestone 2 check on the GPU: the shared protocol and Painter must paint the same pictures.

Runs the group's smoke check (the starting-noise sum and the anchor's timesteps, which must match
on every machine, and the anchor painted for the first 10 dev captions), then paints the same 10
captions with Painter while recording attention and compares the two sets of pictures pixel by
pixel. The word-timing run relies on this: its images double as the anchor's test-split images.

Run from the project folder:  python scripts/m2_check.py
Writes Results/m2_check/summary.json; the pictures go to data/m2_check/.
"""
import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

from w2w import m2_protocol as m2
from w2w.paths import DATA, RESULTS
from w2w.rewind import Painter

N = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out, img_dir = RESULTS / "m2_check", DATA / "m2_check"
    for d in (out, img_dir / "protocol", img_dir / "painter"):
        d.mkdir(parents=True, exist_ok=True)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    dev = m2.load_split("dev_500", root=RESULTS / "m2_splits")
    pipe = m2.load_pipeline(args.device)
    smoke = m2.smoke_check(pipe, dev, out_dir=img_dir / "protocol", n=N)
    painter = Painter(pipe, args.device, m2.ANCHOR.steps, m2.ANCHOR.guidance, spacing=m2.ANCHOR.schedule, cpu_noise=True)
    rows = []
    for r in dev.head(N).itertuples(index=False):
        seed = m2.seed_for(r.caption_idx, 0)
        sync()
        t0 = time.perf_counter()
        tl = painter.paint(r.caption, seed, record=True, previews=False, safe=True)
        sync()
        dt = time.perf_counter() - t0
        tl.final.save(img_dir / "painter" / f"{r.caption_idx}_s0.png")
        ref = np.asarray(Image.open(img_dir / "protocol" / f"{r.caption_idx}_s0.png").convert("RGB"), dtype=np.float32)
        diff = np.abs(ref - np.asarray(tl.final.convert("RGB"), dtype=np.float32))
        rows.append({"caption_idx": int(r.caption_idx), "max_pixel_difference": float(diff.max()),
                     "mean_pixel_difference": round(float(diff.mean()), 4), "flagged": tl.flagged,
                     "timesteps_match": tl.timesteps == m2.EXPECTED_ANCHOR_TIMESTEPS, "seconds": round(dt, 2)})
        print(f"  {r.caption_idx}: Painter vs protocol, max pixel difference {diff.max():.0f}, "
              f"mean {diff.mean():.4f}, {dt:.2f} s with attention recorded", flush=True)
    worst = max(x["max_pixel_difference"] for x in rows)
    summary = {
        "protocol_version": m2.PROTOCOL_VERSION, "protocol_sha": m2.protocol_sha(), **m2.environment(),
        "noise_sum": smoke["noise_sum"], "noise_ok": smoke["noise_ok"], "timesteps_ok": smoke["timesteps_ok"],
        "protocol_images": smoke["images"], "painter_images": rows,
        "max_pixel_difference": worst, "identical": worst == 0,
        "sec_per_image_protocol": round(float(np.mean([x["seconds"] for x in smoke["images"]])), 2),
        "sec_per_image_painter_recording": round(float(np.mean([x["seconds"] for x in rows])), 2),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"noise {'OK' if smoke['noise_ok'] else 'MISMATCH'}, timesteps {'OK' if smoke['timesteps_ok'] else 'MISMATCH'}, "
          f"Painter identical to the protocol: {worst == 0} (largest pixel difference {worst:.0f})")


if __name__ == "__main__":
    main()
