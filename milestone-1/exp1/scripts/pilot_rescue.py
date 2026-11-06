"""Pilot 5: rescue or regenerate? Fixing a missing object while keeping the picture.

Reads the failures that pilot 4 (pilot_recall.py) found on dev_500. Run that first.

Design.
  A failure is an object the caption names and the photo contains, that neither detector finds
  in the seed-0 painting (pilot 4's "missing"). The first such object of each caption is used,
  up to N_FAIL failures, in split order. The seed-0 painting is repainted exactly (and checked
  against pilot 4's saved image). Each failure then gets T tries under each condition:
    regen        a new painting from a new seed (1 to T), the normal sampler (DDIM eta 0):
                 what a user does today when an object is missing
    regen_eta1   the same new seeds with the noisy sampler (eta 1) that rescue uses, so any
                 quality difference caused by the sampler itself shows up here
    rescue_k     go back to after step k of the original painting and finish it with fresh
                 randomness (eta 1, noise seed 1000 k + try), for k in RESCUE_KS
  Each try is judged by:
    fixed   either detector finds the object at score >= 0.5 (pilot 4's "present")
    LPIPS   perceptual distance to the original painting (AlexNet LPIPS; 0 = identical)
    CLIPScore with the caption, and sharpness (variance of the Laplacian), as quality checks
  The scene counts as kept when LPIPS <= TAU_SHARE x the median LPIPS between the original and
  the regen tries (two unrelated paintings of the same caption), a threshold set by the data.

Per condition, over failures:
  fix@1      share of tries that are fixed;     fix@T  share of failures fixed within T tries
  kept@T     share of failures with a try that is fixed AND keeps the scene
  cost       steps painted per try as a share of a full painting ((30 - k) / 30 for rescue_k)

Pre-registered reading (paired bootstrap over failures, 2,000 resamples). The best rescue
step k* is the one with the highest kept@T (ties go to the later k, which is cheaper). The rules
are checked in this order:
  NO_RESCUE   kept@T(k*) < 0.20: going back rarely fixes the object and keeps the scene
  BOTH        kept@T(k*) - kept@T(regen) has an interval above 0, and fix@T(regen) - fix@T(k*)
              <= 0.10: rescue fixes nearly as often as regenerating and keeps the picture
  TRADE_OFF   the kept@T interval is above 0, but regeneration fixes more than 0.10 more
  UNCLEAR     anything else
  Sampler check: SAMPLER_EFFECT if the median sharpness of regen_eta1 is below 0.8 x regen's, or
  their paired CLIPScore difference has an interval excluding 0; otherwise SAME.
  k* is chosen here, on the development split, and is then fixed for the test-split run. Because
  k* is the best of four on these same failures, its intervals here are optimistic; BOTH and
  TRADE_OFF count as claims only once the test-split run confirms them.

Outputs: Results/m2_pilot_rescue/ (tries.csv, failures.csv, summary.json, rescue_curve.png,
rescue_example_*.png); images: data/m2_pilot_rescue/.

Run from the project folder (milestone-1/exp1):  python scripts/pilot_rescue.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_moments import strip  # noqa: E402
from pilot_recall import PRESENT_TH, TwoDetectors  # noqa: E402
from w2w.paths import DATA, RESULTS  # noqa: E402
from w2w.rewind import Painter  # noqa: E402

STEPS, GUIDANCE, BASE_SEED = 30, 7.5, 0
N_FAIL, T, RESCUE_KS = 60, 4, (3, 6, 9, 12)
TAU_SHARE, N_BOOT = 0.5, 2000
CONDITIONS = ["regen", "regen_eta1"] + [f"rescue_{k:02d}" for k in RESCUE_KS]


# ------------------------------------------------------------------ measures

class Measures:
    def __init__(self, device):
        import lpips
        from transformers import CLIPModel, CLIPProcessor
        self.lpips = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.device = device

    def _t(self, img):
        x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0)
        return x.permute(2, 0, 1)[None].to(self.device)

    @torch.no_grad()
    def distance(self, a, b):
        return float(self.lpips(self._t(a), self._t(b)).item())

    @torch.no_grad()
    def clipscore(self, img, text):
        x = self.proc(text=[text], images=img, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.clip(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                        pixel_values=x["pixel_values"].to(self.clip.dtype))
        i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float(100 * max(0.0, float((i @ t.T)[0, 0])))


def sharpness(img):
    from scipy.ndimage import laplace
    return float(laplace(np.asarray(img.convert("L"), dtype=np.float32)).var())


def cost(cond):
    return (STEPS - int(cond.split("_")[1])) / STEPS if cond.startswith("rescue_") else 1.0


# ------------------------------------------------------------------ analysis

def per_failure(tries, tau):
    """One row per (failure, condition): fixed within T, fix rate, kept fix, best LPIPS among fixes."""
    rows = []
    for (fid, cond), g in tries.groupby(["fid", "cond"], sort=False):
        fixed = g[g.fixed]
        rows.append({"fid": fid, "cond": cond, "fixed_any": bool(len(fixed)), "fix_rate": float(g.fixed.mean()),
                     "kept_fix": bool(((g.fixed) & (g.lpips <= tau)).any()),
                     "best_lpips_fixed": float(fixed.lpips.min()) if len(fixed) else float("nan")})
    return pd.DataFrame(rows)


def paired_diff(pf, a, b, col, n_boot=N_BOOT, seed=0):
    """mean(col | b) - mean(col | a) over failures, with a paired bootstrap 95% interval."""
    w = pf.pivot(index="fid", columns="cond", values=col).astype(float)
    d = (w[b] - w[a]).dropna().to_numpy()
    if d.size == 0:
        return {"diff": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, d.size), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": round(float(d.mean()), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "n": int(d.size)}


def best_rescue(table):
    """The rescue condition with the highest kept@T; ties go to the later (cheaper) step."""
    rescue = [c for c in table if c.startswith("rescue_")]
    return max(rescue, key=lambda c: (table[c]["kept_at_T"], int(c.split("_")[1])))


def read(table, kept_gain, fix_gap):
    k = best_rescue(table)
    if table[k]["kept_at_T"] < 0.20:
        return "NO_RESCUE"
    if kept_gain["ci95"][0] > 0:
        return "BOTH" if fix_gap["diff"] <= 0.10 else "TRADE_OFF"
    return "UNCLEAR"


def analyse(tries):
    regen = tries[tries.cond == "regen"]
    floor = float(regen.lpips.median())
    tau = TAU_SHARE * floor
    pf = per_failure(tries, tau)
    table = {}
    for cond in CONDITIONS:
        g, p = tries[tries.cond == cond], pf[pf.cond == cond]
        table[cond] = {"fix_at_1": round(float(g.fixed.mean()), 3), "fix_at_T": round(float(p.fixed_any.mean()), 3),
                       "kept_at_T": round(float(p.kept_fix.mean()), 3),
                       "median_best_lpips_fixed": round(float(p.best_lpips_fixed.median()), 3),
                       "median_lpips_all": round(float(g.lpips.median()), 3), "cost_per_try": round(cost(cond), 3),
                       "median_clipscore": round(float(g.clipscore.median()), 2),
                       "median_sharpness": round(float(g.sharpness.median()), 1)}
    k = best_rescue(table)
    kept_gain = paired_diff(pf, "regen", k, "kept_fix")
    fix_gap = paired_diff(pf, k, "regen", "fixed_any")
    sharp_ratio = table["regen_eta1"]["median_sharpness"] / max(1e-9, table["regen"]["median_sharpness"])
    clip_diff = paired_diff(tries.groupby(["fid", "cond"], as_index=False).clipscore.mean(), "regen", "regen_eta1", "clipscore")
    sampler = "SAMPLER_EFFECT" if (sharp_ratio < 0.8 or clip_diff["ci95"][0] > 0 or clip_diff["ci95"][1] < 0) else "SAME"
    never = pf.groupby("fid").fixed_any.any()
    return pf, {"failures": int(tries.fid.nunique()), "lpips_floor_regen_median": round(floor, 3), "tau": round(tau, 3),
                "by_condition": table, "best_rescue": k, "kept_gain_best_vs_regen": kept_gain,
                "fix_gap_regen_minus_best": fix_gap, "verdict": read(table, kept_gain, fix_gap),
                "sampler": {"sharpness_ratio_eta1_over_eta0": round(sharp_ratio, 3),
                            "clipscore_eta1_minus_eta0": clip_diff, "verdict": sampler},
                "never_fixed_by_any_condition": int((~never).sum())}


def plot(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tab = summary["by_condition"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for cond, v in tab.items():
        ax.scatter(v["median_best_lpips_fixed"], v["fix_at_T"], s=60, color="#e45756" if cond.startswith("rescue") else "#4c78a8")
        ax.annotate(cond.replace("rescue_", "from step "), (v["median_best_lpips_fixed"], v["fix_at_T"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axvline(summary["tau"], ls="--", color="grey", lw=1)
    ax.set_xlabel("change to the original picture (median LPIPS of the best fix)")
    ax.set_ylabel(f"failures fixed within {T} tries")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "rescue_curve.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recall", default="m2_pilot_recall")
    ap.add_argument("--out", default="m2_pilot_rescue")
    args = ap.parse_args()

    inst = pd.read_csv(RESULTS / args.recall / "instances.csv")
    fails = inst[inst.judge == "missing"].drop_duplicates("caption_idx", keep="first").head(N_FAIL).reset_index(drop=True)
    assert len(fails), "pilot 4 found no failures; run scripts/pilot_recall.py first"

    from diffusers import StableDiffusionPipeline
    from PIL import Image
    from w2w.generate import SD_ID
    pipe = StableDiffusionPipeline.from_pretrained(SD_ID, dtype=torch.float16).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    painter = Painter(pipe, args.device, STEPS, GUIDANCE)
    det, meas = TwoDetectors(args.device), Measures(args.device)

    out_dir, img_dir = RESULTS / args.out, DATA / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    rows, base_diffs = [], []
    for fid, f in enumerate(fails.itertuples()):
        base = painter.paint(f.caption, BASE_SEED, previews=False, record=False)
        saved = np.asarray(Image.open(DATA / args.recall / f"{f.caption_idx}.png").convert("RGB"), dtype=np.float32)
        base_diffs.append(float(np.abs(np.asarray(base.final, dtype=np.float32) - saved).mean()))
        assert base_diffs[-1] < 0.5, f"failure {fid}: the seed-0 repaint differs from pilot 4's image ({base_diffs[-1]:.3f})"
        base.final.save(img_dir / f"{fid:02d}_base.png")
        for cond in CONDITIONS:
            for t in range(T):
                if cond == "regen":
                    img = painter.paint(f.caption, 1 + t, previews=False, record=False).final
                elif cond == "regen_eta1":
                    img = painter.paint(f.caption, 1 + t, eta=1.0, noise_seed=7000 + t, previews=False, record=False).final
                else:
                    k = int(cond.split("_")[1])
                    img = painter.paint(f.caption, BASE_SEED, branch_from=base, at=k, eta=1.0, noise_seed=1000 * k + t,
                                        previews=False, record=False).final
                img.save(img_dir / f"{fid:02d}_{cond}_t{t}.png")
                s1, s2 = det.scores(img).get(f.object, (0.0, 0.0))
                rows.append({"fid": fid, "caption_idx": int(f.caption_idx), "object": f.object, "cond": cond, "t": t,
                             "frcnn": round(s1, 4), "retina": round(s2, 4), "fixed": max(s1, s2) >= PRESENT_TH,
                             "lpips": round(meas.distance(img, base.final), 4),
                             "clipscore": round(meas.clipscore(img, f.caption), 3), "sharpness": round(sharpness(img), 2)})
        done = pd.DataFrame(rows)
        fixed = done[done.fid == fid].groupby("cond").fixed.sum().to_dict()
        print(f"[{(time.time() - t0) / 60:5.1f} min] failure {fid:02d} '{f.object}' in '{f.caption}': "
              f"fixed tries {fixed}, base repaint diff {base_diffs[-1]:.3f}", flush=True)

    tries = pd.DataFrame(rows)
    tries.to_csv(out_dir / "tries.csv", index=False, lineterminator="\n")
    pf, summary = analyse(tries)
    pf.to_csv(out_dir / "failures.csv", index=False, lineterminator="\n")
    summary = {"settings": {"steps": STEPS, "guidance": GUIDANCE, "base_seed": BASE_SEED, "failures_wanted": N_FAIL,
                            "tries": T, "rescue_ks": list(RESCUE_KS), "present_th": PRESENT_TH, "tau_share": TAU_SHARE,
                            "bootstrap": N_BOOT, "scheduler": "DDIM"},
               "base_repaint_max_mean_abs_diff": round(max(base_diffs), 4), **summary,
               "minutes": round((time.time() - t0) / 60, 1)}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    plot(summary, out_dir)
    for fid in range(min(6, len(fails))):
        strip([img_dir / f"{fid:02d}_base.png"] + [img_dir / f"{fid:02d}_{c}_t0.png" for c in CONDITIONS],
              ["original"] + [c.replace("rescue_", "from ") for c in CONDITIONS], out_dir / f"rescue_example_{fid:02d}.png",
              f"missing '{fails.object[fid]}': {fails.caption[fid]}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
