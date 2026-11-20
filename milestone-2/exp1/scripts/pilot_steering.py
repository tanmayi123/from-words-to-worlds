"""Pilot: does each word have its own steering window?

Question. If we push a word's attention into a box starting at evaluation k, does the
object still move there when k is early, stop moving once k passes that word's commit
step, and does the word's own commit step predict where that cut-off falls better than
one fixed cut-off for every word?

Design (fixed before the run):
  20 prompts, each naming one COCO object (the target word), seeds 0 and 1.
  DDIM, 30 steps, guidance 7.5, 512 x 512. One UNet evaluation per step.
  Baseline run: no steering, attention recorded, the target word's commit step measured
  (threshold 0.8, also 0.7 and 0.9). The target box is the opposite side of the image from
  where the object naturally lands (detector centre, or the word's attention if the
  detector finds nothing).
  Steered runs: the bias switches on at evaluation k for k in STARTS and stays on.
  Judge: torchvision Faster R-CNN (COCO). Success = a detection of the target class with
  score >= 0.5 whose centre lies inside the box.
  Strength: the smallest of CAL_STRENGTHS that moves the object at k = 0 in at least 4 of
  the first 6 prompts (seed 0). If none does, the end-of-text slots are steered together
  with the word at the largest strength; if that also fails, the pilot stops and says so.

Pre-registered reading:
  steering works      k = 0 success rate >= 0.60 over usable units
  windows differ      interquartile range of commit steps >= 3 evaluations
  timing predicts     per-word rule (success iff k <= commit + d) beats one global rule
                      (success iff k <= K) by >= 5 points of leave-one-prompt-out accuracy,
                      and Spearman rho(commit, window close) > 0.3 with a bootstrap 95%
                      interval above 0
  PASS = all three. WEAK = the first two only. COLLAPSED = steering works but commits
  barely differ. NO_STEER = steering does not move objects even at k = 0.

Run from the project folder (milestone-1/exp1):  python scripts/pilot_steering.py
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from w2w.attention import AttentionStore, RecordingCrossAttnProcessor, commit_step, step_callback, word_tokens
from w2w.paths import DATA, RESULTS
from w2w.steer import SteerController, attach_eval_counter, steering_processors

PROMPTS = [
    ("a cat sitting on a wooden bench", "cat"),
    ("a dog lying on a couch next to a lamp", "dog"),
    ("a red umbrella on a sandy beach", "umbrella"),
    ("a horse standing in a green field", "horse"),
    ("a clock on a brick wall", "clock"),
    ("a bicycle parked next to a tree", "bicycle"),
    ("a bird perched on a wooden fence", "bird"),
    ("a cup of coffee on a wooden table", "cup"),
    ("a car parked on a quiet street", "car"),
    ("a giraffe standing near a tall tree", "giraffe"),
    ("a boat on a calm lake at sunset", "boat"),
    ("a vase of flowers on a windowsill", "vase"),
    ("a cow grazing in a field with mountains", "cow"),
    ("a laptop on a desk in a bright office", "laptop"),
    ("an apple and a banana on a plate", "banana"),
    ("a bench in a park with a dog", "dog"),
    ("a kitchen counter with a bottle", "bottle"),
    ("a city street with a bus", "bus"),
    ("a living room with a chair", "chair"),
    ("a grassy hill with a sheep", "sheep"),
]
STEPS, GUIDANCE, SEEDS = 30, 7.5, (0, 1)
STARTS = (0, 3, 6, 9, 12, 15, 18, 21)
CAL_STRENGTHS, CAL_PROMPTS, CAL_NEED = (2.5, 5.0, 10.0, 25.0), 6, 4
MIN_SCORE = 0.5
BOX_RIGHT, BOX_LEFT = (0.55, 0.25, 0.95, 0.95), (0.05, 0.25, 0.45, 0.95)
THRESHOLDS = (0.7, 0.8, 0.9)


# ------------------------------------------------------------------ generation

class Runner:
    def __init__(self, pipe, device):
        self.pipe, self.device = pipe, device
        self.ctrl = SteerController()
        self.store = AttentionStore()
        self.original = dict(pipe.unet.attn_processors)
        self.steer = steering_processors(self.original, self.ctrl)
        self.record = {n: (RecordingCrossAttnProcessor(self.store, inner=p) if "attn2" in n else p)
                       for n, p in self.steer.items()}
        self.hook = attach_eval_counter(pipe.unet, self.ctrl)
        self.mode = None

    def _set(self, mode):
        if mode != self.mode:
            self.pipe.unet.set_attn_processor(dict(self.record if mode == "record" else self.steer))
            self.mode = mode

    @torch.no_grad()
    def run(self, prompt, seed, record=False, slots=(), box=(0, 0, 1, 1), strength=1.0, start=None):
        self._set("record" if record else "steer")
        self.ctrl.configure(slots=slots, box=box, strength=strength, start=start)
        self.store.reset()
        g = torch.Generator(self.device).manual_seed(seed)
        out = self.pipe(prompt=prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE, generator=g,
                        callback_on_step_end=step_callback(self.store) if record else None)
        n_evals = self.ctrl.eval_idx + 1
        assert n_evals == STEPS, f"expected {STEPS} UNet evaluations, counted {n_evals}"
        nsfw = bool(out.nsfw_content_detected[0]) if getattr(out, "nsfw_content_detected", None) else False
        maps = self.store.stacked() if record else None
        return out.images[0], nsfw, maps

    def close(self):
        self.hook.remove()
        self.pipe.unet.set_attn_processor(dict(self.original))


def target_slot(tokenizer, prompt, word):
    hits = [i for i, t in word_tokens(tokenizer, prompt) if t == word]
    assert hits, f"'{word}' is not a single token in '{prompt}': {word_tokens(tokenizer, prompt)}"
    return hits[0]


def eos_slots(tokenizer, prompt):
    ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
    first = ids.index(tokenizer.eos_token_id)
    return list(range(first, 77))


# ------------------------------------------------------------------ judging

class Detector:
    def __init__(self, device):
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
        w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=w).eval().to(device)
        self.names = w.meta["categories"]
        self.device = device

    @torch.no_grad()
    def __call__(self, img, cls):
        """[(score, cx, cy)] for detections of `cls` with score >= MIN_SCORE, centres in [0, 1]."""
        x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
        r = self.model([x.to(self.device)])[0]
        W, H = img.size
        out = []
        for b, l, s in zip(r["boxes"].tolist(), r["labels"].tolist(), r["scores"].tolist()):
            if self.names[l] == cls and s >= MIN_SCORE:
                out.append((s, (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H))
        return sorted(out, reverse=True)


def in_box(dets, box):
    x0, y0, x1, y1 = box
    return any(x0 <= cx < x1 and y0 <= cy < y1 for _, cx, cy in dets)


def on_side(dets, box):
    right = box[0] >= 0.5
    return any((cx >= 0.5) == right for _, cx, _ in dets)


def natural_x(dets, maps, slot):
    if dets:
        return dets[0][1], "detector"
    ref = maps[-6:-1, slot].astype(np.float32).mean(0)
    ref = ref - ref.min()
    xs = (np.arange(ref.shape[1]) + 0.5) / ref.shape[1]
    return float((ref.sum(0) * xs).sum() / (ref.sum() + 1e-8)), "attention"


# ------------------------------------------------------------------ analysis

def window_close(succ_by_k):
    """Smallest start k from which every later start fails. len(STARTS) sentinel if never closes."""
    ks = sorted(succ_by_k)
    for i, k in enumerate(ks):
        if not any(succ_by_k[j] for j in ks[i:]):
            return k, False
    return ks[-1] + (ks[1] - ks[0]), True


def lopo_accuracy(df, col):
    """Leave-one-prompt-out accuracy of the per-word rule and the global rule."""
    word_hits = glob_hits = total = 0
    for p in sorted(df.prompt_id.unique()):
        tr, te = df[df.prompt_id != p], df[df.prompt_id == p]
        if te.empty:
            continue
        best_d = max(range(-30, 31), key=lambda d: (((tr.k <= tr[col] + d) == tr.success).mean(), -abs(d)))
        best_K = max(range(-1, 31), key=lambda K: (((tr.k <= K) == tr.success).mean(), -K))
        word_hits += int(((te.k <= te[col] + best_d) == te.success).sum())
        glob_hits += int(((te.k <= best_K) == te.success).sum())
        total += len(te)
    return word_hits / total, glob_hits / total


def spearman_ci(x, y, n_boot=2000, seed=0):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 1 and np.unique(y[i]).size > 1:
            boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi)


def figures(df, units, img_dir, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    ok = df[df.usable]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    a = ok.groupby("k").success.mean()
    ax[0].plot(a.index, a.values, marker="o", color="#2b6cb0")
    ax[0].set_xlabel("evaluation at which steering starts (k)")
    ax[0].set_ylabel("object moved into the box")
    ax[0].set_title("Against the absolute step", fontsize=10)
    rel = (ok.k - ok.commit_08)
    bins = np.arange(-24, 28, 3)
    b = ok.assign(rel=pd.cut(rel, bins)).groupby("rel", observed=True).success.agg(["mean", "size"])
    centres = [iv.mid for iv in b.index]
    ax[1].plot(centres, b["mean"].values, marker="o", color="#c05621")
    for cx, (m, n) in zip(centres, b.values):
        ax[1].annotate(str(int(n)), (cx, m), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax[1].axvline(0, color="grey", lw=0.8, ls="--")
    ax[1].set_xlabel("k minus the word's commit step")
    ax[1].set_title("Relative to each word's own commit", fontsize=10)
    for x in ax:
        x.set_ylim(-0.05, 1.05)
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "success_vs_start.png", dpi=150)
    plt.close(fig)

    # Three example units: earliest, middle and latest commit among usable units.
    u = units[units.usable].sort_values("commit_08")
    if u.empty:
        return
    picks = [u.iloc[0], u.iloc[len(u) // 2], u.iloc[-1]]
    thumb = 128
    cols = 1 + len(STARTS)
    sheet = Image.new("RGB", (cols * thumb, len(picks) * (thumb + 18)), "white")
    d = ImageDraw.Draw(sheet)
    for r, row in enumerate(picks):
        names = [f"{row.prompt_id:02d}_{row.seed}_base"] + [f"{row.prompt_id:02d}_{row.seed}_k{k:02d}" for k in STARTS]
        labels = ["no steering"] + [f"k={k}" for k in STARTS]
        box = json.loads(row.box)
        for c, (nm, lab) in enumerate(zip(names, labels)):
            im = Image.open(img_dir / f"{nm}.png").convert("RGB").resize((thumb, thumb))
            di = ImageDraw.Draw(im)
            di.rectangle([box[0] * thumb, box[1] * thumb, box[2] * thumb, box[3] * thumb], outline=(255, 210, 0), width=2)
            y = r * (thumb + 18)
            sheet.paste(im, (c * thumb, y + 18))
            if c == 0:
                d.text((2, y + 2), f"'{row.target}' commits at {row.commit_08}", fill="black")
            else:
                hit = df[(df.prompt_id == row.prompt_id) & (df.seed == row.seed) & (df.k == STARTS[c - 1])].success.item()
                d.text((c * thumb + 2, y + 2), f"{lab} {'moved' if hit else '-'}", fill="black")
    sheet.save(out_dir / "examples.png")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="override the Stable Diffusion repo (testing only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fake-detector", action="store_true", help="testing only: random detections")
    ap.add_argument("--limit", type=int, default=None, help="testing only: first N prompts")
    ap.add_argument("--out", default=None, help="testing only: output folder name")
    ap.add_argument("--cal-need", type=int, default=CAL_NEED, help="testing only: calibration bar")
    args = ap.parse_args()

    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from w2w.generate import SD_ID
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(args.model or SD_ID, dtype=dtype).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    tok = pipe.tokenizer

    name = args.out or "m2_pilot_steering"
    out_dir, img_dir = RESULTS / name, DATA / name
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    if args.fake_detector:
        rng = np.random.default_rng(0)
        detect = lambda img, cls: [(0.9, float(rng.random()), float(rng.random()))] if rng.random() < 0.8 else []
    else:
        detect = Detector(args.device)

    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    runner = Runner(pipe, args.device)
    t0 = time.time()

    # Parity: steering processors with steering off against the stock processors.
    pipe.unet.set_attn_processor(dict(runner.original))
    g = torch.Generator(args.device).manual_seed(0)
    stock = np.asarray(pipe(prompt=prompts[0][0], num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                            generator=g).images[0], dtype=np.float32)
    runner.mode = None
    ours, _, _ = runner.run(prompts[0][0], 0)
    parity = float(np.abs(stock - np.asarray(ours, dtype=np.float32)).mean())
    print(f"parity: mean |stock - steering path, steering off| = {parity:.4f} (0-255 scale)", flush=True)

    base = {}

    def baseline(pid, seed):
        if (pid, seed) not in base:
            prompt, word = prompts[pid]
            img, nsfw, maps = runner.run(prompt, seed, record=True)
            img.save(img_dir / f"{pid:02d}_{seed}_base.png")
            slot = target_slot(tok, prompt, word)
            np.save(img_dir / f"{pid:02d}_{seed}_target_maps.npy", maps[:, slot])
            dets = detect(img, word)
            nx, src = natural_x(dets, maps, slot)
            box = BOX_LEFT if nx >= 0.5 else BOX_RIGHT
            base[(pid, seed)] = dict(
                img=img, nsfw=nsfw, slot=slot, dets=dets, box=box, natural_x=nx, natural_from=src,
                commits={th: commit_step(maps, slot, threshold=th) for th in THRESHOLDS})
        return base[(pid, seed)]

    # Calibration: smallest strength that moves the object at k = 0.
    cal_log, chosen, use_eos = [], None, False
    for s in CAL_STRENGTHS:
        hits = 0
        for pid in range(min(CAL_PROMPTS, len(prompts))):
            b = baseline(pid, 0)
            img, _, _ = runner.run(prompts[pid][0], 0, slots=[b["slot"]], box=b["box"], strength=s, start=0)
            hits += in_box(detect(img, prompts[pid][1]), b["box"])
        cal_log.append({"strength": s, "eos": False, "moved": hits, "of": min(CAL_PROMPTS, len(prompts))})
        print(f"calibration: strength {s}, word only: {hits} moved", flush=True)
        if hits >= args.cal_need:
            chosen = s
            break
    if chosen is None:
        s = CAL_STRENGTHS[-1]
        hits = 0
        for pid in range(min(CAL_PROMPTS, len(prompts))):
            b = baseline(pid, 0)
            sl = [b["slot"]] + eos_slots(tok, prompts[pid][0])
            img, _, _ = runner.run(prompts[pid][0], 0, slots=sl, box=b["box"], strength=s, start=0)
            hits += in_box(detect(img, prompts[pid][1]), b["box"])
        cal_log.append({"strength": s, "eos": True, "moved": hits, "of": min(CAL_PROMPTS, len(prompts))})
        print(f"calibration: strength {s}, word + end tokens: {hits} moved", flush=True)
        if hits >= args.cal_need:
            chosen, use_eos = s, True

    summary = {"settings": {"steps": STEPS, "guidance": GUIDANCE, "scheduler": "DDIM", "seeds": list(SEEDS),
                            "starts": list(STARTS), "min_score": MIN_SCORE, "box_left": BOX_LEFT,
                            "box_right": BOX_RIGHT, "prompts": len(prompts)},
               "parity_mean_abs_pixel_diff": parity, "calibration": cal_log,
               "strength": chosen, "steer_end_tokens": use_eos}
    if chosen is None:
        summary["verdict"] = "NO_STEER"
        summary["note"] = "No strength moved the object at k = 0 in enough calibration prompts; main run skipped."
        json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2)
        print(json.dumps(summary, indent=2))
        runner.close()
        return

    rows, unit_rows = [], []
    for pid, (prompt, word) in enumerate(prompts):
        for seed in SEEDS:
            b = baseline(pid, seed)
            slots = [b["slot"]] + (eos_slots(tok, prompt) if use_eos else [])
            base_in = in_box(b["dets"], b["box"])
            c = b["commits"]
            usable = (not base_in) and c[0.8] is not None and not b["nsfw"]
            succ = {}
            for k in STARTS:
                img, nsfw, _ = runner.run(prompt, seed, slots=slots, box=b["box"], strength=chosen, start=k)
                img.save(img_dir / f"{pid:02d}_{seed}_k{k:02d}.png")
                dets = detect(img, word)
                succ[k] = in_box(dets, b["box"]) and not nsfw
                rows.append(dict(prompt_id=pid, prompt=prompt, target=word, seed=seed, k=k,
                                 success=bool(succ[k]), on_side=on_side(dets, b["box"]), present=bool(dets),
                                 nsfw=nsfw, usable=usable, commit_07=c[0.7], commit_08=c[0.8], commit_09=c[0.9]))
            close, censored = window_close(succ)
            unit_rows.append(dict(prompt_id=pid, prompt=prompt, target=word, seed=seed, usable=usable,
                                  baseline_present=bool(b["dets"]), baseline_in_box=base_in, nsfw=b["nsfw"],
                                  natural_x=round(b["natural_x"], 3), natural_from=b["natural_from"],
                                  box=json.dumps(b["box"]), commit_07=c[0.7], commit_08=c[0.8], commit_09=c[0.9],
                                  k0_success=succ[STARTS[0]], window_close=close, close_censored=censored))
            print(f"[{time.time() - t0:6.0f}s] {pid:02d} seed {seed} '{word}': commit {c[0.8]}, "
                  f"moved at k = {[k for k in STARTS if succ[k]]}", flush=True)
    runner.close()

    df, units = pd.DataFrame(rows), pd.DataFrame(unit_rows)
    df.to_csv(out_dir / "runs.csv", index=False, lineterminator="\n")
    units.to_csv(out_dir / "units.csv", index=False, lineterminator="\n")

    ok_u, ok = units[units.usable], df[df.usable]
    k0 = float(ok_u.k0_success.mean()) if len(ok_u) else float("nan")
    q1, q3 = (np.percentile(ok_u.commit_08, [25, 75]) if len(ok_u) else (np.nan, np.nan))
    spread = {"min": int(ok_u.commit_08.min()), "q1": float(q1), "median": float(ok_u.commit_08.median()),
              "q3": float(q3), "max": int(ok_u.commit_08.max()), "iqr": float(q3 - q1)} if len(ok_u) else {}
    acc = {}
    for th, col in zip(THRESHOLDS, ("commit_07", "commit_08", "commit_09")):
        sub = ok.dropna(subset=[col])
        if sub.prompt_id.nunique() >= 3:
            w, g_ = lopo_accuracy(sub, col)
            acc[str(th)] = {"per_word": round(w, 4), "global": round(g_, 4), "gain": round(w - g_, 4)}
    corr = {}
    wu = ok_u[ok_u.k0_success]
    for th, col in zip(THRESHOLDS, ("commit_07", "commit_08", "commit_09")):
        sub = wu.dropna(subset=[col])
        if len(sub) >= 5 and sub[col].nunique() > 1 and sub.window_close.nunique() > 1:
            r, lo, hi = spearman_ci(sub[col], sub.window_close)
            corr[str(th)] = {"rho": round(r, 3), "ci95": [round(lo, 3), round(hi, 3)], "n": int(len(sub)),
                             "censored": int(sub.close_censored.sum())}

    works = k0 >= 0.60
    differ = spread.get("iqr", 0) >= 3
    a8, c8 = acc.get("0.8", {}), corr.get("0.8", {})
    predicts = bool(a8 and c8 and a8["gain"] >= 0.05 and c8["rho"] > 0.3 and c8["ci95"][0] > 0)
    verdict = ("NO_STEER" if not works else "COLLAPSED" if not differ else "PASS" if predicts else "WEAK")

    summary.update({
        "units": int(len(units)), "usable_units": int(len(ok_u)),
        "excluded": {"baseline_already_in_box": int(units.baseline_in_box.sum()),
                     "never_committed": int(units.commit_08.isna().sum()), "nsfw": int(units.nsfw.sum())},
        "baseline_object_present": float(units.baseline_present.mean()),
        "k0_success_rate": round(k0, 4),
        "success_by_k": {int(k): round(float(v), 4) for k, v in ok.groupby("k").success.mean().items()},
        "commit_spread_08": spread, "lopo_accuracy": acc, "spearman_commit_vs_close": corr,
        "criteria": {"steering_works": works, "windows_differ": differ, "timing_predicts": predicts},
        "verdict": verdict, "minutes": round((time.time() - t0) / 60, 1)})
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2)
    figures(df, units, img_dir, out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
