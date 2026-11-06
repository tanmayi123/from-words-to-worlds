"""Pilot 2: changing a painting while it is being painted.

Task 09 found that biasing one word's attention scores does not move its object. This
pilot tests three other ways to change a picture partway through, and for each one asks
the same timing question: is there a step after which the change no longer takes, and
does the step at which the word committed in the unchanged run predict it?

Arm "swap"   (word swap). The prompt is switched from step k onward, with one object
             word replaced ("cat" -> "dog"). Judge: the detector finds the new object.
Arm "detail" (detail edit). The prompt is switched from step k onward to a version that
             adds a detail to the object ("a smiling cat", "a red car"). Judge: CLIP. For
             each image, s = cos(image, edited caption) - cos(image, original caption);
             e(k) = (s(k) - s(original run)) / (s(edited run) - s(original run)), and the
             edit counts as taken when e(k) >= 0.5. Edits whose fully edited run beats the
             original by less than 0.01 in s are excluded as "the model cannot draw this".
Arm "drag"   (backward guidance, Chen et al. WACV 2024). From step k for G steps, the
             latents are nudged by gradient descent so the word's attention moves into a
             box on the other side of the image. Judge: the detector, as in task 09. The
             step size is the smallest of DRAG_ETAS that moves the object at k = 0 in at
             least 4 of the first 6 prompts; if none does, the arm stops (NO_STEER).

All arms: Stable Diffusion 1.5, DDIM, 30 steps, guidance 7.5, seeds 0 and 1. The same
initial noise for every condition of a prompt and seed. In the "k = 0" condition the
change is present from the start. The unchanged run records attention, and the object
word's commit step is measured at thresholds 0.7, 0.8 and 0.9.

Pre-registered reading, per arm (the same rules as task 09):
  change works     swap: the new object appears at k = 0 in >= 80% of usable units
                   detail: e >= 0.5 at k = 9 in >= 50% of usable units
                   drag: the object moves into the box at k = 0 in >= 60% of usable units
  windows differ   interquartile range of commit steps >= 3
  timing predicts  per-word rule (taken iff k <= commit + d) beats one global rule
                   (taken iff k <= K) by >= 5 points of leave-one-prompt-out accuracy,
                   and Spearman rho(commit, window close) > 0.3, bootstrap 95% CI above 0
  PASS = all three, WEAK = the first two, COLLAPSED = works but commits barely differ,
  NO_STEER = the change does not take even at the start.

Run from the project folder (milestone-1/exp1):  python scripts/pilot_editing.py            (all arms)
                               python scripts/pilot_editing.py --arm swap  (one arm)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_steering import (BOX_LEFT, BOX_RIGHT, Detector, in_box, lopo_accuracy,  # noqa: E402
                            natural_x, spearman_ci, target_slot, window_close)
from w2w.attention import AttentionStore, RecordingCrossAttnProcessor, commit_step  # noqa: E402
from w2w.paths import DATA, RESULTS  # noqa: E402
from w2w.steer import SteerController, guide_latents, steering_processors  # noqa: E402

STEPS, GUIDANCE, SEEDS = 30, 7.5, (0, 1)
KS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
DRAG_KS = (0, 3, 6, 9, 12, 15, 18, 21)
DRAG_ETAS, DRAG_G, DRAG_ITERS = (0.05, 0.1, 0.2), 8, 3
DRAG_PROMPTS, CAL_PROMPTS, CAL_NEED = 12, 6, 4
THRESHOLDS = (0.7, 0.8, 0.9)
COLS = ("commit_07", "commit_08", "commit_09")

# (source prompt, object word, replacement written into the prompt, detector class)
SWAPS = [
    ("a cat sitting on a wooden bench", "cat", "dog", "dog"),
    ("a dog lying on a couch next to a lamp", "dog", "cat", "cat"),
    ("a red umbrella on a sandy beach", "umbrella", "kite", "kite"),
    ("a horse standing in a green field", "horse", "cow", "cow"),
    ("a clock on a brick wall", "clock", "stop sign", "stop sign"),
    ("a bicycle parked next to a tree", "bicycle", "motorcycle", "motorcycle"),
    ("a bird perched on a wooden fence", "bird", "cat", "cat"),
    ("a cup of coffee on a wooden table", "cup", "bowl", "bowl"),
    ("a car parked on a quiet street", "car", "truck", "truck"),
    ("a giraffe standing near a tall tree", "giraffe", "elephant", "elephant"),
    ("a boat on a calm lake at sunset", "boat", "bird", "bird"),
    ("a vase of flowers on a windowsill", "vase", "cup", "cup"),
    ("a cow grazing in a field with mountains", "cow", "horse", "horse"),
    ("a laptop on a desk in a bright office", "laptop", "book", "book"),
    ("an apple and a banana on a plate", "banana", "orange", "orange"),
    ("a bench in a park with a dog", "dog", "cat", "cat"),
    ("a kitchen counter with a bottle", "bottle", "cup", "cup"),
    ("a city street with a bus", "bus", "truck", "truck"),
    ("a living room with a chair", "chair", "couch", "couch"),
    ("a grassy hill with a sheep", "sheep", "cow", "cow"),
]

# (source prompt, object word, edited prompt, kind)
DETAILS = [
    ("a cat sitting on a wooden bench", "cat", "a smiling cat sitting on a wooden bench", "expression"),
    ("a cat sitting on a wooden bench", "cat", "a cat wearing a red hat sitting on a wooden bench", "accessory"),
    ("a dog lying on a couch", "dog", "a dog wearing sunglasses lying on a couch", "accessory"),
    ("a dog lying on a couch", "dog", "a black dog lying on a couch", "colour"),
    ("a horse standing in a green field", "horse", "a white horse standing in a green field", "colour"),
    ("a horse standing in a green field", "horse", "a horse with a flower crown standing in a green field", "accessory"),
    ("a car parked on a quiet street", "car", "a red car parked on a quiet street", "colour"),
    ("a car parked on a quiet street", "car", "a car covered in snow parked on a quiet street", "texture"),
    ("a bird perched on a wooden fence", "bird", "a blue bird perched on a wooden fence", "colour"),
    ("a bird perched on a wooden fence", "bird", "a bird with an open beak perched on a wooden fence", "expression"),
    ("a cup of coffee on a wooden table", "cup", "a red cup of coffee on a wooden table", "colour"),
    ("a cup of coffee on a wooden table", "cup", "a cup of coffee with latte art on a wooden table", "texture"),
    ("a bicycle parked next to a tree", "bicycle", "a yellow bicycle parked next to a tree", "colour"),
    ("a bicycle parked next to a tree", "bicycle", "a bicycle with a basket parked next to a tree", "accessory"),
    ("a clock on a brick wall", "clock", "a golden clock on a brick wall", "colour"),
    ("a clock on a brick wall", "clock", "a clock with roman numerals on a brick wall", "texture"),
    ("a giraffe standing near a tall tree", "giraffe", "a baby giraffe standing near a tall tree", "shape"),
    ("a giraffe standing near a tall tree", "giraffe", "a giraffe eating leaves near a tall tree", "pose"),
    ("a boat on a calm lake at sunset", "boat", "a red boat on a calm lake at sunset", "colour"),
    ("a boat on a calm lake at sunset", "boat", "a boat with a white sail on a calm lake at sunset", "accessory"),
    ("a vase of flowers on a windowsill", "vase", "a blue vase of flowers on a windowsill", "colour"),
    ("a vase of flowers on a windowsill", "vase", "a vase of sunflowers on a windowsill", "texture"),
    ("a cow grazing in a field with mountains", "cow", "a brown cow grazing in a field with mountains", "colour"),
    ("a cow grazing in a field with mountains", "cow", "a cow wearing a bell grazing in a field with mountains", "accessory"),
]

DRAG = [(p, w) for p, w, _, _ in SWAPS[:DRAG_PROMPTS]]


def swap_prompt(prompt, word, new):
    out, n = re.subn(rf"\b{re.escape(word)}\b", new, prompt, count=1)
    assert n == 1, f"'{word}' not found in '{prompt}'"
    return out


# ------------------------------------------------------------------ generation

class Gen:
    """One pipeline, three processor sets: stock, our attention path, our path + recorder."""

    def __init__(self, pipe, device):
        self.pipe, self.device = pipe, device
        self.ctrl = SteerController()
        self.store = AttentionStore()
        self.original = dict(pipe.unet.attn_processors)
        self.ours = steering_processors(self.original, self.ctrl)
        self.rec = {n: (RecordingCrossAttnProcessor(self.store, inner=p) if "attn2" in n else p)
                    for n, p in self.ours.items()}
        self.mode = None
        self.null = None

    def _set(self, mode):
        if mode != self.mode:
            src = {"stock": self.original, "ours": self.ours, "record": self.rec}[mode]
            self.pipe.unet.set_attn_processor(dict(src))
            self.mode = mode

    @torch.no_grad()
    def embed(self, prompt):
        pe, ne = self.pipe.encode_prompt(prompt, self.device, 1, True)
        return pe, ne

    def noise(self, seed):
        g = torch.Generator(self.device).manual_seed(seed)
        n = self.pipe.unet.config.sample_size
        return torch.randn((1, 4, n, n), generator=g, device=self.device, dtype=self.pipe.unet.dtype)

    def run(self, seed, src, tgt=None, k=None, record=False, mode=None, drag=None):
        """src/tgt: (pe, ne) pairs. Switch to tgt at step k (k = 0: tgt from the start).
        drag: dict(slot, box, eta) to apply backward guidance at steps k .. k + G - 1."""
        self._set(mode or ("record" if record else "ours"))
        self.store.reset()
        pipe = self.pipe
        pipe.scheduler.set_timesteps(STEPS, device=self.device)
        ts = pipe.scheduler.timesteps
        lat = self.noise(seed)
        use = tgt if (tgt is not None and k == 0) else src
        losses = []

        if drag is not None and k == 0:
            lat, l = guide_latents(pipe.unet, self.ctrl, lat, ts[0], src[0], drag["slot"], drag["box"],
                                   drag["eta"], DRAG_ITERS)
            losses.append(l)

        def cb(p, i, t, kw):
            if record:
                self.store.end_step()
            nxt = i + 1
            if tgt is not None and k is not None and k > 0 and nxt == k:
                kw["prompt_embeds"] = torch.cat([tgt[1], tgt[0]])
            if drag is not None and nxt < len(ts) and k <= nxt < k + DRAG_G:
                kw["latents"], l = guide_latents(p.unet, self.ctrl, kw["latents"], ts[nxt],
                                                 kw["prompt_embeds"][-1:], drag["slot"], drag["box"],
                                                 drag["eta"], DRAG_ITERS)
                losses.append(l)
            return kw

        out = pipe(prompt_embeds=use[0], negative_prompt_embeds=use[1], latents=lat,
                   num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                   callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"])
        nsfw = bool(out.nsfw_content_detected[0]) if getattr(out, "nsfw_content_detected", None) else False
        maps = self.store.stacked() if record else None
        return out.images[0], nsfw, maps, losses


class Clip:
    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.device = device

    @torch.no_grad()
    def gap(self, img, edited, original):
        """cos(image, edited caption) - cos(image, original caption)."""
        x = self.proc(text=[edited, original], images=img, return_tensors="pt", padding=True).to(self.device)
        out = self.model(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                         pixel_values=x["pixel_values"].to(self.model.dtype))
        i = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        t = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        s = (i @ t.T)[0].float()
        return float(s[0] - s[1])


# ------------------------------------------------------------------ analysis

def analyse(df, units, ks, works_key, works_bar):
    ok_u, ok = units[units.usable], df[df.usable]
    res = {"units": int(len(units)), "usable_units": int(len(ok_u)),
           "success_by_k": {int(k): round(float(v), 4) for k, v in ok.groupby("k").success.mean().items()}}
    works_rate = float(ok[ok.k == works_key].success.mean()) if len(ok) else float("nan")
    res[f"success_at_k{works_key}"] = round(works_rate, 4)
    c = ok_u.commit_08.dropna()
    if len(c):
        q1, q3 = np.percentile(c, [25, 75])
        res["commit_spread_08"] = {"min": int(c.min()), "q1": float(q1), "median": float(c.median()),
                                   "q3": float(q3), "max": int(c.max()), "iqr": float(q3 - q1)}
    res["window_close"] = {"median": float(ok_u.window_close.median()) if len(ok_u) else None,
                           "censored": int(ok_u.close_censored.sum()) if len(ok_u) else 0}
    if len(ok_u):
        res["close_minus_commit_08_median"] = float((ok_u.window_close - ok_u.commit_08).median())
    acc, corr = {}, {}
    for th, col in zip(THRESHOLDS, COLS):
        sub = ok.dropna(subset=[col])
        if sub.prompt_id.nunique() >= 3:
            w, g = lopo_accuracy(sub, col)
            acc[str(th)] = {"per_word": round(w, 4), "global": round(g, 4), "gain": round(w - g, 4)}
        su = ok_u[ok_u.k0_success].dropna(subset=[col])
        if len(su) >= 5 and su[col].nunique() > 1 and su.window_close.nunique() > 1:
            r, lo, hi = spearman_ci(su[col], su.window_close)
            corr[str(th)] = {"rho": round(r, 3), "ci95": [round(lo, 3), round(hi, 3)], "n": int(len(su)),
                             "censored": int(su.close_censored.sum())}
    res["lopo_accuracy"], res["spearman_commit_vs_close"] = acc, corr
    works = works_rate >= works_bar
    differ = res.get("commit_spread_08", {}).get("iqr", 0) >= 3
    a8, c8 = acc.get("0.8", {}), corr.get("0.8", {})
    predicts = bool(a8 and c8 and a8["gain"] >= 0.05 and c8["rho"] > 0.3 and c8["ci95"][0] > 0)
    res["criteria"] = {"change_works": works, "windows_differ": differ, "timing_predicts": predicts}
    res["verdict"] = "NO_STEER" if not works else "COLLAPSED" if not differ else "PASS" if predicts else "WEAK"
    return res


def plot(df, name, out_dir, ylabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = df[df.usable]
    if ok.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    a = ok.groupby("k").success.mean()
    ax[0].plot(a.index, a.values, marker="o", color="#2b6cb0")
    ax[0].set_xlabel("step at which the change starts (k)")
    ax[0].set_ylabel(ylabel)
    ax[0].set_title("Against the absolute step", fontsize=10)
    rel = ok.k - ok.commit_08
    b = ok.assign(rel=pd.cut(rel, np.arange(-30, 34, 3))).groupby("rel", observed=True).success.agg(["mean", "size"])
    xs = [iv.mid for iv in b.index]
    ax[1].plot(xs, b["mean"].values, marker="o", color="#c05621")
    for x, (m, n) in zip(xs, b.values):
        ax[1].annotate(str(int(n)), (x, m), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax[1].axvline(0, color="grey", lw=0.8, ls="--")
    ax[1].set_xlabel("k minus the object word's commit step")
    ax[1].set_title("Relative to the word's own commit", fontsize=10)
    for x in ax:
        x.set_ylim(-0.05, 1.05)
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_success_vs_start.png", dpi=150)
    plt.close(fig)


def strip(img_dir, names, labels, out_path, title):
    from PIL import Image, ImageDraw
    th = 128
    sheet = Image.new("RGB", (len(names) * th, th + 34), "white")
    d = ImageDraw.Draw(sheet)
    d.text((2, 2), title, fill="black")
    for c, (nm, lab) in enumerate(zip(names, labels)):
        p = img_dir / f"{nm}.png"
        if p.exists():
            sheet.paste(Image.open(p).convert("RGB").resize((th, th)), (c * th, 34))
        d.text((c * th + 2, 18), lab, fill="black")
    sheet.save(out_path)


def commits(maps, slot):
    return {col: commit_step(maps, slot, threshold=th) for th, col in zip(THRESHOLDS, COLS)}


# ------------------------------------------------------------------ arms

def arm_swap(gen, detect, tok, out_dir, img_dir, items, log):
    rows, units = [], []
    for pid, (src_p, word, new, cls) in enumerate(items):
        tgt_p = swap_prompt(src_p, word, new)
        src, tgt = gen.embed(src_p), gen.embed(tgt_p)
        slot = target_slot(tok, src_p, word)
        for seed in SEEDS:
            img, nsfw, maps, _ = gen.run(seed, src, record=True)
            img.save(img_dir / f"swap_{pid:02d}_{seed}_base.png")
            c = commits(maps, slot)
            base_has_new = bool(detect(img, cls))
            succ = {}
            for k in KS:
                im, nf, _, _ = gen.run(seed, src, tgt, k=k)
                im.save(img_dir / f"swap_{pid:02d}_{seed}_k{k:02d}.png")
                succ[k] = bool(detect(im, cls)) and not nf
                rows.append(dict(prompt_id=pid, seed=seed, k=k, success=succ[k], old_present=bool(detect(im, word)),
                                 nsfw=nf, **c))
            usable = (not base_has_new) and succ[0] and c["commit_08"] is not None and not nsfw
            for r in rows[-len(KS):]:
                r["usable"] = usable
            close, cens = window_close(succ)
            units.append(dict(prompt_id=pid, seed=seed, source=src_p, edited=tgt_p, usable=usable,
                              base_has_new=base_has_new, k0_success=succ[0], window_close=close,
                              close_censored=cens, **c))
            log(f"swap {pid:02d} seed {seed} {word}->{new}: commit {c['commit_08']}, "
                f"new object at k = {[k for k in KS if succ[k]]}")
    return pd.DataFrame(rows), pd.DataFrame(units)


def arm_detail(gen, clip, tok, out_dir, img_dir, items, log):
    rows, units = [], []
    pid_of = {}
    for uid, (src_p, word, tgt_p, kind) in enumerate(items):
        pid = pid_of.setdefault(src_p, len(pid_of))       # prompts sharing a source share a fold
        src, tgt = gen.embed(src_p), gen.embed(tgt_p)
        slot = target_slot(tok, src_p, word)
        for seed in SEEDS:
            img, nsfw, maps, _ = gen.run(seed, src, record=True)
            img.save(img_dir / f"detail_{uid:02d}_{seed}_base.png")
            c = commits(maps, slot)
            s_src = clip.gap(img, tgt_p, src_p)
            imgs, gaps = {}, {}
            for k in KS:
                im, nf, _, _ = gen.run(seed, src, tgt, k=k)
                im.save(img_dir / f"detail_{uid:02d}_{seed}_k{k:02d}.png")
                imgs[k], gaps[k] = im, (clip.gap(im, tgt_p, src_p) if not nf else float("nan"))
            span = gaps[0] - s_src
            renderable = bool(np.isfinite(span) and span >= 0.01)
            base_arr = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)
            full_d = float(np.abs(np.asarray(imgs[0].convert("L").resize((64, 64)), dtype=np.float32) - base_arr).mean())
            succ = {}
            for k in KS:
                e = (gaps[k] - s_src) / span if renderable and np.isfinite(gaps[k]) else float("nan")
                succ[k] = bool(np.isfinite(e) and e >= 0.5)
                d = float(np.abs(np.asarray(imgs[k].convert("L").resize((64, 64)), dtype=np.float32) - base_arr).mean())
                rows.append(dict(prompt_id=pid, unit=uid, seed=seed, kind=kind, k=k, success=succ[k],
                                 edit_score=round(e, 4) if np.isfinite(e) else None,
                                 layout_change=round(d / full_d, 4) if full_d > 0 else None, **c))
            usable = renderable and c["commit_08"] is not None and not nsfw
            for r in rows[-len(KS):]:
                r["usable"] = usable
            close, cens = window_close(succ)
            units.append(dict(prompt_id=pid, unit=uid, seed=seed, source=src_p, edited=tgt_p, kind=kind,
                              usable=usable, renderable=renderable, clip_span=round(span, 4),
                              k0_success=succ[0], window_close=close, close_censored=cens, **c))
            log(f"detail {uid:02d} seed {seed} [{kind}] '{tgt_p}': commit {c['commit_08']}, span {span:.3f}, "
                f"taken at k = {[k for k in KS if succ[k]]}")
    return pd.DataFrame(rows), pd.DataFrame(units)


def arm_drag(gen, detect, tok, out_dir, img_dir, items, log):
    base = {}

    def baseline(pid, seed):
        if (pid, seed) not in base:
            p, w = items[pid]
            src = gen.embed(p)
            img, nsfw, maps, _ = gen.run(seed, src, record=True)
            img.save(img_dir / f"drag_{pid:02d}_{seed}_base.png")
            slot = target_slot(tok, p, w)
            dets = detect(img, w)
            nx, _ = natural_x(dets, maps, slot)
            box = BOX_LEFT if nx >= 0.5 else BOX_RIGHT
            base[(pid, seed)] = dict(src=src, slot=slot, dets=dets, box=box, nsfw=nsfw, c=commits(maps, slot))
        return base[(pid, seed)]

    cal, eta = [], None
    n_cal = min(CAL_PROMPTS, len(items))
    for e in DRAG_ETAS:
        hits = 0
        for pid in range(n_cal):
            b = baseline(pid, 0)
            im, _, _, _ = gen.run(0, b["src"], k=0, drag=dict(slot=b["slot"], box=b["box"], eta=e))
            im.save(img_dir / f"drag_cal_{pid:02d}_eta{e:g}.png")
            hits += in_box(detect(im, items[pid][1]), b["box"])
        cal.append({"eta": e, "moved": hits, "of": n_cal})
        log(f"drag calibration: eta {e}: {hits} of {n_cal} moved")
        if hits >= min(CAL_NEED, n_cal):
            eta = e
            break
    if eta is None:
        return None, None, {"calibration": cal, "eta": None, "verdict": "NO_STEER",
                            "note": "No step size moved the object at k = 0 in enough calibration prompts."}

    rows, units = [], []
    for pid, (p, w) in enumerate(items):
        for seed in SEEDS:
            b = baseline(pid, seed)
            c = b["c"]
            base_in = in_box(b["dets"], b["box"])
            succ, last_loss = {}, None
            for k in DRAG_KS:
                im, nf, _, losses = gen.run(seed, b["src"], k=k, drag=dict(slot=b["slot"], box=b["box"], eta=eta))
                im.save(img_dir / f"drag_{pid:02d}_{seed}_k{k:02d}.png")
                succ[k] = in_box(detect(im, w), b["box"]) and not nf
                rows.append(dict(prompt_id=pid, seed=seed, k=k, success=succ[k], nsfw=nf,
                                 final_loss=losses[-1] if losses else None, **c))
            usable = (not base_in) and c["commit_08"] is not None and not b["nsfw"]
            for r in rows[-len(DRAG_KS):]:
                r["usable"] = usable
            close, cens = window_close(succ)
            units.append(dict(prompt_id=pid, seed=seed, prompt=p, target=w, usable=usable, box=json.dumps(b["box"]),
                              baseline_in_box=base_in, k0_success=succ[0], window_close=close,
                              close_censored=cens, **c))
            log(f"drag {pid:02d} seed {seed} '{w}': commit {c['commit_08']}, moved at k = "
                f"{[k for k in DRAG_KS if succ[k]]}")
    return pd.DataFrame(rows), pd.DataFrame(units), {"calibration": cal, "eta": eta}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["all", "swap", "detail", "drag"], default="all")
    ap.add_argument("--model", default=None, help="testing only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fake-judges", action="store_true", help="testing only")
    ap.add_argument("--limit", type=int, default=None, help="testing only")
    ap.add_argument("--out", default="m2_pilot_editing")
    args = ap.parse_args()

    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from w2w.generate import SD_ID
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(args.model or SD_ID, dtype=dtype).to(args.device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.requires_grad_(False)       # gradients are taken with respect to the latents only
    tok = pipe.tokenizer
    out_dir, img_dir = RESULTS / args.out, DATA / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)

    if args.fake_judges:
        rng = np.random.default_rng(0)
        detect = lambda img, cls: [(0.9, float(rng.random()), float(rng.random()))] if rng.random() < 0.7 else []

        class FakeClip:
            def gap(self, img, a, b):
                return float(rng.normal(0.02, 0.02))
        clip = FakeClip()
    else:
        detect = Detector(args.device)
        clip = None

    gen = Gen(pipe, args.device)
    lim = (lambda xs: xs[: args.limit]) if args.limit else (lambda xs: xs)

    # Parity: our attention path against the stock processors, same noise and embeddings.
    src = gen.embed(SWAPS[0][0])
    stock, _, _, _ = gen.run(0, src, mode="stock")
    ours, _, _, _ = gen.run(0, src)
    parity = float(np.abs(np.asarray(stock, np.float32) - np.asarray(ours, np.float32)).mean())
    log(f"parity: mean |stock - our path| = {parity:.4f} (0-255 scale)")
    overall = {"settings": {"steps": STEPS, "guidance": GUIDANCE, "scheduler": "DDIM", "seeds": list(SEEDS),
                            "ks": list(KS), "drag_ks": list(DRAG_KS), "drag_etas": list(DRAG_ETAS),
                            "drag_guided_steps": DRAG_G, "drag_iters": DRAG_ITERS},
               "parity_mean_abs_pixel_diff": parity, "arms": {}}

    def finish(name, df, units, extra, works_key, works_bar, ylabel):
        res = dict(extra or {})
        if df is not None:
            df.to_csv(out_dir / f"{name}_runs.csv", index=False, lineterminator="\n")
            units.to_csv(out_dir / f"{name}_units.csv", index=False, lineterminator="\n")
            res.update(analyse(df, units, KS, works_key, works_bar))
            plot(df, name, out_dir, ylabel)
        res["minutes"] = round((time.time() - t0) / 60, 1)
        overall["arms"][name] = res
        json.dump(overall, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2)
        log(f"{name}: verdict {res.get('verdict')}")

    if args.arm in ("all", "swap"):
        df, units = arm_swap(gen, detect, tok, out_dir, img_dir, lim(SWAPS), log)
        finish("swap", df, units, None, 0, 0.8, "new object appears")
        u = units[units.usable]
        if len(u):
            r = u.sort_values("commit_08").iloc[len(u) // 2]
            strip(img_dir, [f"swap_{r.prompt_id:02d}_{r.seed}_base"] + [f"swap_{r.prompt_id:02d}_{r.seed}_k{k:02d}" for k in KS],
                  ["original"] + [f"k={k}" for k in KS], out_dir / "swap_example.png",
                  f"'{r.source}' -> '{r.edited}', commit {r.commit_08}")

    if args.arm in ("all", "detail"):
        if clip is None:
            clip = Clip(args.device)
        df, units = arm_detail(gen, clip, tok, out_dir, img_dir, lim(DETAILS), log)
        finish("detail", df, units, None, 9, 0.5, "detail edit taken")
        for uid in (0, 1):                                  # the smiling cat and the red hat
            if uid < len(units) // len(SEEDS):
                r = units[units.unit == uid].iloc[0]
                strip(img_dir, [f"detail_{uid:02d}_0_base"] + [f"detail_{uid:02d}_0_k{k:02d}" for k in KS],
                      ["original"] + [f"k={k}" for k in KS], out_dir / f"detail_example_{uid}.png",
                      f"'{r.edited}', commit {r.commit_08}")

    if args.arm in ("all", "drag"):
        try:
            df, units, extra = arm_drag(gen, detect, tok, out_dir, img_dir, lim(DRAG), log)
            finish("drag", df, units, extra, 0, 0.6, "object moved into the box")
        except torch.OutOfMemoryError as e:
            finish("drag", None, None, {"verdict": "OOM", "note": str(e)[:500]}, 0, 0.6, "")

    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
