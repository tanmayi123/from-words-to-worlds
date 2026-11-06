"""Pilot 3: when is a missing object lost, and does a mask buy time for edits?

Arm "loss" (the moment an object is lost).
  Paint COCO captions from the frozen eval_2000 split (seed 0, DDIM 30 steps, guidance 7.5)
  until 30 failures are found. A failure is an object that the caption names, that the
  real photo's COCO annotations contain, and that the detector cannot find in the painting
  at score >= 0.3 ("person" and "dining table" are skipped: too frequent and too loosely
  named). For each failure, go back to after step k for k in LOSS_KS and finish the
  painting R = 8 times with fresh randomness (DDIM eta 1, a new noise seed per repaint).
  p(k) = share of the 8 repaints in which the detector finds the object at score >= 0.5.
  The moment of loss L is the earliest k from which p stays at or below 1/8.

Arm "window" (does a mask buy time?).
  The 12 edits below (4 object swaps, 8 added details), seeds 0 and 1. Each edit is made
  from after step k for k in WINDOW_KS, twice: plain (the whole picture follows the new
  sentence) and masked (Prompt-to-Prompt local blending: outside the edited words' region
  every step is copied from the original painting). An edit counts as taken when its CLIP
  edit score e >= 0.6 (stricter than task 10's 0.5, which proved generous) and, for swaps,
  the detector finds the new object at >= 0.5. The scene counts as kept when at least 70%
  of the original survives (pixel measure, as in task 11). For each unit and condition,
  the latest usable k is the largest k with the edit taken and the scene kept.

Pre-registered reading:
  loss    MOMENTS     among failures that are possible at all (p(0) >= 3/8), the median L
                      is >= 3 and the interquartile range of L is >= 3: failures have their
                      own moments, spread over the painting
          EARLY       possible failures, but median L < 3: objects are lost at the very start
          PROMPT      more than half the failures have p(0) <= 1/8: the sentence, not the
                      timing, is the problem
          also: early warning = Spearman rho(object word's early attention share, L) > 0.3
          with a bootstrap 95% interval above 0 (needs >= 10 possible failures)
  window  MASK_HELPS  median gain in the latest usable k (masked minus plain) >= 3 steps
          NO_GAIN     otherwise

Run from the project folder (milestone-1/exp1):  python scripts/pilot_moments.py [--arm loss|window|all]
"""
import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_editing import Clip  # noqa: E402
from pilot_steering import Detector, spearman_ci  # noqa: E402
from w2w.attention import attention_share, commit_step  # noqa: E402
from w2w.live import sentence_words  # noqa: E402
from w2w.paths import COCO, DATA, RESULTS  # noqa: E402
from w2w.rewind import Painter  # noqa: E402

STEPS, GUIDANCE = 30, 7.5
LOSS_KS, R, N_FAIL, SCAN = (0, 3, 6, 9, 12, 15, 18), 8, 30, 300
FAIL_TH, FOUND_TH = 0.3, 0.5
WINDOW_KS, SEEDS = (0, 3, 6, 9, 12, 15, 18), (0, 1)
EDIT_TH, SCENE_TH, BLEND_TH = 0.6, 0.7, 0.3
SKIP = {"person", "dining table"}

SYNONYMS = {"bike": "bicycle", "bikes": "bicycle", "motorbike": "motorcycle", "motorbikes": "motorcycle",
            "plane": "airplane", "planes": "airplane", "jet": "airplane", "jets": "airplane",
            "sofa": "couch", "television": "tv", "phone": "cell phone", "cellphone": "cell phone",
            "doughnut": "donut", "doughnuts": "donut", "teddy": "teddy bear", "hydrant": "fire hydrant",
            "puppy": "dog", "kitten": "cat", "kitty": "cat", "racket": "tennis racket", "racquet": "tennis racket",
            "skis": "skis", "glass": "wine glass", "ball": "sports ball", "plant": "potted plant"}

# (id, original, object word, edited, kind, detector class for swaps)
EDITS = [
    ("catdog", "a cat sitting on a wooden bench", "cat", "a dog sitting on a wooden bench", "swap", "dog"),
    ("bicycle", "a bicycle parked next to a tree", "bicycle", "a motorcycle parked next to a tree", "swap", "motorcycle"),
    ("horsecow", "a horse standing in a green field", "horse", "a cow standing in a green field", "swap", "cow"),
    ("cartruck", "a car parked on a quiet street", "car", "a truck parked on a quiet street", "swap", "truck"),
    ("hat", "a cat sitting on a wooden bench", "cat", "a cat wearing a red hat sitting on a wooden bench", "detail", None),
    ("sunglasses", "a dog lying on a couch", "dog", "a dog wearing sunglasses lying on a couch", "detail", None),
    ("snow", "a car parked on a quiet street", "car", "a car covered in snow parked on a quiet street", "detail", None),
    ("redcar", "a car parked on a quiet street", "car", "a red car parked on a quiet street", "detail", None),
    ("bluebird", "a bird perched on a wooden fence", "bird", "a blue bird perched on a wooden fence", "detail", None),
    ("crown", "a horse standing in a green field", "horse", "a horse with a flower crown standing in a green field", "detail", None),
    ("whitehorse", "a horse standing in a green field", "horse", "a white horse standing in a green field", "detail", None),
    ("latte", "a cup of coffee on a wooden table", "cup", "a cup of coffee with latte art on a wooden table", "detail", None),
]


def grey64(img):
    return np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)


def named_objects(caption, classes):
    """COCO classes the caption names, with the caption word(s) that name them, in caption order."""
    words = [w.strip(".,!?;:'\"").lower() for w in caption.split()]
    out, seen = [], set()
    for i, w in enumerate(words):
        two = " ".join(words[i:i + 2])
        for cand, span in ((two, 2), (w, 1)):
            cls = SYNONYMS.get(cand, cand)
            if cls not in classes and cand.endswith("es") and cand[:-2] in classes:
                cls = cand[:-2]
            if cls not in classes and cand.endswith("s") and cand[:-1] in classes:
                cls = cand[:-1]
            if cls in classes and cls not in seen and cls not in SKIP:
                out.append((cls, words[i:i + span]))
                seen.add(cls)
                break
    return out


def word_slots(tok, caption, target_words):
    """Token slots of the caption words in `target_words` (first occurrence of the sequence)."""
    sw = sentence_words(tok, caption)
    texts = [w["text"].lower() for w in sw]
    n = len(target_words)
    for i in range(len(texts) - n + 1):
        if texts[i:i + n] == target_words:
            return [s for w in sw[i:i + n] for s in w["slots"]]
    return []


def edit_slots(tok, src, dst, obj):
    """Slots of the changed words plus the object word, in the edited and in the original sentence."""
    a, b = sentence_words(tok, src), sentence_words(tok, dst)
    ta, tb = [w["text"].lower() for w in a], [w["text"].lower() for w in b]
    new, old = set(), set()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb).get_opcodes():
        if tag != "equal":
            new.update(s for w in b[j1:j2] for s in w["slots"])
            old.update(s for w in a[i1:i2] for s in w["slots"])
    new.update(s for w in b if w["text"].lower() == obj for s in w["slots"])
    old.update(s for w in a if w["text"].lower() == obj for s in w["slots"])
    return sorted(new), sorted(old)


def plot_loss(units, curves, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for c in curves:
        ax[0].plot(LOSS_KS, c, color="#c05621", alpha=0.25, lw=1)
    if curves:
        ax[0].plot(LOSS_KS, np.median(np.array(curves), 0), color="#c05621", lw=2.5, marker="o", label="median")
        ax[0].legend(frameon=False)
    ax[0].set_xlabel("go back to after step k, then finish 8 times")
    ax[0].set_ylabel("share of repaints with the object")
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].set_title("Survival of the missing object", fontsize=10)
    pos = units[units.possible]
    if len(pos):
        vals = pos.loss_step.fillna(LOSS_KS[-1] + 3)
        ax[1].hist(vals, bins=np.arange(-1.5, LOSS_KS[-1] + 6, 3), color="#2b6cb0")
    ax[1].set_xlabel("moment of loss (step); rightmost bar = not lost by step 18")
    ax[1].set_ylabel("failures")
    ax[1].set_title("When objects are lost (possible failures)", fontsize=10)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_survival.png", dpi=150)
    plt.close(fig)


def plot_window(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for cond, col in (("plain", "#2b6cb0"), ("masked", "#c05621")):
        d = df[df.condition == cond].groupby("k")
        ax[0].plot(d.taken.mean().index, d.taken.mean().values, marker="o", color=col, label=cond)
        ax[1].plot(d.scene.median().index, d.scene.median().values, marker="o", color=col, label=cond)
    ax[0].set_ylabel("edit taken")
    ax[1].set_ylabel("scene kept (median)")
    for a in ax:
        a.set_xlabel("edit starts after step k")
        a.set_ylim(-0.05, 1.05)
        a.legend(frameon=False)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "window_plain_vs_masked.png", dpi=150)
    plt.close(fig)


def strip(paths, labels, out_path, title):
    from PIL import Image, ImageDraw
    th = 128
    sheet = Image.new("RGB", (len(paths) * th, th + 34), "white")
    d = ImageDraw.Draw(sheet)
    d.text((2, 2), title, fill="black")
    for c, (p, lab) in enumerate(zip(paths, labels)):
        if p.exists():
            sheet.paste(Image.open(p).convert("RGB").resize((th, th)), (c * th, 34))
        d.text((c * th + 2, 18), lab, fill="black")
    sheet.save(out_path)


def arm_loss(painter, detect, out_dir, img_dir, log):
    from w2w.data import load_manifest
    ev = pd.read_csv(RESULTS / "m1_data" / "eval_2000.csv")
    man = load_manifest()
    cats = dict(zip(man.image_id, man.categories))
    classes = set(detect.names) - {"N/A", "__background__"}
    tok = painter.pipe.tokenizer
    fails, scanned = [], 0
    for r in ev.itertuples():
        if len(fails) >= N_FAIL or scanned >= SCAN:
            break
        named = [(c, w) for c, w in named_objects(r.caption, classes) if c in set(cats.get(r.image_id, []))]
        if not named:
            continue
        scanned += 1
        base = painter.paint(r.caption, 0, previews=False, record=True)
        for cls, words in named:
            if detect(base.final, cls, FAIL_TH):
                continue
            slots = word_slots(tok, r.caption, words)
            if not slots:
                continue
            content = [s for w in sentence_words(tok, r.caption) if w["content"] for s in w["slots"]]
            share = attention_share(base.maps, content)
            idx = [content.index(s) for s in slots if s in content]
            early = float(share[:3, idx].sum(1).mean()) if idx else float("nan")
            fails.append(dict(fid=len(fails), image_id=int(r.image_id), caption=r.caption, cls=cls,
                              words=" ".join(words), slot=slots[0], early_share=round(early, 4),
                              commit_08=commit_step(base.maps, slots[0], threshold=0.8), base=base))
            base.final.save(img_dir / f"loss_{len(fails) - 1:02d}_base.png")
            log(f"failure {len(fails) - 1:02d}: '{r.caption}' is missing '{cls}' (scanned {scanned})")
            break
    log(f"scan: {scanned} captions painted, {len(fails)} failures found")

    rows, units, curves = [], [], []
    for f in fails:
        p = []
        for k in LOSS_KS:
            hits = 0
            for rep in range(R):
                tl = painter.paint(f["caption"], 0, branch_from=f["base"], at=k, eta=1.0,
                                   noise_seed=1000 * k + rep, previews=False, record=False)
                found = bool(detect(tl.final, f["cls"], FOUND_TH))
                hits += found
                if rep < 2:
                    tl.final.save(img_dir / f"loss_{f['fid']:02d}_k{k:02d}_r{rep}.png")
                rows.append(dict(fid=f["fid"], k=k, rep=rep, found=found))
            p.append(hits / R)
        loss = None
        for i, k in enumerate(LOSS_KS):
            if all(x <= 1 / R for x in p[i:]):
                loss = k
                break
        curves.append(p)
        units.append(dict(fid=f["fid"], image_id=f["image_id"], caption=f["caption"], object=f["cls"],
                          words=f["words"], early_share=f["early_share"], commit_08=f["commit_08"],
                          p0=p[0], possible=p[0] >= 3 / R, loss_step=loss,
                          **{f"p_k{k:02d}": x for k, x in zip(LOSS_KS, p)}))
        log(f"failure {f['fid']:02d} '{f['cls']}': p = {p}, lost at {loss}")
        f.pop("base")
    df, units = pd.DataFrame(rows), pd.DataFrame(units)
    df.to_csv(out_dir / "loss_repaints.csv", index=False, lineterminator="\n")
    units.to_csv(out_dir / "loss_failures.csv", index=False, lineterminator="\n")

    res = {"captions_scanned": scanned, "failures": int(len(units))}
    if len(units):
        pos = units[units.possible]
        res["possible_failures"] = int(len(pos))
        res["prompt_failures_p0_le_1_of_8"] = int((units.p0 <= 1 / R).sum())
        res["median_survival_by_k"] = {int(k): float(np.median([c[i] for c in curves])) for i, k in enumerate(LOSS_KS)}
        if len(pos):
            L = pos.loss_step.fillna(LOSS_KS[-1] + 3)
            q1, q3 = np.percentile(L, [25, 75])
            res["loss_step_possible"] = {"median": float(L.median()), "q1": float(q1), "q3": float(q3),
                                         "iqr": float(q3 - q1), "not_lost_by_18": int(pos.loss_step.isna().sum()),
                                         "lost_by_6": float((L <= 6).mean()), "lost_by_9": float((L <= 9).mean())}
            sub = pos.dropna(subset=["early_share"])
            if len(sub) >= 10 and sub.early_share.nunique() > 1 and L.nunique() > 1:
                rho, lo, hi = spearman_ci(sub.early_share, L.loc[sub.index])
                res["early_warning_spearman"] = {"rho": round(rho, 3), "ci95": [round(lo, 3), round(hi, 3)], "n": int(len(sub))}
        ls = res.get("loss_step_possible", {})
        if res["prompt_failures_p0_le_1_of_8"] > len(units) / 2:
            verdict = "PROMPT"
        elif ls and ls["median"] >= 3 and ls["iqr"] >= 3:
            verdict = "MOMENTS"
        elif ls:
            verdict = "EARLY"
        else:
            verdict = "PROMPT"
        ew = res.get("early_warning_spearman")
        res["early_warning_works"] = bool(ew and ew["rho"] > 0.3 and ew["ci95"][0] > 0)
        res["verdict"] = verdict
        plot_loss(units, curves, out_dir)
        for fid in units[units.possible].fid.head(2):
            strip([img_dir / f"loss_{fid:02d}_base.png"] + [img_dir / f"loss_{fid:02d}_k{k:02d}_r0.png" for k in LOSS_KS],
                  ["original"] + [f"after {k}" for k in LOSS_KS], out_dir / f"loss_example_{fid:02d}.png",
                  f"'{units.loc[units.fid == fid, 'caption'].item()}' missing '{units.loc[units.fid == fid, 'object'].item()}'")
    return res


def arm_window(painter, detect, clip, out_dir, img_dir, log):
    tok = painter.pipe.tokenizer
    rows, units = [], []
    for eid, src, obj, dst, kind, cls in EDITS:
        new, old = edit_slots(tok, src, dst, obj)
        for seed in SEEDS:
            orig = painter.paint(src, seed, previews=False, record=True)
            g0 = grey64(orig.final)
            s0 = clip.gap(orig.final, dst, src)
            orig.final.save(img_dir / f"win_{eid}_{seed}_orig.png")
            res = {}
            for cond in ("plain", "masked"):
                for k in WINDOW_KS:
                    blend = {"orig": orig, "slots_new": new, "slots_orig": old, "th": BLEND_TH} if cond == "masked" else None
                    tl = painter.paint(dst, seed, branch_from=orig, at=k, previews=False,
                                       record=cond == "masked", blend=blend)
                    tl.final.save(img_dir / f"win_{eid}_{seed}_{cond}_k{k:02d}.png")
                    res[(cond, k)] = tl.final
            full = res[("plain", 0)]
            s_full = clip.gap(full, dst, src)
            d_full = float(np.abs(grey64(full) - g0).mean())
            latest = {}
            for cond in ("plain", "masked"):
                best = -1
                for k in WINDOW_KS:
                    img = res[(cond, k)]
                    e = (clip.gap(img, dst, src) - s0) / (s_full - s0) if s_full - s0 > 1e-6 else float("nan")
                    scene = 1 - float(np.abs(grey64(img) - g0).mean()) / d_full if d_full > 0 else float("nan")
                    taken = bool(np.isfinite(e) and e >= EDIT_TH and (cls is None or detect(img, cls, FOUND_TH)))
                    ok = taken and scene >= SCENE_TH
                    if ok:
                        best = k
                    rows.append(dict(edit=eid, kind=kind, seed=seed, condition=cond, k=k, edit_score=round(e, 3),
                                     scene=round(scene, 3), taken=taken, usable=ok))
                latest[cond] = best
            units.append(dict(edit=eid, kind=kind, seed=seed, renderable=bool(s_full - s0 > 0.01),
                              latest_plain=latest["plain"], latest_masked=latest["masked"],
                              gain=latest["masked"] - latest["plain"]))
            log(f"window {eid} seed {seed}: latest usable start plain {latest['plain']}, masked {latest['masked']}")
    df, units = pd.DataFrame(rows), pd.DataFrame(units)
    df.to_csv(out_dir / "window_runs.csv", index=False, lineterminator="\n")
    units.to_csv(out_dir / "window_units.csv", index=False, lineterminator="\n")
    plot_window(df, out_dir)
    for eid in ("hat", "sunglasses"):
        for cond in ("plain", "masked"):
            strip([img_dir / f"win_{eid}_0_orig.png"] + [img_dir / f"win_{eid}_0_{cond}_k{k:02d}.png" for k in WINDOW_KS],
                  ["original"] + [f"after {k}" for k in WINDOW_KS], out_dir / f"window_{eid}_{cond}.png", f"{eid}, {cond}")
    u = units[units.renderable & ((units.latest_plain >= 0) | (units.latest_masked >= 0))]
    res = {"units": int(len(units)), "compared_units": int(len(u)),
           "taken_by_k": {c: {int(k): round(float(v), 3) for k, v in df[df.condition == c].groupby("k").taken.mean().items()}
                          for c in ("plain", "masked")},
           "scene_median_by_k": {c: {int(k): round(float(v), 3) for k, v in df[df.condition == c].groupby("k").scene.median().items()}
                                 for c in ("plain", "masked")}}
    if len(u):
        res.update({"median_latest_plain": float(u.latest_plain.median()), "median_latest_masked": float(u.latest_masked.median()),
                    "median_gain": float(u.gain.median()), "share_gain_ge_0": float((u.gain >= 0).mean())})
        res["by_kind"] = {k: float(g.gain.median()) for k, g in u.groupby("kind")}
    res["verdict"] = "MASK_HELPS" if res.get("median_gain", 0) >= 3 else "NO_GAIN"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["all", "loss", "window"], default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="m2_pilot_moments")
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline
    from w2w.generate import SD_ID
    pipe = StableDiffusionPipeline.from_pretrained(SD_ID, dtype=torch.float16).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    painter = Painter(pipe, args.device, STEPS, GUIDANCE)
    base_detect = Detector(args.device)

    class Detect:
        names = base_detect.names

        def __call__(self, img, cls, th):
            return [d for d in base_detect(img, cls) if d[0] >= th]
    import pilot_steering
    pilot_steering.MIN_SCORE = min(FAIL_TH, FOUND_TH)     # the base detector filters at its own floor
    detect = Detect()

    out_dir, img_dir = RESULTS / args.out, DATA / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)

    summary_path = out_dir / "summary.json"
    summary = json.load(open(summary_path, encoding="utf-8")) if summary_path.exists() else {}
    summary["settings"] = {"steps": STEPS, "guidance": GUIDANCE, "scheduler": "DDIM", "loss_ks": list(LOSS_KS),
                           "repaints": R, "failures_wanted": N_FAIL, "scan_limit": SCAN, "fail_th": FAIL_TH,
                           "found_th": FOUND_TH, "window_ks": list(WINDOW_KS), "edit_th": EDIT_TH,
                           "scene_th": SCENE_TH, "blend_th": BLEND_TH}
    if args.arm in ("all", "loss"):
        summary["loss"] = arm_loss(painter, detect, out_dir, img_dir, log)
        summary["loss"]["minutes"] = round((time.time() - t0) / 60, 1)
        json.dump(summary, open(summary_path, "w", encoding="utf-8"), indent=2)
        log(f"loss: verdict {summary['loss'].get('verdict')}")
    if args.arm in ("all", "window"):
        clip = Clip(args.device)
        t1 = time.time()
        summary["window"] = arm_window(painter, detect, clip, out_dir, img_dir, log)
        summary["window"]["minutes"] = round((time.time() - t1) / 60, 1)
        json.dump(summary, open(summary_path, "w", encoding="utf-8"), indent=2)
        log(f"window: verdict {summary['window'].get('verdict')}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
