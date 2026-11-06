"""Pilot 4: does a word's commit step predict whether its object appears?

This is the proposal's central claim, tested on the development split before any test-split run.

Design.
  Paint every caption of dev_500 once (seed 0, DDIM 30 steps, guidance 7.5), recording
  cross-attention. An instance is an object that the caption names and that the photo's COCO
  annotations contain ("person" and "dining table" are skipped, as in pilot 3). For each
  instance we record:
    commit   the commit step (r >= 0.8, as in the proposal) of the last token of the object's
             word(s); a word that never settles gets 30, later than any step;
    early    the object's share of the attention given to the caption's content words,
             averaged over steps 0 to 2 (as in pilot 3).
  Two detectors judge each instance: Faster R-CNN v2 and RetinaNet v2, both COCO-trained but of
  different design. present: either finds the object at score >= 0.5. missing: neither finds it
  at >= 0.3. Anything in between is ambiguous and left out of the tests (and counted).

Pre-registered reading (needs >= 30 missing instances, otherwise TOO_FEW):
  AUROC = probability that a missing instance has a later commit than a present one (ties 1/2),
  with a 95% interval from 2,000 bootstrap resamples of captions.
  Within-class AUROC = the same, counting only pairs of the same object class.
    PREDICTS    AUROC >= 0.65, its interval above 0.5, and the within-class interval above 0.5
    CLASS_ONLY  AUROC interval above 0.5, within-class interval not: commit step only tracks
                which kinds of object get dropped
    WEAK        both intervals above 0.5 but AUROC < 0.65
    NONE        AUROC interval includes 0.5
  The same rules are reported for the early attention share (lower share = missing) as a
  secondary measure, under "early".

Outputs: Results/m2_pilot_recall/ (instances.csv, summary.json, commit_vs_recall.png);
images: data/m2_pilot_recall/. Pilot 5 (pilot_rescue.py) reads instances.csv.

Run from the project folder (milestone-1/exp1):  python scripts/pilot_recall.py
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
from pilot_moments import named_objects, word_slots  # noqa: E402
from w2w.attention import attention_share, commit_step  # noqa: E402
from w2w.live import sentence_words  # noqa: E402
from w2w.paths import DATA, RESULTS  # noqa: E402
from w2w.rewind import Painter  # noqa: E402

STEPS, GUIDANCE, SEED = 30, 7.5, 0
PRESENT_TH, MISSING_TH = 0.5, 0.3
COMMIT_R, EARLY_STEPS = 0.8, 3
MIN_MISSING, N_BOOT, PREDICTS_BAR = 30, 2000, 0.65


# ------------------------------------------------------------------ judging

class TwoDetectors:
    """Highest score each of two COCO detectors gives a class in an image."""

    def __init__(self, device):
        from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_V2_Weights,
                                                  fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2)
        w1, w2 = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        self.models = [fasterrcnn_resnet50_fpn_v2(weights=w1).eval().to(device),
                       retinanet_resnet50_fpn_v2(weights=w2).eval().to(device)]
        self.names = [w1.meta["categories"], w2.meta["categories"]]
        assert self.names[0] == self.names[1], "the two detectors must share COCO's category list"
        self.device = device

    @torch.no_grad()
    def scores(self, img):
        """{class: (best Faster R-CNN score, best RetinaNet score)} for every class either detects."""
        x = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).to(self.device)
        best = {}
        for j, (m, names) in enumerate(zip(self.models, self.names)):
            r = m([x])[0]
            for l, s in zip(r["labels"].tolist(), r["scores"].tolist()):
                pair = best.setdefault(names[l], [0.0, 0.0])
                pair[j] = max(pair[j], float(s))
        return {k: tuple(v) for k, v in best.items()}


def verdict(s1, s2, present_th=PRESENT_TH, missing_th=MISSING_TH):
    """'present', 'missing' or 'ambiguous' from the two detectors' best scores."""
    top = max(s1, s2)
    if top >= present_th:
        return "present"
    if top < missing_th:
        return "missing"
    return "ambiguous"


# ------------------------------------------------------------------ statistics

def auroc(later, earlier):
    """P(x > y) + P(x = y) / 2 for x drawn from `later`, y from `earlier`."""
    a, b = np.asarray(later, float)[:, None], np.asarray(earlier, float)[None, :]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.size * b.size))


def within_class_auroc(df, col, sign=1.0):
    """AUROC over pairs (missing, present) of the same class only; NaN when no class has both."""
    wins = pairs = 0.0
    for _, g in df.groupby("object"):
        m = sign * g.loc[g.judge == "missing", col].to_numpy(float)
        p = sign * g.loc[g.judge == "present", col].to_numpy(float)
        if m.size and p.size:
            a, b = m[:, None], p[None, :]
            wins += (a > b).sum() + 0.5 * (a == b).sum()
            pairs += m.size * p.size
    return float(wins / pairs) if pairs else float("nan")


def test_measure(df, col, sign=1.0, n_boot=N_BOOT, seed=0):
    """AUROC and within-class AUROC of `sign * col` as a predictor of 'missing', with caption-level bootstrap.

    sign = +1: larger values predict missing (commit step). sign = -1: smaller values do (attention share).
    """
    d = df[df.judge.isin(["present", "missing"])]
    point = auroc(sign * d.loc[d.judge == "missing", col], sign * d.loc[d.judge == "present", col])
    point_w = within_class_auroc(d, col, sign)
    caps = d.caption_idx.unique()
    groups = {c: g for c, g in d.groupby("caption_idx")}
    rng = np.random.default_rng(seed)
    boot, boot_w = [], []
    for _ in range(n_boot):
        b = pd.concat([groups[c] for c in rng.choice(caps, caps.size, replace=True)])
        boot.append(auroc(sign * b.loc[b.judge == "missing", col], sign * b.loc[b.judge == "present", col]))
        boot_w.append(within_class_auroc(b, col, sign))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    lo_w, hi_w = np.nanpercentile(boot_w, [2.5, 97.5])
    return {"auroc": round(point, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "within_class_auroc": round(point_w, 3), "within_class_ci95": [round(float(lo_w), 3), round(float(hi_w), 3)]}


def read(res, n_missing, min_missing=MIN_MISSING, bar=PREDICTS_BAR):
    """The pre-registered verdict for one measure."""
    if n_missing < min_missing:
        return "TOO_FEW"
    above, above_w = res["ci95"][0] > 0.5, res["within_class_ci95"][0] > 0.5
    if not above:
        return "NONE"
    if not above_w:
        return "CLASS_ONLY"
    return "PREDICTS" if res["auroc"] >= bar else "WEAK"


# ------------------------------------------------------------------ run

def plot(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df[df.judge.isin(["present", "missing"])]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.arange(-0.5, STEPS + 1.5, 1)
    for j, (lab, c) in enumerate((("present", "#4c78a8"), ("missing", "#e45756"))):
        v = d.loc[d.judge == lab]
        ax[0].hist(v.commit, bins=bins, density=True, alpha=0.6, color=c, label=f"{lab} (n = {len(v)})")
        ax[1].hist(v.early, bins=20, range=(0, 1), density=True, alpha=0.6, color=c, label=lab)
    ax[0].set_xlabel("commit step of the object word (30 = never settles)")
    ax[1].set_xlabel("object word's share of attention, steps 0 to 2")
    for a in ax:
        a.set_ylabel("density")
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "commit_vs_recall.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="dev_500")
    ap.add_argument("--out", default="m2_pilot_recall")
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline
    from w2w.data import load_manifest
    from w2w.generate import SD_ID
    pipe = StableDiffusionPipeline.from_pretrained(SD_ID, dtype=torch.float16).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    painter = Painter(pipe, args.device, STEPS, GUIDANCE)
    det = TwoDetectors(args.device)
    classes = set(det.names[0]) - {"N/A", "__background__"}
    tok = pipe.tokenizer

    out_dir, img_dir = RESULTS / args.out, DATA / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(RESULTS / "m1_data" / f"{args.split}.csv")
    cats = {int(r.image_id): set(r.categories) for r in load_manifest().itertuples()}
    t0 = time.time()

    rows, painted = [], 0
    for r in split.itertuples():
        named = [(c, w) for c, w in named_objects(r.caption, classes) if c in cats.get(int(r.image_id), set())]
        if not named:
            continue
        tl = painter.paint(r.caption, SEED, previews=False, record=True)
        painted += 1
        tl.final.save(img_dir / f"{r.caption_idx}.png")
        scores = det.scores(tl.final)
        content = [s for w in sentence_words(tok, r.caption) if w["content"] for s in w["slots"]]
        share = attention_share(tl.maps, content)
        for cls, words in named:
            slots = word_slots(tok, r.caption, words)
            if not slots:
                continue
            slot = slots[-1]
            c = commit_step(tl.maps, slot, threshold=COMMIT_R)
            idx = [content.index(s) for s in slots if s in content]
            s1, s2 = scores.get(cls, (0.0, 0.0))
            rows.append({"caption_idx": int(r.caption_idx), "image_id": int(r.image_id), "caption": r.caption,
                         "object": cls, "words": " ".join(words), "slot": int(slot),
                         "commit": STEPS if c is None else int(c), "never_settles": c is None,
                         "early": round(float(share[:EARLY_STEPS, idx].sum(1).mean()), 4) if idx else float("nan"),
                         "frcnn": round(s1, 4), "retina": round(s2, 4), "judge": verdict(s1, s2)})
        if painted % 50 == 0:
            n_miss = sum(x["judge"] == "missing" for x in rows)
            print(f"[{(time.time() - t0) / 60:5.1f} min] {painted} captions painted, {len(rows)} instances, {n_miss} missing", flush=True)

    df = pd.DataFrame(rows)
    assert len(df), "no caption in the split names an object its photo contains"
    df.to_csv(out_dir / "instances.csv", index=False, lineterminator="\n")
    counts = df.judge.value_counts().to_dict()
    n_missing = int(counts.get("missing", 0))
    commit_res = test_measure(df, "commit", +1.0)
    early_res = test_measure(df.dropna(subset=["early"]), "early", -1.0)
    d = df[df.judge.isin(["present", "missing"])]
    summary = {
        "settings": {"split": args.split, "steps": STEPS, "guidance": GUIDANCE, "seed": SEED, "scheduler": "DDIM",
                     "present_th": PRESENT_TH, "missing_th": MISSING_TH, "commit_r": COMMIT_R,
                     "early_steps": EARLY_STEPS, "bootstrap": N_BOOT, "predicts_bar": PREDICTS_BAR},
        "captions_painted": int(df.caption_idx.nunique()) if len(df) else 0,
        "instances": int(len(df)), "judge_counts": {k: int(v) for k, v in counts.items()},
        "missing_rate": round(n_missing / max(1, counts.get("missing", 0) + counts.get("present", 0)), 3),
        "detectors_disagree": int(((df.frcnn >= PRESENT_TH) != (df.retina >= PRESENT_TH)).sum()),
        "median_commit": {"present": float(d.loc[d.judge == "present", "commit"].median()),
                          "missing": float(d.loc[d.judge == "missing", "commit"].median())},
        "never_settles": {"present": int(d.loc[d.judge == "present", "never_settles"].sum()),
                          "missing": int(d.loc[d.judge == "missing", "never_settles"].sum())},
        "missing_by_object": {k: int(v) for k, v in d[d.judge == "missing"].object.value_counts().head(15).items()},
        "commit": {**commit_res, "verdict": read(commit_res, n_missing)},
        "early": {**early_res, "verdict": read(early_res, n_missing)},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    plot(df, out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
