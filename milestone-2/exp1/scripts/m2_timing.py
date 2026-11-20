"""Milestone 2, word timing at scale: when does Stable Diffusion paint each named object, and
does a late start predict that the object goes missing?

This is the confirmatory run of pilot 4 (scripts/pilot_recall.py). On the development split the
pilot found that a late commit step predicts a missing object (AUROC 0.817, 95% interval 0.735
to 0.885; within-class 0.82; verdict PREDICTS). Here the same test runs once on the held-out
eval_2000 split, with the group's object labels (Praniti's groundtruth files) and the group's
judges (Tanmayi's exp4: Faster R-CNN v2 or OWL-ViT). The reading rules are fixed in
docs/m2_preregistration.md before the run.

Phases, each resumable:
  generate  paint each caption once under one setting with Painter, recording cross-attention.
            Images and the generation log follow the group protocol (w2w.m2_protocol), so the
            pictures double as that setting's images for the image-quality scores. For every
            labelled object: the commit step at r >= 0.7, 0.8 and 0.9, the timestep of the 0.8
            commit, the early attention share and the final-step drop.
  judge     score every image with Faster R-CNN v2, OWL-ViT and RetinaNet v2.
  analyze   the pre-registered tests, the sensitivity checks and the figures.

Run from the project folder:
  python scripts/m2_timing.py generate --split eval_2000
  python scripts/m2_timing.py judge --split eval_2000
  python scripts/m2_timing.py analyze --split eval_2000
The Thursday grid on the first 200 dev captions uses the same phases, for example:
  python scripts/m2_timing.py generate --split dev_500 --first 200 --spacing trailing
  python scripts/m2_timing.py generate --split dev_500 --first 200 --guidance 3
  python scripts/m2_timing.py generate --split dev_500 --first 200 --seed-id 1
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_recall import read, verdict  # noqa: E402
from w2w import m2_protocol as m2  # noqa: E402
from w2w.attention import attention_share, commit_step, correlation_curve, final_step_drop  # noqa: E402
from w2w.live import sentence_words  # noqa: E402
from w2w.paths import DATA, RESULTS  # noqa: E402

STEPS = 30
THRESHOLDS = (0.7, 0.8, 0.9)
PRIMARY = "commit_08"
EARLY_STEPS = 3
DET_TH, OWL_TH = 0.5, 0.1                      # the group's judges (exp4)
PILOT_PRESENT, PILOT_MISSING = 0.5, 0.3       # pilot 4's two-detector rule
N_BOOT, BOOT_SEED = 2000, 0
SPLITS = RESULTS / "m2_splits"
OWNER = "Santhosh"
# Captions an earlier experiment already painted, by row of the split. Pilot 3's moment-of-loss arm
# painted eval_2000 in order until its 30th failure, at row 431; exp4's 60 word-timing captions
# (rows 2 to 94) fall inside the same block. The primary test leaves these rows out.
SEEN_BEFORE = {"eval_2000": 432}


# ------------------------------------------------------------------ setting and folders

def setting(args) -> m2.GenConfig:
    return m2.GenConfig("ddim", args.spacing, STEPS, args.guidance)


def run_dir(args) -> Path:
    tag = f"{args.split}{'_first' + str(args.first) if args.first else ''}__{setting(args).config_id}__s{args.seed_id}"
    d = RESULTS / "m2_timing" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_split(name: str, first: int | None = None) -> pd.DataFrame:
    df = m2.load_split(name, root=SPLITS)
    return df.head(first) if first else df


def targets_for(split: str) -> pd.DataFrame:
    g = m2.load_split(f"groundtruth_{split}", root=SPLITS)
    return g[g.target.astype(bool)].copy()


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ words to token slots

def whitespace_word_slots(tok, caption: str) -> list[list[int]]:
    """Token slots of each whitespace-separated word of the caption (slot 0 is the start token).

    Praniti's labels index words by caption.split(). CLIP never merges text across a space, so
    tokenizing word by word gives the caption's own tokens; the assertion checks that."""
    pos, spans = 1, []
    for w in caption.split():
        n = len(tok(w, add_special_tokens=False)["input_ids"])
        spans.append(list(range(pos, pos + n)))
        pos += n
    total = len(tok(caption, add_special_tokens=True)["input_ids"]) - 2
    assert total == pos - 1, f"word-by-word tokens ({pos - 1}) differ from the caption's ({total}): {caption!r}"
    return spans


def object_slots(tok, caption: str, word_start: int, n_words: int) -> tuple[list[int], int, str]:
    """(all slots of the object's words, the slot measured, its token). The measured slot is the
    last one whose token has a letter or digit, so "people," is measured on "people"."""
    ids = tok(caption, add_special_tokens=True)["input_ids"]
    spans = whitespace_word_slots(tok, caption)
    slots = [s for sp in spans[word_start:word_start + n_words] for s in sp]
    texts = {s: tok.convert_ids_to_tokens(ids[s]).replace("</w>", "") for s in slots}
    alnum = [s for s in slots if any(ch.isalnum() for ch in texts[s])]
    if not alnum:
        raise ValueError(f"no word token for words {word_start}..{word_start + n_words - 1} of {caption!r}")
    return slots, alnum[-1], texts[alnum[-1]]


# ------------------------------------------------------------------ phase 1: generate

def measure_objects(tok, caption: str, maps: np.ndarray, timesteps: list[int], objects: pd.DataFrame) -> list[dict]:
    """One row per labelled object of one caption, from its painting's recorded attention."""
    words = sentence_words(tok, caption)
    content = [s for w in words if w["content"] for s in w["slots"]]
    share = attention_share(maps, content) if content else None
    rows = []
    for o in objects.itertuples(index=False):
        n_words = len(str(o.words).split())
        slots, slot, token = object_slots(tok, caption, int(o.word_start), n_words)
        if slot >= maps.shape[1]:
            continue
        row = {"caption_idx": int(o.caption_idx), "image_id": int(o.image_id), "category": o.category,
               "words": o.words, "word_start": int(o.word_start), "slot": int(slot), "token": token,
               "n_tokens": len(slots), "n_instances": int(o.n_instances),
               "cover_after_crop": float(o.cover_after_crop)}
        for th in THRESHOLDS:
            c = commit_step(maps, slot, threshold=th)
            row[f"commit_{int(round(th * 10)):02d}"] = STEPS if c is None else int(c)
        c8 = row[PRIMARY]
        row["never_settles"] = c8 == STEPS
        row["commit_t"] = int(timesteps[c8]) if c8 < STEPS else -1
        idx = [content.index(s) for s in slots if s in content]
        row["early"] = round(float(share[:EARLY_STEPS, idx].sum(1).mean()), 4) if idx else float("nan")
        row["final_drop"] = round(final_step_drop(maps, slot), 4)
        # per-step curves, kept for Milestone 3 (when is a missing object lost, relative to its commit?)
        r_curve = correlation_curve(maps, slot)
        s_curve = share[:, idx].sum(1) if idx else np.full(maps.shape[0], np.nan)
        row["curves"] = {**{f"r_{t:02d}": round(float(v), 4) for t, v in enumerate(r_curve)},
                         **{f"share_{t:02d}": round(float(v), 4) for t, v in enumerate(s_curve)}}
        rows.append(row)
    return rows


def append_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


OBJECT_COLUMNS = ["caption_idx", "image_id", "category", "words", "word_start", "slot", "token", "n_tokens",
                  "n_instances", "cover_after_crop", "commit_07", "commit_08", "commit_09", "never_settles",
                  "commit_t", "early", "final_drop"]
CURVE_COLUMNS = ["caption_idx", "category", "word_start", "slot"] + [f"r_{t:02d}" for t in range(STEPS)] + \
                [f"share_{t:02d}" for t in range(STEPS)]
CAPTION_COLUMNS = ["caption_idx", "image_id", "file", "flagged", "seconds", "n_objects"]


def generate(args) -> None:
    from w2w.rewind import Painter
    cfg = setting(args)
    out = run_dir(args)
    split = load_split(args.split, args.first)
    targets = targets_for(args.split)
    by_caption = {k: g for k, g in targets.groupby("caption_idx")}
    done = set(pd.read_csv(out / "captions.csv").caption_idx) if (out / "captions.csv").exists() else set()
    for name in ("objects.csv", "curves.csv"):  # drop rows of a caption cut off before it was marked done
        if (out / name).exists():
            o = pd.read_csv(out / name)
            o[o.caption_idx.isin(done)].to_csv(out / name, index=False, lineterminator="\n")

    pipe = m2.load_pipeline(args.device)
    painter = Painter(pipe, args.device, STEPS, args.guidance, spacing=args.spacing, cpu_noise=True)
    tok = pipe.tokenizer
    log = m2.RunLog(RESULTS / "m2_runs", owner=OWNER, axis=args.axis, images_dir=DATA / "m2_images")
    img_dir = log.images_dir / args.split / cfg.config_id
    img_dir.mkdir(parents=True, exist_ok=True)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    started, t0, made = m2._now(), time.time(), 0
    for r in split.itertuples(index=False):
        cap = int(r.caption_idx)
        if cap in done:
            continue
        seed = m2.seed_for(cap, args.seed_id)
        sync()
        t1 = time.perf_counter()
        tl = painter.paint(r.caption, seed, record=True, previews=False, safe=True)
        sync()
        dt = time.perf_counter() - t1
        path = img_dir / f"{cap}_s{args.seed_id}.png"
        tl.final.save(path)
        rel = path.relative_to(log.images_dir).as_posix()
        log.image(cfg, args.split, cap, args.seed_id, seed, rel, tl.flagged, dt)
        objs = measure_objects(tok, r.caption, tl.maps, tl.timesteps, by_caption[cap]) if cap in by_caption else []
        append_csv(out / "objects.csv", objs, OBJECT_COLUMNS)
        append_csv(out / "curves.csv", [{**{k: o[k] for k in CURVE_COLUMNS[:4]}, **o["curves"]} for o in objs],
                   CURVE_COLUMNS)
        append_csv(out / "captions.csv", [{"caption_idx": cap, "image_id": int(r.image_id), "file": rel,
                                           "flagged": int(tl.flagged), "seconds": round(dt, 3),
                                           "n_objects": len(objs)}], CAPTION_COLUMNS)
        made += 1
        if made % 50 == 0:
            print(f"[{(time.time() - t0) / 60:5.1f} min] {len(done) + made}/{len(split)} captions, "
                  f"{(time.time() - t0) / made:.2f} s each", flush=True)
    note = "made with Painter while recording attention, which adds time per image"
    if args.first:
        note += f"; first {args.first} captions"
    log.add_run(cfg, args.split, len(split), [args.seed_id], started=started,
                dtype=str(pipe.unet.dtype).replace("torch.", ""), notes=note)
    print(f"generate done: {len(done) + made} captions, {made} new, in {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ phase 2: judge

class Judges:
    """Faster R-CNN v2 and OWL-ViT exactly as in the group's exp4 notebook, plus RetinaNet v2 for
    pilot 4's rule. Each gives, per image, its best score for every COCO class."""

    def __init__(self, device: str):
        from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_V2_Weights, RetinaNet_ResNet50_FPN_V2_Weights,
                                                  fasterrcnn_resnet50_fpn_v2, retinanet_resnet50_fpn_v2)
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
        w1, w2 = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        self.frcnn = fasterrcnn_resnet50_fpn_v2(weights=w1).to(device).eval()
        self.retina = retinanet_resnet50_fpn_v2(weights=w2).to(device).eval()
        self.tf = w1.transforms()
        self.cats = w1.meta["categories"]
        assert self.cats == w2.meta["categories"]
        self.classes = [c for c in self.cats if c not in ("N/A", "__background__")]
        self.owl_proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device).eval()
        self.queries = [f"a photo of a {c}" for c in self.classes]
        self.device = device

    @torch.no_grad()
    def _detector(self, model, ims) -> list[dict]:
        out = []
        for o in model([self.tf(im).to(self.device) for im in ims]):
            best = {}
            for lab, sc in zip(o["labels"].tolist(), o["scores"].tolist()):
                c = self.cats[lab]
                best[c] = max(best.get(c, 0.0), float(sc))
            out.append(best)
        return out

    @torch.no_grad()
    def scores(self, ims) -> list[dict]:
        """Per image: {class: (Faster R-CNN, RetinaNet, OWL-ViT)} best scores."""
        a, b = self._detector(self.frcnn, ims), self._detector(self.retina, ims)
        inp = self.owl_proc(text=[self.queries] * len(ims), images=ims, return_tensors="pt").to(self.device)
        owl = torch.sigmoid(self.owl(**inp).logits).max(1).values.float().cpu().numpy()
        return [{c: (a[i].get(c, 0.0), b[i].get(c, 0.0), float(owl[i][k])) for k, c in enumerate(self.classes)}
                for i in range(len(ims))]


JUDGE_COLUMNS = ["caption_idx", "category", "frcnn", "retina", "owl"]


def judge(args, batch: int = 8) -> None:
    from PIL import Image
    out = run_dir(args)
    objs = pd.read_csv(out / "objects.csv")
    caps = pd.read_csv(out / "captions.csv")
    done = set(pd.read_csv(out / "judges.csv").caption_idx) if (out / "judges.csv").exists() else set()
    todo = caps[caps.caption_idx.isin(objs.caption_idx) & ~caps.caption_idx.isin(done)]
    j = Judges(args.device)
    want = {k: sorted(set(g.category)) for k, g in objs.groupby("caption_idx")}
    t0 = time.time()
    for i in range(0, len(todo), batch):
        part = todo.iloc[i:i + batch]
        ims = [Image.open(DATA / "m2_images" / f).convert("RGB") for f in part.file]
        rows = []
        for cap, sc in zip(part.caption_idx, j.scores(ims)):
            for c in want[cap]:
                f, r, o = sc[c]
                rows.append({"caption_idx": int(cap), "category": c, "frcnn": round(f, 4), "retina": round(r, 4),
                             "owl": round(o, 4)})
        append_csv(out / "judges.csv", rows, JUDGE_COLUMNS)
        if (i // batch + 1) % 25 == 0:
            print(f"[{(time.time() - t0) / 60:5.1f} min] judged {min(i + batch, len(todo))}/{len(todo)} images", flush=True)
    print(f"judge done: {len(done) + len(todo)} images in {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ phase 3: analyze

def judge_group(f, o):
    """The group's rule (exp4): present when either judge finds the class."""
    return "present" if (f >= DET_TH or o >= OWL_TH) else "missing"


def judge_frcnn(f, o):
    return "present" if f >= DET_TH else "missing"


def judge_agree(f, o):
    """Only cases where both judges agree; disagreements are left out."""
    a, b = f >= DET_TH, o >= OWL_TH
    return "present" if (a and b) else ("missing" if not (a or b) else "ambiguous")


def _wins(x_m, w_m, x_p, w_p) -> tuple[float, float]:
    """Weighted count of (missing, present) pairs where the missing value is larger (ties 1/2), and of all pairs."""
    if x_m.size == 0 or x_p.size == 0:
        return 0.0, 0.0
    order = np.argsort(x_p, kind="mergesort")
    xs, cw = x_p[order], np.concatenate([[0.0], np.cumsum(w_p[order])])
    lo, hi = np.searchsorted(xs, x_m, side="left"), np.searchsorted(xs, x_m, side="right")
    return float((w_m * (cw[lo] + 0.5 * (cw[hi] - cw[lo]))).sum()), float(w_m.sum() * cw[-1])


def _aurocs(val, miss, grp, w) -> tuple[float, float]:
    """(AUROC, within-group AUROC) with row weights `w` (bootstrap multiplicities)."""
    m, p = miss & (w > 0), ~miss & (w > 0)
    num, den = _wins(val[m], w[m], val[p], w[p])
    wn = wd = 0.0
    for gcode in np.unique(grp[m]):
        gm, gp = m & (grp == gcode), p & (grp == gcode)
        a, b = _wins(val[gm], w[gm], val[gp], w[gp])
        wn, wd = wn + a, wd + b
    return (num / den if den else float("nan")), (wn / wd if wd else float("nan"))


def test(df: pd.DataFrame, col: str, sign: float = 1.0, group: str = "object", n_boot: int = N_BOOT,
         seed: int = BOOT_SEED) -> dict:
    """pilot_recall.test_measure with a choice of grouping column, computed with weights instead of
    copied rows (the same resampled captions, the same numbers, much faster). AUROC that
    `sign * col` is larger for missing objects; 95% interval from caption-level bootstrap."""
    d = df[df.judge.isin(["present", "missing"])].reset_index(drop=True)
    val = sign * d[col].to_numpy(float)
    miss = (d.judge == "missing").to_numpy()
    grp = pd.factorize(d[group])[0]
    caps = d.caption_idx.unique()
    sorter = np.argsort(caps)

    def codes(x):  # position of each caption id in `caps`
        return sorter[np.searchsorted(caps, x, sorter=sorter)]

    code = codes(d.caption_idx.to_numpy())
    point, point_w = _aurocs(val, miss, grp, np.ones(len(d)))
    rng = np.random.default_rng(seed)
    boot, boot_w = [], []
    for _ in range(n_boot):
        counts = np.bincount(codes(rng.choice(caps, caps.size, replace=True)), minlength=caps.size)
        a, b = _aurocs(val, miss, grp, counts[code].astype(float))
        boot.append(a)
        boot_w.append(b)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    lo_w, hi_w = np.nanpercentile(boot_w, [2.5, 97.5])
    return {"auroc": round(point, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "within_class_auroc": round(point_w, 3), "within_class_ci95": [round(float(lo_w), 3), round(float(hi_w), 3)],
            "n_present": int((~miss).sum()), "n_missing": int(miss.sum()), "n_captions": int(caps.size)}


def judged_table(out: Path) -> pd.DataFrame:
    objs = pd.read_csv(out / "objects.csv")
    caps = pd.read_csv(out / "captions.csv")[["caption_idx", "flagged"]]
    js = pd.read_csv(out / "judges.csv")
    df = objs.merge(js, on=["caption_idx", "category"], how="left", validate="many_to_one").merge(caps, on="caption_idx")
    assert df.frcnn.notna().all(), "some objects have no judge scores: run the judge phase"
    df["object"] = df.category
    df["judge_group"] = [judge_group(f, o) for f, o in zip(df.frcnn, df.owl)]
    df["judge_pilot"] = [verdict(f, r, PILOT_PRESENT, PILOT_MISSING) for f, r in zip(df.frcnn, df.retina)]
    df["judge_frcnn"] = [judge_frcnn(f, o) for f, o in zip(df.frcnn, df.owl)]
    df["judge_agree"] = [judge_agree(f, o) for f, o in zip(df.frcnn, df.owl)]
    df["size_tertile"] = pd.qcut(df.cover_after_crop.rank(method="first"), 3, labels=["small", "medium", "large"])
    return df


def run_test(df, judge_col, col, sign, group="object", label="", n_boot=N_BOOT) -> dict:
    d = df.assign(judge=df[judge_col])
    d = d.dropna(subset=[col])
    res = test(d, col, sign, group=group, n_boot=n_boot)
    res["verdict"] = read(res, res["n_missing"])
    res["label"] = label
    return res


def unseen_captions(split: str) -> set:
    """caption_idx of the split's rows that no earlier experiment painted (see SEEN_BEFORE)."""
    return set(load_split(split).caption_idx.iloc[SEEN_BEFORE.get(split, 0):])


def analyze(args) -> dict:
    out = run_dir(args)
    df = judged_table(out)
    df.to_csv(out / "objects_judged.csv", index=False, lineterminator="\n")
    clean = df[df.flagged == 0]
    base = clean[clean.caption_idx.isin(unseen_captions(args.split))]       # the primary population
    primary = run_test(base, "judge_group", PRIMARY, +1.0, label="primary: commit r>=0.8, group judges")
    secondary = run_test(base, "judge_group", "early", -1.0, label="secondary: early share, group judges")
    first_obj = base.sort_values(["caption_idx", "word_start"]).groupby("caption_idx").head(1)
    checks = [
        run_test(clean, "judge_group", PRIMARY, +1.0, label="all captions, including those painted before"),
        run_test(base, "judge_pilot", PRIMARY, +1.0, label="pilot 4's judges (Faster R-CNN, RetinaNet)"),
        run_test(base, "judge_frcnn", PRIMARY, +1.0, label="Faster R-CNN alone"),
        run_test(base, "judge_agree", PRIMARY, +1.0, label="both judges agree"),
        run_test(base, "judge_group", "commit_07", +1.0, label="commit r>=0.7"),
        run_test(base, "judge_group", "commit_09", +1.0, label="commit r>=0.9"),
        run_test(base[base.category != "person"], "judge_group", PRIMARY, +1.0, label="without person"),
        run_test(first_obj, "judge_group", PRIMARY, +1.0, label="first object of each caption only"),
        run_test(base, "judge_group", PRIMARY, +1.0, group="size_tertile", label="within object-size thirds"),
    ]
    d = base.assign(judge=base.judge_group)
    pm = d[d.judge.isin(["present", "missing"])]
    by_step = pm.groupby(PRIMARY).judge.apply(lambda s: (s == "missing").mean()).round(4)
    summary = {
        "settings": {"split": args.split, "first": args.first, "config_id": setting(args).config_id,
                     "seed_id": args.seed_id, "steps": STEPS, "commit_r": 0.8, "thresholds": list(THRESHOLDS),
                     "early_steps": EARLY_STEPS, "frcnn_th": DET_TH, "owl_th": OWL_TH,
                     "pilot_present_th": PILOT_PRESENT, "pilot_missing_th": PILOT_MISSING,
                     "bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED,
                     "protocol_sha": m2.protocol_sha(), "script_sha": file_sha(Path(__file__))},
        "captions": int(pd.read_csv(out / "captions.csv").shape[0]),
        "flagged_images": int(pd.read_csv(out / "captions.csv").flagged.sum()),
        "objects": int(len(df)), "objects_in_flagged_images": int((df.flagged == 1).sum()),
        "rows_painted_before": SEEN_BEFORE.get(args.split, 0),
        "primary_population": {"captions": int(base.caption_idx.nunique()), "objects": int(len(base))},
        "judge_counts": d.judge.value_counts().to_dict(),
        "missing_rate": round(float((pm.judge == "missing").mean()), 4),
        "median_commit": {k: float(g[PRIMARY].median()) for k, g in pm.groupby("judge")},
        "median_commit_timestep": {k: float(g.commit_t[g.commit_t >= 0].median()) for k, g in pm.groupby("judge")},
        "never_settles": {k: int(g.never_settles.sum()) for k, g in pm.groupby("judge")},
        "missing_rate_by_commit_step": {int(k): float(v) for k, v in by_step.items()},
        "missing_by_class": {k: int(v) for k, v in pm[pm.judge == "missing"].category.value_counts().head(15).items()},
        "primary": primary, "secondary": secondary, "sensitivity": checks,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot(pm, out)
    print(json.dumps({k: summary[k] for k in ("captions", "objects", "judge_counts", "missing_rate", "median_commit",
                                              "primary", "secondary")}, indent=2))
    for c in checks:
        print(f"  {c['label']:44s} AUROC {c['auroc']:.3f} {c['ci95']}  within {c['within_class_auroc']:.3f} "
              f"{c['within_class_ci95']}  missing {c['n_missing']:4d}  {c['verdict']}")
    return summary


def plot(pm: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.arange(-0.5, STEPS + 1.5, 1)
    for lab, c in (("present", "#4c78a8"), ("missing", "#e45756")):
        v = pm[pm.judge == lab]
        ax[0].hist(v[PRIMARY], bins=bins, density=True, alpha=0.6, color=c, label=f"{lab} (n = {len(v)})")
        ax[1].hist(v.early.dropna(), bins=20, range=(0, 1), density=True, alpha=0.6, color=c, label=lab)
    ax[0].set_xlabel(f"commit step of the object word ({STEPS} = never settles)")
    ax[1].set_xlabel(f"object word's share of attention, steps 0 to {EARLY_STEPS - 1}")
    for a in ax:
        a.set_ylabel("density")
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "commit_vs_recall.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ phase 4: the same objects under several seeds

OBJECT_KEY = ["caption_idx", "category", "word_start"]
MIN_DISCORDANT = 30


def within_object_test(df: pd.DataFrame, col: str, sign: float = 1.0, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> dict:
    """Compares paintings of the same caption and object that differ only in their starting noise.

    `df` has one row per object and seed, with `judge` present or missing. Only discordant
    objects count: missing under at least one seed and present under another. For each one,
    every (missing seed, present seed) pair is a test, won when `sign * col` is larger in the
    missing painting (ties count half). The within-object AUROC is all wins over all pairs, so
    nothing about the caption, the object's class or its size can produce it. 95% interval from
    caption-level bootstrap."""
    d = df[df.judge.isin(["present", "missing"])].dropna(subset=[col])
    wins, pairs, diffs, caps = [], [], [], []
    for key, g in d.groupby(OBJECT_KEY):
        m = sign * g.loc[g.judge == "missing", col].to_numpy(float)
        p = sign * g.loc[g.judge == "present", col].to_numpy(float)
        if m.size and p.size:
            a, b = m[:, None], p[None, :]
            wins.append(float((a > b).sum() + 0.5 * (a == b).sum()))
            pairs.append(float(m.size * p.size))
            diffs.append(float(sign * (m.mean() - p.mean())))      # in the column's own units
            caps.append(key[0])
    n = len(wins)
    if n == 0:
        return {"n_discordant": 0, "auroc": float("nan"), "ci95": [float("nan")] * 2,
                "mean_difference": float("nan"), "mean_difference_ci95": [float("nan")] * 2,
                "share_in_predicted_direction": float("nan"), "reading": "TOO_FEW"}
    wins, pairs, diffs, caps = map(np.asarray, (wins, pairs, diffs, caps))
    uniq = np.unique(caps)
    code = np.searchsorted(uniq, caps)
    rng = np.random.default_rng(seed)
    boot_a, boot_d = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size)[code].astype(float)
        boot_a.append((w * wins).sum() / (w * pairs).sum() if (w * pairs).sum() else np.nan)
        boot_d.append((w * diffs).sum() / w.sum() if w.sum() else np.nan)
    lo, hi = np.nanpercentile(boot_a, [2.5, 97.5])
    dlo, dhi = np.nanpercentile(boot_d, [2.5, 97.5])
    auroc = float(wins.sum() / pairs.sum())
    reading = "TOO_FEW" if n < MIN_DISCORDANT else ("SEEN_COMING" if lo > 0.5 else "NOT_SEEN")
    return {"n_discordant": int(n), "n_captions": int(uniq.size), "auroc": round(auroc, 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "mean_difference": round(float(diffs.mean()), 3), "mean_difference_ci95": [round(float(dlo), 3), round(float(dhi), 3)],
            "share_in_predicted_direction": round(float((sign * diffs > 0).mean()), 3), "reading": reading}


def seed_runs(args) -> pd.DataFrame:
    """objects_judged.csv of every seed_id in args.seed_ids, stacked, with a seed_id column."""
    tables = []
    for s in args.seed_ids:
        d = run_dir(argparse.Namespace(**{**vars(args), "seed_id": s}))
        t = pd.read_csv(d / "objects_judged.csv")
        t["seed_id"] = s
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


def analyze_seeds(args) -> dict:
    """The primary test repeated under each seed, then the within-object test across seeds."""
    df = seed_runs(args)
    base = df[(df.flagged == 0) & df.caption_idx.isin(unseen_captions(args.split))].assign(judge=lambda x: x.judge_group)
    per_seed = [run_test(base[base.seed_id == s], "judge_group", PRIMARY, +1.0, label=f"seed_id {s}")
                for s in args.seed_ids]
    complete = base.groupby(OBJECT_KEY).seed_id.transform("nunique") == len(args.seed_ids)
    full = base[complete]
    n_missing = full.assign(m=full.judge == "missing").groupby(OBJECT_KEY).m.sum()
    summary = {
        "settings": {"split": args.split, "config_id": setting(args).config_id, "seed_ids": list(args.seed_ids),
                     "rows_painted_before": SEEN_BEFORE.get(args.split, 0), "min_discordant": MIN_DISCORDANT,
                     "bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED, "script_sha": file_sha(Path(__file__))},
        "objects_in_every_seed": int(n_missing.size),
        "objects_by_seeds_missing": {int(k): int(v) for k, v in n_missing.value_counts().sort_index().items()},
        "per_seed": per_seed,
        "within_object_commit": within_object_test(base, PRIMARY, +1.0),
        "within_object_early": within_object_test(base, "early", -1.0),
    }
    tag = f"{args.split}{'_first' + str(args.first) if args.first else ''}"
    out = RESULTS / "m2_timing" / f"{tag}__{setting(args).config_id}__seeds{''.join(map(str, args.seed_ids))}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=["generate", "judge", "analyze", "seeds"])
    ap.add_argument("--split", default="eval_2000", choices=["eval_2000", "dev_500"])
    ap.add_argument("--first", type=int, default=None, help="only the first N captions of the split")
    ap.add_argument("--spacing", default="leading", choices=list(m2.SPACINGS))
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed-id", type=int, default=0)
    ap.add_argument("--seed-ids", type=int, nargs="+", default=[0, 1, 2], help="for the seeds phase")
    ap.add_argument("--axis", default="word timing")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"generate": generate, "judge": judge, "analyze": analyze, "seeds": analyze_seeds}[args.phase](args)


if __name__ == "__main__":
    main()
