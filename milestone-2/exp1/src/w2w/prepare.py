"""Image preprocessing, evaluation splits and the data card.

Run from the project root after `pip install -e .`:  python -m w2w.prepare

What it does, in order:
  1. Converts every val2017 image to RGB, center-crops it to a square and resizes it to
     512 x 512 with Lanczos filtering, saved as PNG so real and generated images go through
     the same lossless format before any metric sees them.
  2. Flags images whose short side is under 256 px. They stay in the caption data but are
     excluded from the image reference set used for FID, KID and precision/recall, because
     upsampling them more than 2x produces blur that would penalise every generated image.
  3. Freezes three disjoint image sets with one fixed seed, drawn in this order so that
     eval_2000 is the same as when it was first frozen:
       eval_2000:  2,000 reference-eligible images, one caption each (drawn at random), the
                   test set. eval_500 is its first 500 rows, for multi-seed variance runs.
       dev_500:    500 further images, one caption each, for choosing settings (guidance,
                   checkpoints, thresholds) without touching the test set.
       ref_2000:   2,000 further images, photos only, used solely as the real reference for
                   FID and KID: real-vs-real scores against the test photos give each metric
                   its floor at the same sample size.
  4. Writes a data card (JSON) with every count, every decision, and the seed.
"""
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .data import IMAGES_DIR, caption_table, load_manifest
from .paths import COCO, RESULTS

SIZE = 512
MIN_SHORT_SIDE = 256
SEED = 2026
OUT_IMAGES = COCO / "val2017_512"
SPLITS = COCO / "splits"
CARD_DIR = RESULTS / "m1_data"


def square_512(path: Path) -> tuple[Image.Image, dict]:
    img = Image.open(path)
    info = {"mode_in": img.mode, "w": img.width, "h": img.height}
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((SIZE, SIZE), Image.LANCZOS)
    info["crop_fraction_kept"] = round(s * s / (w * h), 4)
    return img, info


def main() -> None:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    CARD_DIR.mkdir(parents=True, exist_ok=True)

    df = load_manifest()
    caps = caption_table(df)           # rebuilt from the manifest, so this runs without notebook 01's extras

    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="preprocess images", mininterval=30):
        out = OUT_IMAGES / (Path(r.file_name).stem + ".png")
        if out.exists():                 # resume: read the original's header only, skip the work
            with Image.open(IMAGES_DIR / r.file_name) as im:
                w, h, mode = im.width, im.height, im.mode
            s = min(w, h)
            info = {"mode_in": mode, "w": w, "h": h, "crop_fraction_kept": round(s * s / (w * h), 4)}
        else:
            img, info = square_512(IMAGES_DIR / r.file_name)
            tmp = out.with_name(out.stem + ".tmp.png")
            img.save(tmp)
            tmp.replace(out)             # atomic, so an interrupted run never leaves a truncated PNG
        rows.append({"image_id": r.image_id, "file_512": out.name, **info,
                     "reference_eligible": min(info["w"], info["h"]) >= MIN_SHORT_SIDE})
    pre = pd.DataFrame(rows)
    pre.to_parquet(COCO / "preprocess_log.parquet", index=False)

    # Frozen evaluation subsets: reference-eligible images, one random caption each.
    rng = random.Random(SEED)
    eligible = sorted(pre.loc[pre.reference_eligible, "image_id"].tolist())
    chosen = rng.sample(eligible, 2000)
    eval_rows = []
    for iid in chosen:
        options = caps.loc[caps.image_id == iid, "caption_idx"].tolist()
        ci = rng.choice(options)
        eval_rows.append({"image_id": iid, "caption_idx": ci,
                          "caption": caps.loc[caps.caption_idx == ci, "caption"].item()})
    ev = pd.DataFrame(eval_rows)

    # Development and reference sets, drawn after the test set from the same generator, so the
    # test set stays exactly as first frozen. All three are disjoint by image.
    rest = sorted(set(eligible) - set(chosen))
    dev_ids = rng.sample(rest, 500)
    dev_rows = []
    for iid in dev_ids:
        options = caps.loc[caps.image_id == iid, "caption_idx"].tolist()
        ci = rng.choice(options)
        dev_rows.append({"image_id": iid, "caption_idx": ci,
                         "caption": caps.loc[caps.caption_idx == ci, "caption"].item()})
    dev = pd.DataFrame(dev_rows)
    rest = sorted(set(rest) - set(dev_ids))
    ref_ids = rng.sample(rest, 2000)
    files = dict(zip(pre.image_id, pre.file_512))
    ref = pd.DataFrame({"image_id": ref_ids, "file_512": [files[i] for i in ref_ids]})
    assert not (set(chosen) & set(dev_ids)) and not (set(chosen) & set(ref_ids)) and not (set(dev_ids) & set(ref_ids))

    for target in (SPLITS, CARD_DIR):    # a copy under Results/ is committed with the repository
        ev.to_csv(target / "eval_2000.csv", index=False, lineterminator="\n")
        ev.head(500).to_csv(target / "eval_500.csv", index=False, lineterminator="\n")
        dev.to_csv(target / "dev_500.csv", index=False, lineterminator="\n")
        ref.to_csv(target / "ref_2000.csv", index=False, lineterminator="\n")

    per_image = caps.groupby("image_id").size()
    cleaning = json.load(open(COCO / "cleaning_log.json", encoding="utf-8"))
    card = {
        "source": "MS-COCO 2017 validation split (images + captions_val2017.json + instances_val2017.json)",
        "images": int(len(df)),
        "captions_after_cleaning": int(len(caps)),
        "captions_per_image": {"min": int(per_image.min()), "median": float(per_image.median()),
                               "max": int(per_image.max())},
        "caption_cleaning": cleaning,
        "caption_policy": "whitespace and spacing-before-punctuation repaired; exact duplicates within an "
                          "image (case-insensitive) dropped; wording, spelling and case never changed",
        "image_preprocessing": {
            "converted_to_rgb": int((pre.mode_in != "RGB").sum()),
            "center_crop_then_resize": f"{SIZE}x{SIZE}, Lanczos, PNG",
            "mean_fraction_of_image_kept_by_crop": round(float(pre.crop_fraction_kept.mean()), 4),
            "min_fraction_kept": float(pre.crop_fraction_kept.min()),
        },
        "reference_set": {
            "rule": f"short side >= {MIN_SHORT_SIDE} px",
            "eligible_images": int(pre.reference_eligible.sum()),
            "excluded_small_images": int((~pre.reference_eligible).sum()),
        },
        "splits": {"seed": SEED, "disjoint_by_image": True,
                   "eval_2000": "Results/m1_data/eval_2000.csv (test set)",
                   "eval_500": "Results/m1_data/eval_500.csv (first 500 rows of eval_2000)",
                   "dev_500": "Results/m1_data/dev_500.csv (choosing settings)",
                   "ref_2000": "Results/m1_data/ref_2000.csv (real reference photos for FID and KID)",
                   "unused_eligible_images": int(len(eligible) - 4500)},
        "object_categories": {"images_with_at_least_one_annotated_object": int((df.categories.map(len) > 0).sum()),
                              "images_with_none": int((df.categories.map(len) == 0).sum())},
    }
    with open(CARD_DIR / "data_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(json.dumps(card, indent=2))
    examples_figure(pre, df)


def examples_figure(pre: pd.DataFrame, df: pd.DataFrame) -> None:
    """Before and after for four representative cases: wide, tall, grayscale, small."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aspect = pre.w / pre.h
    picks = [
        ("widest", pre.loc[aspect.idxmax()]),
        ("tallest", pre.loc[aspect.idxmin()]),
        ("grayscale", pre.loc[pre.mode_in != "RGB"].iloc[0]),
        ("small (excluded from reference)", pre.loc[~pre.reference_eligible].iloc[0]),
    ]
    names = dict(zip(df.image_id, df.file_name))
    fig, ax = plt.subplots(2, 4, figsize=(10, 5.2))
    for j, (label, r) in enumerate(picks):
        orig = Image.open(IMAGES_DIR / names[r.image_id]).convert("RGB")
        ax[0, j].imshow(orig)
        ax[0, j].set_title(f"{label}\n{r.w} x {r.h}, {r.mode_in}", fontsize=8)
        ax[1, j].imshow(Image.open(OUT_IMAGES / r.file_512))
        ax[1, j].set_title(f"512 x 512, keeps {r.crop_fraction_kept:.0%} of the image", fontsize=8)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("original", fontsize=9); ax[1, 0].set_ylabel("preprocessed", fontsize=9)
    fig.tight_layout()
    fig.savefig(CARD_DIR / "preprocessing_examples.png", dpi=150)


if __name__ == "__main__":
    main()
