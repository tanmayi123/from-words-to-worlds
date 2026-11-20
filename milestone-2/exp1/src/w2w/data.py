"""COCO 2017 validation set: download, clean, and build one manifest the rest of the project reads.

The manifest has one row per image with its cleaned captions and object categories.
Cleaning is deliberately light and fully logged, because every dropped or changed
caption is a decision the report has to defend.
"""
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from .paths import COCO

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_DIR = COCO / "val2017"
ANN_DIR = COCO / "annotations"
MANIFEST = COCO / "manifest.parquet"
CLEANING_LOG = COCO / "cleaning_log.json"


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def download() -> None:
    """Fetch val2017 images (about 815 MB) and the 2017 annotations (about 252 MB)."""
    COCO.mkdir(parents=True, exist_ok=True)
    img_zip = COCO / "val2017.zip"
    ann_zip = COCO / "annotations_trainval2017.zip"
    _download(IMAGES_URL, img_zip)
    _download(ANNOTATIONS_URL, ann_zip)
    if not IMAGES_DIR.exists():
        with zipfile.ZipFile(img_zip) as z:
            z.extractall(COCO)
    if not (ANN_DIR / "captions_val2017.json").exists():
        with zipfile.ZipFile(ann_zip) as z:
            members = [m for m in z.namelist() if m.endswith(("captions_val2017.json", "instances_val2017.json"))]
            z.extractall(COCO, members)


_WS = re.compile(r"\s+")


def clean_caption(text: str) -> str:
    """Normalise whitespace and trailing punctuation spacing. Nothing else.

    COCO captions were typed by annotators, so they carry stray newlines, double
    spaces and the odd space before a period. We fix those and leave wording alone.
    """
    t = _WS.sub(" ", text.replace("\n", " ")).strip()
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    return t


def build_manifest(tokenizer=None, max_tokens: int = 77) -> pd.DataFrame:
    """One row per image: file, size, mode, cleaned captions, COCO object categories.

    If a tokenizer is given, captions longer than the CLIP window are flagged
    (they are kept; the encoder truncates them, and the flag goes in the report).
    """
    caps = json.load(open(ANN_DIR / "captions_val2017.json", encoding="utf-8"))
    inst = json.load(open(ANN_DIR / "instances_val2017.json", encoding="utf-8"))
    cat_name = {c["id"]: c["name"] for c in inst["categories"]}

    captions_by_image: dict[int, list[str]] = {}
    for a in caps["annotations"]:
        captions_by_image.setdefault(a["image_id"], []).append(a["caption"])
    cats_by_image: dict[int, set[str]] = {}
    for a in inst["annotations"]:
        cats_by_image.setdefault(a["image_id"], set()).add(cat_name[a["category_id"]])

    log = Counter()
    rows = []
    for im in tqdm(caps["images"], desc="manifest"):
        path = IMAGES_DIR / im["file_name"]
        with Image.open(path) as img:
            mode, (w, h) = img.mode, img.size
        raw = captions_by_image.get(im["id"], [])
        cleaned, seen = [], set()
        for c in raw:
            cc = clean_caption(c)
            if cc != c:
                log["captions_whitespace_fixed"] += 1
            if not cc:
                log["captions_empty_dropped"] += 1
                continue
            key = cc.lower()
            if key in seen:
                log["captions_duplicate_dropped"] += 1
                continue
            seen.add(key)
            cleaned.append(cc)
        over = 0
        if tokenizer is not None:
            for c in cleaned:
                n = len(tokenizer(c, truncation=False)["input_ids"])
                if n > max_tokens:
                    over += 1
            log["captions_over_token_limit"] += over
        if mode != "RGB":
            log["images_not_rgb"] += 1
        if min(w, h) < 256:
            log["images_short_side_under_256"] += 1
        rows.append({
            "image_id": im["id"],
            "file_name": im["file_name"],
            "width": w, "height": h, "mode": mode,
            "n_captions_raw": len(raw),
            "captions": cleaned,
            "n_captions": len(cleaned),
            "captions_over_limit": over,
            "categories": sorted(cats_by_image.get(im["id"], set())),
        })
    df = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    log["images_total"] = len(df)
    log["captions_total"] = int(df["n_captions"].sum())
    df.to_parquet(MANIFEST, index=False)
    json.dump(dict(log), open(CLEANING_LOG, "w", encoding="utf-8"), indent=2)
    return df


def load_manifest() -> pd.DataFrame:
    return pd.read_parquet(MANIFEST)


def caption_table(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten to one row per caption, with a stable caption index used by the embedding cache."""
    out = df.explode("captions").rename(columns={"captions": "caption"})
    out = out[["image_id", "file_name", "caption"]].reset_index(drop=True)
    out.insert(0, "caption_idx", range(len(out)))
    return out


def load_image_square(path: Path, size: int = 512) -> Image.Image:
    """Center-crop to square, then resize. Never squash: COCO is mostly 4:3 and wider."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
