"""Milestone 2 shared scoring, used by the whole group.

Scores any image log written by `m2_protocol.RunLog` (runs.csv + images.csv + images/), so every
member's images are measured the same way and their numbers sit in one table.

What it measures
  Object recall   Each labelled object (a `target` row of groundtruth_*.csv, or an object of a
                  stress prompt) is judged present or missing by two judges (exp4's, calibrated on
                  real photos):
                    Faster R-CNN v2 (torchvision, COCO weights): present at a class score >= 0.5
                    OWL-ViT base-patch32, query "a photo of a {class}": present at a score >= 0.1
                  An object is present when either judge finds it and missing when both miss it.
  CLIPScore       100 * max(cos(image, caption), 0) with CLIP ViT-L/14.
  FID, KID        Inception-v3 pool features (the standard FID network, via torchmetrics) of the
                  generated images against ref_2000_v2, 2,000 COCO val2017 photos that share no
                  photo with eval_2000 or either dev set. KID is reported x 1000.
  Real-photo floor  The FID and KID that real COCO photos get against ref_2000_v2 at the same sample
                  size: the photos behind the captions that were painted (none of them is in
                  ref_2000_v2). FID depends on sample size, so a generated FID is only read next to
                  the floor for the same number of images.
  Inception Score exp(E_x KL(p(y|x) || p(y))) from the same Inception network's 1,008 class
                  logits (as torchmetrics and torch-fidelity compute it). Like FID it depends on the
                  number of images, so it is compared only at equal sample sizes.
  Diversity       For a caption painted with seed_id 0 and seed_id 1 under one setting: LPIPS
                  (AlexNet) between the two images, and 1 - cosine of their CLIP image embeddings.
                  Higher means the two pictures differ more.

Images the safety checker blacked out (`flagged` = 1) are not scored and are left out of every
comparison; in a paired comparison the caption is dropped from both sides.

Intervals are 95% percentile bootstrap intervals over captions (2,000 resamples, seed 0). With
several seeds a caption and all its seeds are resampled together. Paired comparisons resample the
same captions on both sides, so the difference is measured on identical captions and identical
starting noise.

Use (Colab, GPU):
    import m2_scoring as sc
    imgs = sc.read_logs([("/content/drive/MyDrive/m2/images.csv", "/content/drive/MyDrive/m2/images")])
    scorers = sc.Scorers()
    sc.score_log(imgs, "/content/drive/MyDrive/m2/scores", scorers)          # resumable
    sc.reference_features("ref_2000_v2", photo_dir, "/content/drive/MyDrive/m2/scores", scorers)
    sc.score_diversity(imgs, "/content/drive/MyDrive/m2/scores", scorers)
    sc.score_inception_logits(imgs, "/content/drive/MyDrive/m2/scores", scorers)  # Inception Score
    sc.real_photo_features("dev_500", photo_dir, "/content/drive/MyDrive/m2/scores", scorers)  # floor

Then, on any machine (CPU is enough):
    table = sc.summary_table("/content/drive/MyDrive/m2/scores")
"""
from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCORING_VERSION = "m2-score-v2.1"

FRCNN_MIN_SCORE = 0.5
OWL_MIN_SCORE = 0.1
OWL_TEMPLATE = "a photo of a {}"
OWL_ID = "google/owlvit-base-patch32"
CLIP_ID = "openai/clip-vit-large-patch14"
COCO_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"
IMAGE_SIZE = 512
N_BOOT = 2000
BOOT_SEED = 0

SPLITS_DIR = Path(__file__).resolve().parent.parent / "splits"


# ------------------------------------------------------------------ splits and labels

def load_split(name: str, splits_dir=None) -> pd.DataFrame:
    """A split CSV from M2/splits (beside this file's folder), or `splits_dir`."""
    for base in [Path(splits_dir)] if splits_dir else [SPLITS_DIR, Path("M2/splits"), Path("milestone-2/splits")]:
        if (base / f"{name}.csv").exists():
            return pd.read_csv(base / f"{name}.csv")
    import m2_protocol  # falls back to the protocol's own lookup (and its GitHub download)
    return m2_protocol.load_split(name)


GROUNDTRUTH = {"dev_500": "groundtruth_dev_500", "eval_2000": "groundtruth_eval_2000",
               "eval_500": "groundtruth_eval_2000"}


def load_targets(split: str, splits_dir=None) -> pd.DataFrame:
    """One row per labelled object: key, image_id, category, target_id.

    COCO splits: the `target` rows of the split's groundtruth file (eval_500 uses eval_2000's labels,
    restricted to its 500 captions). stress_prompts: every object named in its `objects` column."""
    if split == "stress_prompts":
        sp = load_split("stress_prompts", splits_dir)
        rows = []
        for r in sp.itertuples(index=False):
            for j, obj in enumerate(json.loads(r.objects)):
                rows.append({"key": r.prompt_id, "image_id": -1, "category": obj, "target_id": f"{r.prompt_id}:{j}:{obj}"})
        return pd.DataFrame(rows)
    if split not in GROUNDTRUTH:
        raise ValueError(f"no object labels for split {split!r}")
    gt = load_split(GROUNDTRUTH[split], splits_dir)
    gt = gt[gt.target.astype(str).str.lower() == "true"].copy()
    if split == "eval_500":
        gt = gt[gt.caption_idx.isin(load_split("eval_500", splits_dir).caption_idx)]
    gt["key"] = gt.caption_idx.astype(str)
    gt["target_id"] = gt.caption_idx.astype(str) + ":" + gt.category + ":" + gt.word_start.astype(str)
    return gt[["key", "image_id", "category", "target_id"]].reset_index(drop=True)


def prompts_for(split: str, splits_dir=None) -> dict:
    """key (as a string) -> the prompt it was painted from."""
    df = load_split("eval_2000" if split == "eval_500" else split, splits_dir)
    if split == "stress_prompts":
        return dict(zip(df.prompt_id.astype(str), df.prompt))
    return dict(zip(df.caption_idx.astype(str), df.caption))


# ------------------------------------------------------------------ image logs

def read_logs(logs) -> pd.DataFrame:
    """Combine one or more (images.csv, images_dir) pairs into one table with an absolute `path`.
    A repeated (config_id, split, key, seed_id) keeps its last row; rows whose file is missing are
    dropped with a warning."""
    frames = []
    for images_csv, images_dir in logs:
        df = pd.read_csv(images_csv, dtype={"key": str})
        df["path"] = [str(Path(images_dir) / f) for f in df.file]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["flagged"] = df.flagged.astype(int)
    df = df.drop_duplicates(["config_id", "split", "key", "seed_id"], keep="last")
    exists = df.path.map(os.path.exists)
    if (~exists).any():
        print(f"warning: {int((~exists).sum())} logged images not found on disk, left out")
    return df[exists].reset_index(drop=True)


def center_square(im):
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def load_rgb(path, size: int = IMAGE_SIZE):
    """An RGB image at size x size. Real photos get a centre square crop first."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = center_square(im).resize((size, size), Image.BICUBIC)
    return im


def download_coco(image_ids, out_dir, workers: int = 16) -> Path:
    """COCO val2017 photos by id into out_dir/<id>.jpg (skips files already there)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def get(i):
        p = out / f"{int(i)}.jpg"
        if not p.exists():
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(COCO_URL.format(int(i)), tmp)
            tmp.rename(p)
        return p

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, sorted(set(int(i) for i in image_ids))))
    print(f"{len(set(image_ids))} photos in {out}")
    return out


# ------------------------------------------------------------------ models

def _tensor(out, *names):
    import torch
    if torch.is_tensor(out):
        return out
    for n in names:
        if getattr(out, n, None) is not None:
            return getattr(out, n)
    raise TypeError(f"unexpected model output {type(out)}")


class Scorers:
    """The scoring networks, each loaded the first time it is needed and then kept on the GPU."""

    def __init__(self, device: str = "cuda"):
        import torch
        self.torch, self.device, self._m = torch, device, {}

    def _load(self, name):
        if name in self._m:
            return self._m[name]
        d = self.device
        if name in ("inception", "inception_logits"):
            from torchmetrics.image.fid import NoTrainInceptionV3
            feat = "2048" if name == "inception" else "logits_unbiased"
            m = NoTrainInceptionV3(name="inception-v3-compat", features_list=[feat]).to(d).eval()
        elif name == "clip":
            from transformers import CLIPModel, CLIPProcessor
            m = (CLIPModel.from_pretrained(CLIP_ID).to(d).eval(), CLIPProcessor.from_pretrained(CLIP_ID))
        elif name == "frcnn":
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
            w = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            m = (fasterrcnn_resnet50_fpn_v2(weights=w).to(d).eval(), w.transforms(),
                 {c: i for i, c in enumerate(w.meta["categories"]) if c not in ("N/A", "__background__")})
        elif name == "owl":
            from transformers import OwlViTForObjectDetection, OwlViTProcessor
            m = (OwlViTForObjectDetection.from_pretrained(OWL_ID).to(d).eval(), OwlViTProcessor.from_pretrained(OWL_ID))
        elif name == "lpips":
            import lpips
            m = lpips.LPIPS(net="alex", verbose=False).to(d).eval()
        self._m[name] = m
        return m

    def _inception(self, name, ims) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(np.stack([np.array(im) for im in ims])).permute(0, 3, 1, 2).to(self.device)
        with torch.no_grad():
            f = self._load(name)(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        return f.reshape(len(ims), -1).double().cpu().numpy()

    def inception_features(self, ims) -> np.ndarray:
        """2,048 pool features (FID and KID)."""
        return self._inception("inception", ims)

    def inception_logits(self, ims) -> np.ndarray:
        """1,008 class logits (Inception Score)."""
        return self._inception("inception_logits", ims).astype(np.float32)

    def clip_embed(self, ims, texts=None):
        """Unit image embeddings, and unit text embeddings when `texts` is given."""
        torch = self.torch
        with torch.no_grad():
            clip, proc = self._load("clip")
            px = proc(images=ims, return_tensors="pt")["pixel_values"].to(self.device)
            img = _tensor(clip.get_image_features(pixel_values=px), "image_embeds", "pooler_output")
            img = torch.nn.functional.normalize(img.float(), dim=-1)
            if texts is None:
                return img.cpu().numpy(), None
            t = proc(text=list(texts), return_tensors="pt", padding=True, truncation=True, max_length=77)
            txt = _tensor(clip.get_text_features(input_ids=t["input_ids"].to(self.device),
                                                      attention_mask=t["attention_mask"].to(self.device)),
                          "text_embeds", "pooler_output")
            txt = torch.nn.functional.normalize(txt.float(), dim=-1)
        return img.cpu().numpy(), txt.cpu().numpy()

    def frcnn_scores(self, im, categories) -> dict:
        """Highest Faster R-CNN v2 score for each category (NaN for a non-COCO category)."""
        torch = self.torch
        from torchvision.transforms.functional import pil_to_tensor
        frcnn, tf, classes = self._load("frcnn")
        with torch.no_grad():
            out = frcnn([tf(pil_to_tensor(im)).to(self.device)])[0]
        labels, scores = out["labels"].cpu().numpy(), out["scores"].cpu().numpy()
        res = {}
        for c in categories:
            if c not in classes:
                res[c] = float("nan")
                continue
            s = scores[labels == classes[c]]
            res[c] = float(s.max()) if len(s) else 0.0
        return res

    def owl_scores(self, im, categories) -> dict:
        """Highest OWL-ViT score of "a photo of a {class}" over all boxes, for each category. OWL-ViT
        scores each query on its own (a sigmoid per query), so this equals querying one class at a time."""
        torch = self.torch
        cats = list(dict.fromkeys(categories))
        owl, proc = self._load("owl")
        inputs = proc(text=[[OWL_TEMPLATE.format(c) for c in cats]], images=im, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = owl(**inputs).logits[0]          # (boxes, queries)
        best = torch.sigmoid(logits).max(dim=0).values.cpu().numpy()
        return {c: float(best[j]) for j, c in enumerate(cats)}

    def lpips(self, a, b) -> float:
        torch = self.torch
        net = self._load("lpips")
        t = lambda im: (torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float() / 127.5 - 1).to(self.device)  # noqa: E731
        with torch.no_grad():
            return float(net(t(a), t(b)).item())


# ------------------------------------------------------------------ scoring images

def _name(split, config_id):
    return f"{split}__{config_id}"


def judge(scorers: Scorers, im, cats) -> list[dict]:
    fr, ow = scorers.frcnn_scores(im, cats), scorers.owl_scores(im, cats)
    rows = []
    for c in cats:
        f_found = bool(fr[c] >= FRCNN_MIN_SCORE) if not np.isnan(fr[c]) else False
        o_found = bool(ow[c] >= OWL_MIN_SCORE)
        rows.append({"category": c, "frcnn_score": fr[c], "owl_score": ow[c], "frcnn_found": int(f_found),
                     "owl_found": int(o_found), "present": int(f_found or o_found)})
    return rows


def score_config(imgs: pd.DataFrame, out_dir, scorers: Scorers, splits_dir=None, batch: int = 16,
                 overwrite: bool = False) -> None:
    """Score every image of one (config_id, split): CLIPScore, both judges per labelled object, and
    the Inception and CLIP features kept for FID, KID and diversity. Incremental: images already
    scored (the same key and seed_id) are kept, so adding seeds to a setting scores only the new ones."""
    (cfg,), (split,) = imgs.config_id.unique(), imgs.split.unique()
    out = Path(out_dir)
    for d in ("images", "objects", "features"):
        (out / d).mkdir(parents=True, exist_ok=True)
    name = _name(split, cfg)
    files = [out / "images" / f"{name}.csv", out / "objects" / f"{name}.csv", out / "features" / f"{name}.npz"]
    prior = None
    if not overwrite and all(f.exists() for f in files):
        prior_img = pd.read_csv(files[0], dtype={"key": str})
        done = set(zip(prior_img.key.astype(str), prior_img.seed_id.astype(int)))
        todo = [(str(k), int(sd)) not in done for k, sd in zip(imgs.key, imgs.seed_id)]
        if not any(todo):
            print(f"{name}: already scored")
            return
        try:
            prior_obj = pd.read_csv(files[1], dtype={"key": str})
        except pd.errors.EmptyDataError:
            prior_obj = pd.DataFrame()
        prior = (prior_img, prior_obj, dict(np.load(files[2])))
        imgs = imgs[todo]
        print(f"{name}: {len(done)} images already scored, scoring {len(imgs)} new ones")
    targets = load_targets(split, splits_dir)
    by_key = targets.groupby("key")
    prompts = prompts_for(split, splits_dir)
    imgs = imgs.sort_values(["seed_id", "key"]).reset_index(drop=True)
    img_rows, obj_rows, feats = [], [], {"key": [], "seed_id": [], "inception": [], "clip": []}
    ok = imgs[imgs.flagged == 0]
    for r in imgs[imgs.flagged == 1].itertuples(index=False):
        img_rows.append({"config_id": cfg, "split": split, "key": r.key, "seed_id": r.seed_id, "flagged": 1,
                         "seconds": r.seconds, "clipscore": np.nan})
    for start in range(0, len(ok), batch):
        chunk = ok.iloc[start:start + batch]
        ims = [load_rgb(p) for p in chunk.path]
        inc = scorers.inception_features(ims)
        img_e, txt_e = scorers.clip_embed(ims, [prompts[str(k)] for k in chunk.key])
        cs = 100 * np.clip((img_e * txt_e).sum(1), 0, None)
        for i, r in enumerate(chunk.itertuples(index=False)):
            img_rows.append({"config_id": cfg, "split": split, "key": r.key, "seed_id": r.seed_id, "flagged": 0,
                             "seconds": r.seconds, "clipscore": float(cs[i])})
            feats["key"].append(str(r.key)); feats["seed_id"].append(int(r.seed_id))
            feats["inception"].append(inc[i]); feats["clip"].append(img_e[i])
            if str(r.key) in by_key.groups:
                t = by_key.get_group(str(r.key))
                verdict = {v["category"]: v for v in judge(scorers, ims[i], list(t.category))}
                for tr in t.itertuples(index=False):
                    obj_rows.append({"config_id": cfg, "split": split, "key": r.key, "seed_id": r.seed_id,
                                     "target_id": tr.target_id, **verdict[tr.category]})
        if (start // batch) % 10 == 0:
            print(f"{name}: {min(start + batch, len(ok))}/{len(ok)} images", flush=True)
    n_new = len(feats["key"])
    new = {"key": np.array(feats["key"], dtype=str), "seed_id": np.array(feats["seed_id"], dtype=int),
           "inception": np.array(feats["inception"]) if n_new else np.zeros((0, 2048)),
           "clip": np.array(feats["clip"], dtype=np.float32) if n_new else np.zeros((0, 768), np.float32)}
    img_df, obj_df = pd.DataFrame(img_rows), pd.DataFrame(obj_rows)
    if prior is not None:
        img_df = pd.concat([prior[0], img_df], ignore_index=True)
        obj_df = pd.concat([prior[1], obj_df], ignore_index=True)
        new = {k: np.concatenate([prior[2][k], new[k]]) if n_new else prior[2][k] for k in new}
    img_df.to_csv(files[0], index=False)
    obj_df.to_csv(files[1], index=False)
    np.savez_compressed(files[2], **new)
    print(f"{name}: {len(imgs)} images, {len(ok)} scored, {len(imgs) - len(ok)} flagged and left out, {len(obj_rows)} object verdicts")


def score_log(imgs: pd.DataFrame, out_dir, scorers: Scorers, configs=None, splits=None, **kw) -> None:
    """score_config for every (config_id, split) in an image table, or only the listed ones."""
    for (cfg, split), g in imgs.groupby(["config_id", "split"], sort=False):
        if (configs is None or cfg in configs) and (splits is None or split in splits):
            score_config(g, out_dir, scorers, **kw)


def reference_features(ref_split: str, photo_dir, out_dir, scorers: Scorers, splits_dir=None, batch: int = 32) -> Path:
    """Inception features of the real reference photos, computed once and cached."""
    path = Path(out_dir) / "features" / f"reference__{ref_split}.npz"
    if path.exists():
        print(f"reference features cached: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = load_split(ref_split, splits_dir).image_id.astype(int).tolist()
    feats = []
    for s in range(0, len(ids), batch):
        feats.append(scorers.inception_features([load_rgb(Path(photo_dir) / f"{i}.jpg") for i in ids[s:s + batch]]))
    np.savez_compressed(path, image_id=np.array(ids), inception=np.concatenate(feats))
    print(f"reference features: {len(ids)} photos -> {path}")
    return path


def score_real_photos(split: str, photo_dir, out_dir, scorers: Scorers, splits_dir=None) -> Path:
    """Both judges on the real photos behind a split's captions: the recall a perfect generator
    would get under these judges (the ceiling), and a check of the judges themselves."""
    path = Path(out_dir) / "objects" / f"{split}__real_photo.csv"
    if path.exists():
        print(f"real-photo verdicts cached: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    targets = load_targets(split, splits_dir)
    rows = []
    for (key, image_id), t in targets.groupby(["key", "image_id"], sort=False):
        im = load_rgb(Path(photo_dir) / f"{int(image_id)}.jpg")
        verdict = {v["category"]: v for v in judge(scorers, im, list(t.category))}
        for tr in t.itertuples(index=False):
            rows.append({"config_id": "real_photo", "split": split, "key": key, "seed_id": 0,
                         "target_id": tr.target_id, **verdict[tr.category]})
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"real photos of {split}: {len(rows)} object verdicts -> {path}")
    return path


def score_diversity(imgs: pd.DataFrame, out_dir, scorers: Scorers, seed_a: int = 0, seed_b: int = 1) -> None:
    """LPIPS and CLIP image distance between seed_id 0 and seed_id 1 of the same caption."""
    out = Path(out_dir) / "diversity"
    out.mkdir(parents=True, exist_ok=True)
    for (cfg, split), g in imgs.groupby(["config_id", "split"], sort=False):
        a = g[(g.seed_id == seed_a) & (g.flagged == 0)].set_index("key")
        b = g[(g.seed_id == seed_b) & (g.flagged == 0)].set_index("key")
        keys = [k for k in b.index if k in a.index]
        if not keys:
            continue
        path = out / f"{_name(split, cfg)}.csv"
        if path.exists():
            print(f"{_name(split, cfg)}: diversity already scored")
            continue
        rows = []
        for k in keys:
            ia, ib = load_rgb(a.loc[k, "path"]), load_rgb(b.loc[k, "path"])
            ea, _ = scorers.clip_embed([ia, ib])
            rows.append({"config_id": cfg, "split": split, "key": k, "lpips": scorers.lpips(ia, ib),
                         "clip_distance": float(1 - ea[0] @ ea[1])})
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"{_name(split, cfg)}: diversity on {len(rows)} caption pairs")


def score_inception_logits(imgs: pd.DataFrame, out_dir, scorers: Scorers, batch: int = 32) -> None:
    """Inception class logits of every unflagged image, for the Inception Score. Incremental per
    (config_id, split), like score_config, so it can be run after scoring that predates it."""
    out = Path(out_dir) / "logits"
    out.mkdir(parents=True, exist_ok=True)
    for (cfg, split), g in imgs[imgs.flagged == 0].groupby(["config_id", "split"], sort=False):
        path = out / f"{_name(split, cfg)}.npz"
        keys, seeds, logits = np.array([], dtype=str), np.array([], dtype=int), np.zeros((0, 1008), np.float32)
        if path.exists():
            z = np.load(path)
            keys, seeds, logits = z["key"].astype(str), z["seed_id"], z["logits"]
        done = set(zip(keys, seeds.astype(int)))
        g = g[[(str(k), int(sd)) not in done for k, sd in zip(g.key, g.seed_id)]]
        if g.empty:
            print(f"{_name(split, cfg)}: logits already computed")
            continue
        new = [scorers.inception_logits([load_rgb(p) for p in g.path.iloc[s:s + batch]]) for s in range(0, len(g), batch)]
        keys = np.concatenate([keys, np.array([str(k) for k in g.key], dtype=str)])
        seeds = np.concatenate([seeds, np.array(g.seed_id, dtype=int)])
        logits = np.concatenate([logits] + new)
        np.savez_compressed(path, key=keys, seed_id=seeds, logits=logits)
        print(f"{_name(split, cfg)}: logits for {len(g)} images")


def real_photo_features(split: str, photo_dir, out_dir, scorers: Scorers, splits_dir=None, batch: int = 32) -> Path:
    """Inception pool features and class logits of the real photos behind a split's captions (one
    photo per caption, none in ref_2000_v2), keyed by caption_idx: the real-photo floor for FID, KID
    and the Inception Score. Cached."""
    path = Path(out_dir) / "features" / f"{split}__real_photo.npz"
    if path.exists():
        print(f"real-photo features cached: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_split(split, splits_dir)
    pool, logits = [], []
    for s in range(0, len(df), batch):
        ims = [load_rgb(Path(photo_dir) / f"{int(i)}.jpg") for i in df.image_id.iloc[s:s + batch]]
        pool.append(scorers.inception_features(ims))
        logits.append(scorers.inception_logits(ims))
    np.savez_compressed(path, key=np.array([str(k) for k in df.caption_idx], dtype=str), image_id=np.array(df.image_id, dtype=int),
                        inception=np.concatenate(pool), logits=np.concatenate(logits))
    print(f"real-photo features of {split}: {len(df)} photos -> {path}")
    return path


# ------------------------------------------------------------------ statistics (CPU only)

def _rng():
    return np.random.default_rng(BOOT_SEED)


def _ci(samples) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _boot_index(n: int, n_boot: int = N_BOOT) -> np.ndarray:
    return _rng().integers(0, n, size=(n_boot, n))


def ratio_with_ci(num: np.ndarray, den: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """sum(num) / sum(den) with a bootstrap over captions (one entry per caption)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = _boot_index(len(num), n_boot)
    b = num[idx].sum(1) / den[idx].sum(1)
    return float(num.sum() / den.sum()), *_ci(b)


def mean_with_ci(x, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    return float(x.mean()), *_ci(x[_boot_index(len(x), n_boot)].mean(1))


def load_objects(score_dir, split=None) -> pd.DataFrame:
    fs = sorted((Path(score_dir) / "objects").glob("*.csv"))
    df = pd.concat([pd.read_csv(f, dtype={"key": str}) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    return df if split is None or df.empty else df[df.split == split]


def load_images(score_dir, split=None) -> pd.DataFrame:
    fs = sorted((Path(score_dir) / "images").glob("*.csv"))
    df = pd.concat([pd.read_csv(f, dtype={"key": str}) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    return df if split is None or df.empty else df[df.split == split]


def _seeds(seed_id, seed_ids):
    return tuple(seed_ids) if seed_ids is not None else (seed_id,)


def load_features(score_dir, split, config_id, seed_id: int = 0, seed_ids=None):
    """(keys, Inception features) of one config's unflagged images for one seed_id. With `seed_ids`,
    (keys, seed_ids, features) for all the listed seeds."""
    z = np.load(Path(score_dir) / "features" / f"{_name(split, config_id)}.npz")
    m = np.isin(z["seed_id"], _seeds(seed_id, seed_ids))
    if seed_ids is None:
        return z["key"][m].astype(str), z["inception"][m]
    return z["key"][m].astype(str), z["seed_id"][m].astype(int), z["inception"][m]


def load_logits(score_dir, split, config_id, seed_ids=(0,)):
    """(keys, seed_ids, Inception logits) of one config's unflagged images."""
    z = np.load(Path(score_dir) / "logits" / f"{_name(split, config_id)}.npz")
    m = np.isin(z["seed_id"], seed_ids)
    return z["key"][m].astype(str), z["seed_id"][m].astype(int), z["logits"][m]


def load_reference(score_dir, ref_split: str = "ref_2000_v2") -> np.ndarray:
    return np.load(Path(score_dir) / "features" / f"reference__{ref_split}.npz")["inception"]


def load_real(score_dir, split, keys=None):
    """(keys, Inception features, logits) of the real photos behind a split's captions."""
    z = np.load(Path(score_dir) / "features" / f"{split}__real_photo.npz")
    k = z["key"].astype(str)
    m = np.isin(k, list(map(str, keys))) if keys is not None else np.ones(len(k), bool)
    return k[m], z["inception"][m], z["logits"][m]


def _cluster_counts(groups, n_boot: int = N_BOOT) -> np.ndarray:
    """Bootstrap draw counts per item when whole groups (captions) are resampled together."""
    uniq, gidx = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    return _counts(len(uniq), n_boot)[:, gidx]


def recall(obj: pd.DataFrame, config_id, split, keys=None, seed_id: int = 0, col: str = "present", seed_ids=None):
    """Object recall (share of labelled objects judged present) with its caption-bootstrap interval.
    With several seeds, every seed's verdicts count and a caption's seeds are resampled together."""
    d = obj[(obj.config_id == config_id) & (obj.split == split) & obj.seed_id.isin(_seeds(seed_id, seed_ids))]
    if keys is not None:
        d = d[d.key.isin(set(map(str, keys)))]
    g = d.groupby("key")[col].agg(["sum", "count"])
    return (*ratio_with_ci(g["sum"].values, g["count"].values), int(g["count"].sum()))


def paired_recall_diff(obj: pd.DataFrame, cfg_a, cfg_b, split, keys=None, seed_id: int = 0, n_boot: int = N_BOOT,
                       seed_ids=None):
    """recall(a) - recall(b) on the objects scored under both settings (same captions, same noise)."""
    d = obj[(obj.split == split) & obj.seed_id.isin(_seeds(seed_id, seed_ids)) & obj.config_id.isin([cfg_a, cfg_b])]
    if keys is not None:
        d = d[d.key.isin(set(map(str, keys)))]
    w = d.pivot_table(index=["key", "seed_id", "target_id"], columns="config_id", values="present").dropna()
    w = w.reset_index()
    g = w.assign(diff=w[cfg_a] - w[cfg_b]).groupby("key")["diff"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    idx = _boot_index(len(s), n_boot)
    b = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), *_ci(b), int(c.sum())


def fid(f_gen: np.ndarray, f_ref: np.ndarray) -> float:
    from scipy import linalg
    mu1, mu2 = f_gen.mean(0), f_ref.mean(0)
    s1, s2 = np.cov(f_gen, rowvar=False), np.cov(f_ref, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2)
    covmean = (covmean[0] if isinstance(covmean, tuple) else covmean).real   # older SciPy may return a tuple
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def _poly(a, b):
    return (a @ b.T / a.shape[1] + 1) ** 3


class KID:
    """Unbiased KID (polynomial kernel, as in Binkowski et al. 2018) of generated features against
    a fixed reference set, x 1000, with a bootstrap over generated images. Same-image pairs created
    by resampling are left out of the generated-generated term, so the estimate stays unbiased."""

    def __init__(self, f_gen: np.ndarray, f_ref: np.ndarray):
        k_rr = _poly(f_ref, f_ref)
        m = len(f_ref)
        self.c_rr = (k_rr.sum() - np.trace(k_rr)) / (m * (m - 1))
        self.k_gg = _poly(f_gen, f_gen)
        self.row_gr = _poly(f_gen, f_ref).mean(1)
        self.n = len(f_gen)

    def at(self, counts: np.ndarray) -> np.ndarray:
        """KID for each row of `counts` (how often each generated image is drawn)."""
        counts = np.atleast_2d(counts).astype(float)
        diag = np.diag(self.k_gg)
        tot = np.einsum("bi,ij,bj->b", counts, self.k_gg, counts)
        same = (counts ** 2) @ diag
        pairs = counts.sum(1) ** 2 - (counts ** 2).sum(1)
        gg = (tot - same) / pairs
        gr = counts @ self.row_gr / counts.sum(1)
        return 1000 * (gg + self.c_rr - 2 * gr)

    def value(self) -> float:
        return float(self.at(np.ones(self.n))[0])


def _counts(n: int, n_boot: int = N_BOOT) -> np.ndarray:
    idx = _boot_index(n, n_boot)
    c = np.zeros((n_boot, n))
    np.add.at(c, (np.arange(n_boot)[:, None], idx), 1)
    return c


def kid_with_ci(f_gen, f_ref, n_boot: int = N_BOOT, groups=None):
    """KID x 1000 with its bootstrap interval; `groups` (one caption per image) resamples a caption's
    seeds together."""
    k = KID(f_gen, f_ref)
    c = _counts(k.n, n_boot) if groups is None else _cluster_counts(groups, n_boot)
    return k.value(), *_ci(k.at(c))


def _aligned(score_dir, split, cfg_a, cfg_b, keys=None, seed_ids=None):
    """Features of the (caption, seed) pairs both settings painted, in the same order."""
    if seed_ids is None:
        ka, fa = load_features(score_dir, split, cfg_a)
        kb, fb = load_features(score_dir, split, cfg_b)
        sa, sb = np.zeros(len(ka), int), np.zeros(len(kb), int)
    else:
        ka, sa, fa = load_features(score_dir, split, cfg_a, seed_ids=seed_ids)
        kb, sb, fb = load_features(score_dir, split, cfg_b, seed_ids=seed_ids)
    ia = {(k, int(sd)): i for i, (k, sd) in enumerate(zip(ka, sa))}
    ib = {(k, int(sd)): i for i, (k, sd) in enumerate(zip(kb, sb))}
    allowed = set(map(str, keys)) if keys is not None else None
    common = sorted(p for p in set(ia) & set(ib) if allowed is None or p[0] in allowed)
    return fa[[ia[p] for p in common]], fb[[ib[p] for p in common]], common


def paired_kid_diff(score_dir, split, cfg_a, cfg_b, f_ref, keys=None, n_boot: int = N_BOOT, seed_ids=None):
    """KID(a) - KID(b) on the captions both settings painted, resampling the same captions on both
    sides (with all their seeds when `seed_ids` lists several)."""
    fa, fb, common = _aligned(score_dir, split, cfg_a, cfg_b, keys, seed_ids)
    ka, kb = KID(fa, f_ref), KID(fb, f_ref)
    c = _counts(len(common), n_boot) if seed_ids is None else _cluster_counts([p[0] for p in common], n_boot)
    return ka.value() - kb.value(), *_ci(ka.at(c) - kb.at(c)), len(common)


def paired_mean_diff(df: pd.DataFrame, cfg_a, cfg_b, col: str, keys=None, n_boot: int = N_BOOT, seed_ids=None):
    """mean(a) - mean(b) of a per-image column (e.g. clipscore) over images present on both sides."""
    if keys is not None:
        df = df[df.key.isin(set(map(str, keys)))]
    if seed_ids is None:
        w = df[df.config_id.isin([cfg_a, cfg_b])].pivot_table(index="key", columns="config_id", values=col).dropna()
        d = (w[cfg_a] - w[cfg_b]).values
        return (*mean_with_ci(d, n_boot), len(d))
    df = df[df.seed_id.isin(seed_ids)]
    w = df[df.config_id.isin([cfg_a, cfg_b])].pivot_table(index=["key", "seed_id"], columns="config_id", values=col).dropna()
    w = w.reset_index().assign(diff=lambda t: t[cfg_a] - t[cfg_b])
    g = w.groupby("key")["diff"].agg(["sum", "count"])
    return (*ratio_with_ci(g["sum"].values, g["count"].values, n_boot), int(g["count"].sum()))


def _softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


class InceptionScore:
    """exp(mean_x KL(p(y|x) || p(y))) over a set of images, with weighted (bootstrap) versions."""

    def __init__(self, logits: np.ndarray):
        self.p = _softmax(np.asarray(logits, dtype=np.float64))
        self.negent = (self.p * np.log(np.clip(self.p, 1e-30, None))).sum(1)
        self.n = len(self.p)

    def at(self, counts: np.ndarray) -> np.ndarray:
        counts = np.atleast_2d(counts).astype(float)
        tot = counts.sum(1, keepdims=True)
        py = counts @ self.p / tot
        kl = (counts @ self.negent) / tot[:, 0] - (py * np.log(np.clip(py, 1e-30, None))).sum(1)
        return np.exp(kl)

    def value(self) -> float:
        return float(self.at(np.ones(self.n))[0])


def _basic_ci(theta: float, samples) -> tuple[float, float]:
    """Basic (pivot) bootstrap interval, 2*theta - [q97.5, q2.5]. Used for the Inception Score:
    a resample repeats some images, which the score reads as less variety, so resampled values sit
    below the full-set value and the plain percentile interval would be shifted down."""
    lo, hi = _ci(samples)
    return 2 * theta - hi, 2 * theta - lo


def is_with_ci(logits, n_boot: int = N_BOOT, groups=None):
    """Inception Score of one set of images, without an interval (returned as NaN, like FID's).
    A bootstrap resample repeats images, which the score reads as lost variety, and at a few hundred
    images that shift is larger than the spread, so neither the percentile nor the basic interval
    covers the score reliably. Compare settings with paired_is_diff, where the shift falls on both
    sides. Like FID, the score grows with the number of images: compare only sets of the same size."""
    return InceptionScore(logits).value(), float("nan"), float("nan")


def paired_is_diff(score_dir, split, cfg_a, cfg_b, keys=None, seed_ids=(0,), n_boot: int = N_BOOT):
    """IS(a) - IS(b) on the (caption, seed) pairs both settings painted, resampling captions."""
    ka, sa, la = load_logits(score_dir, split, cfg_a, seed_ids)
    kb, sb, lb = load_logits(score_dir, split, cfg_b, seed_ids)
    ia = {(k, int(sd)): i for i, (k, sd) in enumerate(zip(ka, sa))}
    ib = {(k, int(sd)): i for i, (k, sd) in enumerate(zip(kb, sb))}
    allowed = set(map(str, keys)) if keys is not None else None
    common = sorted(p for p in set(ia) & set(ib) if allowed is None or p[0] in allowed)
    A, B = InceptionScore(la[[ia[p] for p in common]]), InceptionScore(lb[[ib[p] for p in common]])
    c = _cluster_counts([p[0] for p in common], n_boot)
    d = A.value() - B.value()
    return d, *_basic_ci(d, A.at(c) - B.at(c)), len(common)


def real_floor(score_dir, split, f_ref, keys=None, n_boot: int = N_BOOT) -> dict:
    """FID, KID (with interval) and Inception Score of the real photos behind a split's captions
    against ref_2000_v2: the best a generator could score with this many images."""
    k, f, lg = load_real(score_dir, split, keys)
    kv, klo, khi = kid_with_ci(f, f_ref, n_boot)
    iv, ilo, ihi = is_with_ci(lg, n_boot)
    return {"split": split, "n": len(k), "fid": fid(f, f_ref), "kid": kv, "kid_lo": klo, "kid_hi": khi,
            "is": iv, "is_lo": ilo, "is_hi": ihi}


def judge_agreement(obj: pd.DataFrame) -> dict:
    """How often the two judges agree, and Cohen's kappa, over COCO-class objects."""
    d = obj.dropna(subset=["frcnn_score"])
    a, b = d.frcnn_found.values.astype(int), d.owl_found.values.astype(int)
    po = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return {"n": int(len(d)), "agree": po, "kappa": float((po - pe) / (1 - pe)) if pe < 1 else float("nan"),
            "frcnn_only": float(((a == 1) & (b == 0)).mean()), "owl_only": float(((a == 0) & (b == 1)).mean())}


def summary_table(score_dir, split=None, ref_split: str = "ref_2000_v2", keys=None, configs=None) -> pd.DataFrame:
    """One row per (split, config), seed_id 0: images, flagged, seconds per image, object recall,
    CLIPScore, KID and FID (each with its interval where it has one), the Inception Score where its
    logits exist, diversity where seed_id 1 pairs exist, and the real-photo floor at the same number
    of captions where the real photos' features exist."""
    obj, ims = load_objects(score_dir, split), load_images(score_dir, split)
    f_ref = load_reference(score_dir, ref_split)
    div_files = {f.stem: f for f in (Path(score_dir) / "diversity").glob("*.csv")}
    rows = []
    for (sp, cfg), g in ims.groupby(["split", "config_id"]):
        if configs is not None and cfg not in configs:
            continue
        g0 = g[g.seed_id == 0]
        if keys is not None:
            g0 = g0[g0.key.isin(set(map(str, keys)))]
        ok = g0[g0.flagged == 0]
        r, rlo, rhi, n_obj = recall(obj, cfg, sp, keys)
        c, clo, chi = mean_with_ci(ok.clipscore)
        k_keys, f = load_features(score_dir, sp, cfg)
        if keys is not None:
            m = np.isin(k_keys, list(map(str, keys)))
            f, k_keys = f[m], k_keys[m]
        kv, klo, khi = kid_with_ci(f, f_ref)
        row = {"split": sp, "config_id": cfg, "images": len(g0), "flagged": int(g0.flagged.sum()),
               "sec_per_image": float(g0.seconds.mean()), "objects": n_obj, "recall": r, "recall_lo": rlo,
               "recall_hi": rhi, "clipscore": c, "clipscore_lo": clo, "clipscore_hi": chi, "kid": kv, "kid_lo": klo,
               "kid_hi": khi, "fid": fid(f, f_ref)}
        name = _name(sp, cfg)
        if (Path(score_dir) / "logits" / f"{name}.npz").exists():
            lk, _, lg = load_logits(score_dir, sp, cfg)
            if keys is not None:
                lg = lg[np.isin(lk, list(map(str, keys)))]
            row["is"], row["is_lo"], row["is_hi"] = is_with_ci(lg)
        real_split = "eval_500" if sp == "eval_2000" else sp
        if (Path(score_dir) / "features" / f"{real_split}__real_photo.npz").exists():
            fl = real_floor(score_dir, real_split, f_ref, keys=list(k_keys))
            row.update({"floor_n": fl["n"], "floor_fid": fl["fid"], "floor_kid": fl["kid"], "floor_is": fl["is"]})
        if name in div_files:
            dv = pd.read_csv(div_files[name], dtype={"key": str})
            for col in ("lpips", "clip_distance"):
                row[col], row[col + "_lo"], row[col + "_hi"] = mean_with_ci(dv[col])
            row["diversity_pairs"] = len(dv)
        rows.append(row)
    return pd.DataFrame(rows)
