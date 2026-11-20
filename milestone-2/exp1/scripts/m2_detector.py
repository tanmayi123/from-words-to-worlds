"""Milestone 2, early warning: how early can a lost object be seen coming, and from what?

Pre-registered in docs/m2_detector_preregistration.md before any recording existed. Every number
this script reports is read by the rules fixed there.

Development paintings: eval_2000 rows 0 to 999 with labelled objects, seed_ids 0, 1 and 2,
repainted from the word-timing runs with a fuller recorder; each must reproduce its stored picture.
Prospective paintings: rows 1000 to 1999 with labelled objects, seed_ids 3, 4 and 5, made new.

Phases, each resumable:
  record development|prospective   paint with the group anchor and keep, for every painting,
                                   every token's cross-attention at all 30 steps (pooled to
                                   16 x 16), and after each decision step the latent itself and
                                   the model's guess of the finished latent.
  judge prospective                the group judges on the finished prospective pictures.
  preview development|prospective  decode every kept one-step guess, make a quick draft from
                                   decision steps 0, 2, 4 and 8, and score them all with the same
                                   judges and with CLIP.
  features development|prospective one row per object and painting: prompt, attention, preview
                                   and draft features at every decision step, and the label.
  develop                          the nine detector families (and the boosted-tree check) under
                                   5-fold cross-validation with folds by caption.
  lock                             fit every detector on all development rows and seal them, once.
  test                             apply the sealed detectors to the prospective paintings; every
                                   run is logged. Builds the prospective features if needed.
  figures                          how early, within-object, and one example filmstrip.

Run from the project folder:
  python scripts/m2_detector.py record development
  python scripts/m2_detector.py record prospective
  python scripts/m2_detector.py judge prospective
  python scripts/m2_detector.py preview development
  python scripts/m2_detector.py preview prospective
  python scripts/m2_detector.py features development
  python scripts/m2_detector.py develop
  python scripts/m2_detector.py lock
  python scripts/m2_detector.py test
  python scripts/m2_detector.py figures
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_timing as mt  # noqa: E402
from w2w import m2_protocol as m2  # noqa: E402
from w2w.live import sentence_words  # noqa: E402
from w2w.paths import DATA, RESULTS  # noqa: E402

SPLIT = "eval_2000"
CFG = m2.GenConfig("ddim", "leading", 30, 7.5)
STEPS = CFG.steps
PARTS = {"development": {"rows": (0, 1000), "seed_ids": (0, 1, 2)},
         "prospective": {"rows": (1000, 2000), "seed_ids": (3, 4, 5)}}
DECISION_STEPS = (0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20)
DRAFT_STEPS, DRAFT_LEN = (0, 2, 4, 8), 5     # quick drafts: from these steps, 5 DPM-Solver++ steps to the end
POOL = 4                                   # the 64 x 64 attention grid, average-pooled to 16 x 16
OUT = RESULTS / "m2_detector"
REC = DATA / "m2_detector"
IMAGES = DATA / "m2_images"
STORED = RESULTS / "m2_timing"             # the word-timing runs
MAX_MISMATCH = 0.01                        # the share of development repaints allowed to differ
OWNER = "Santhosh"

ATT_NAMES = ("share", "peak", "focus", "overlap", "lead", "steady")
JUDGE_NAMES = ("frcnn", "retina", "owl")
SCORERS = JUDGE_NAMES + ("clip",)


# ------------------------------------------------------------------ where things go

def run_dir(part: str, seed_id: int) -> Path:
    d = OUT / "runs" / f"{part}__s{seed_id}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rec_path(seed_id: int, cap: int) -> Path:
    return REC / f"s{seed_id}" / f"{cap}.npz"


def image_path(cap: int, seed_id: int) -> Path:
    return IMAGES / SPLIT / CFG.config_id / f"{cap}_s{seed_id}.png"


def part_captions(part: str, limit: int | None = None) -> pd.DataFrame:
    """The part's rows of eval_2000 that have at least one labelled object, in split order."""
    lo, hi = PARTS[part]["rows"]
    rows = mt.load_split(SPLIT).iloc[lo:hi]
    rows = rows[rows.caption_idx.isin(set(mt.targets_for(SPLIT).caption_idx))]
    return rows.head(limit) if limit else rows


def append_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    mt.append_csv(path, rows, cols)


# ------------------------------------------------------------------ phase 1: record

OBJECT_COLUMNS = ["seed_id"] + mt.OBJECT_COLUMNS + ["obj_slots", "n_words", "n_objects"]
CAPTION_COLUMNS = ["caption_idx", "image_id", "seed_id", "seed", "flagged", "seconds", "n_objects",
                   "same_image", "max_pixel_diff", "same_measures"]


def pool_maps(maps: np.ndarray, keep: int) -> np.ndarray:
    """(steps, 77, 64, 64) attention to (steps, keep, 16, 16) float16, average-pooled."""
    t, _, h, w = maps.shape
    m = maps[:, :keep].astype(np.float32).reshape(t, keep, h // POOL, POOL, w // POOL, POOL)
    return m.mean(axis=(3, 5)).astype(np.float16)


def n_caption_tokens(tok, caption: str) -> int:
    """Caption tokens between the start and end tokens, as the 77-slot prompt holds them."""
    return len(tok(caption, truncation=True, max_length=77)["input_ids"]) - 2


def content_slots(tok, caption: str) -> list[int]:
    return [s for w in sentence_words(tok, caption) if w["content"] for s in w["slots"]]


def stored_measures(seed_id: int) -> pd.DataFrame | None:
    p = STORED / f"{SPLIT}__{CFG.config_id}__s{seed_id}" / "objects.csv"
    return pd.read_csv(p).set_index(mt.OBJECT_KEY) if p.exists() else None


def record(args) -> None:
    import torch
    from PIL import Image
    from w2w.rewind import Painter
    part = args.part
    caps = part_captions(part, args.limit)
    targets = mt.targets_for(SPLIT)
    by_caption = {k: g for k, g in targets.groupby("caption_idx")}
    pipe = m2.load_pipeline(args.device)
    painter = Painter(pipe, args.device, STEPS, CFG.guidance, spacing=CFG.schedule, cpu_noise=True)
    tok = pipe.tokenizer
    log = m2.RunLog(RESULTS / "m2_runs", owner=OWNER, axis="early warning", images_dir=IMAGES) \
        if part == "prospective" else None
    started = m2._now()

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    for seed_id in PARTS[part]["seed_ids"]:
        out = run_dir(part, seed_id)
        (REC / f"s{seed_id}").mkdir(parents=True, exist_ok=True)
        done = set(pd.read_csv(out / "captions.csv").caption_idx) if (out / "captions.csv").exists() else set()
        if (out / "objects.csv").exists():                # rows of a caption cut off before it was marked done
            o = pd.read_csv(out / "objects.csv")
            o[o.caption_idx.isin(done)].to_csv(out / "objects.csv", index=False, lineterminator="\n")
        stored = stored_measures(seed_id) if part == "development" else None
        if part == "development" and stored is None:
            raise FileNotFoundError(f"word-timing objects.csv for seed_id {seed_id} is missing: development "
                                    "paintings are checked against it")
        t0, made = time.time(), 0
        for r in caps.itertuples(index=False):
            cap = int(r.caption_idx)
            if cap in done:
                continue
            seed = m2.seed_for(cap, seed_id)
            sync()
            t1 = time.perf_counter()
            tl = painter.paint(r.caption, seed, record=True, previews=False, safe=True, keep_x0=DECISION_STEPS)
            sync()
            dt = time.perf_counter() - t1
            objs = mt.measure_objects(tok, r.caption, tl.maps, tl.timesteps, by_caption[cap])
            n_words = len(r.caption.split())
            for o in objs:
                slots, _, _ = mt.object_slots(tok, r.caption, o["word_start"], len(str(o["words"]).split()))
                o.update(seed_id=seed_id, obj_slots=" ".join(map(str, slots)), n_words=n_words, n_objects=len(objs))
            keep = n_caption_tokens(tok, r.caption) + 2
            np.savez(rec_path(seed_id, cap), maps=pool_maps(tl.maps, keep),
                     x0=np.stack([tl.x0[k][0].float().numpy() for k in DECISION_STEPS]).astype(np.float16),
                     latents=np.stack([tl.latents[k + 1][0].float().numpy() for k in DECISION_STEPS]).astype(np.float16),
                     x0_steps=np.array(DECISION_STEPS), content=np.array(content_slots(tok, r.caption), dtype=int),
                     timesteps=np.array(tl.timesteps))
            row = {"caption_idx": cap, "image_id": int(r.image_id), "seed_id": seed_id, "seed": seed,
                   "flagged": int(tl.flagged), "seconds": round(dt, 3), "n_objects": len(objs)}
            if part == "development":
                ref = np.asarray(Image.open(image_path(cap, seed_id)).convert("RGB"), dtype=np.int16)
                diff = int(np.abs(np.asarray(tl.final, dtype=np.int16) - ref).max())
                row.update(same_image=int(diff == 0), max_pixel_diff=diff,
                           same_measures=int(all(same_as_stored(stored, cap, o) for o in objs)))
            else:
                path = image_path(cap, seed_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                tl.final.save(path)
                log.image(CFG, SPLIT, cap, seed_id, seed, path.relative_to(IMAGES).as_posix(), tl.flagged, dt)
                row.update(same_image="", max_pixel_diff="", same_measures="")
            append_csv(out / "objects.csv", objs, OBJECT_COLUMNS)
            append_csv(out / "captions.csv", [row], CAPTION_COLUMNS)
            made += 1
            if made % 50 == 0:
                print(f"[{(time.time() - t0) / 60:5.1f} min] {part} seed_id {seed_id}: {len(done) + made}/{len(caps)} "
                      f"captions, {(time.time() - t0) / made:.2f} s each", flush=True)
        print(f"record {part} seed_id {seed_id} done: {len(done) + made} captions, {made} new, "
              f"in {(time.time() - t0) / 60:.1f} min", flush=True)
        if part == "development":
            c = pd.read_csv(out / "captions.csv")
            bad = c[(c.same_image != 1) | (c.same_measures != 1)]
            print(f"  repaints that differ from the stored picture or measures: {len(bad)} of {len(c)}", flush=True)
            if len(bad) > MAX_MISMATCH * len(c):
                raise SystemExit(f"more than {MAX_MISMATCH:.0%} of seed_id {seed_id}'s repaints differ: stop and report")
    if part == "prospective":
        update_run_row(log.root, PROSPECTIVE_NOTE, started)


def same_as_stored(stored: pd.DataFrame, cap: int, o: dict) -> bool:
    """The repaint's commit step and early share equal the word-timing runs' for this object."""
    try:
        s = stored.loc[(cap, o["category"], o["word_start"])]
    except KeyError:
        return False
    return int(s["commit_08"]) == int(o["commit_08"]) and abs(float(s["early"]) - float(o["early"])) < 1e-3


PROSPECTIVE_NOTE = "seed_ids 3, 4 and 5 cover rows 1000 to 1999 with labelled objects only (early-warning test)"


def logged_seed_ids(root: Path, config_id: str = CFG.config_id, split: str = SPLIT) -> list[int]:
    im = pd.read_csv(Path(root) / "images.csv")
    return sorted(int(s) for s in im[(im.config_id == config_id) & (im.split == split)].seed_id.unique())


def update_run_row(root: Path, note: str, started: str, config_id: str = CFG.config_id, split: str = SPLIT,
                   owner: str = OWNER) -> pd.DataFrame:
    """Bring the anchor's eval_2000 row of runs.csv up to date after new seed_ids: counts, timing and
    seed_ids recounted from images.csv, the note added once. Axis and start time stay as they were."""
    root = Path(root)
    runs = pd.read_csv(root / "runs.csv", dtype=str, keep_default_na=False)
    im = pd.read_csv(root / "images.csv")
    mine = im[(im.config_id == config_id) & (im.split == split)]
    sel = (runs.config_id == config_id) & (runs.split == split) & (runs.owner == owner)
    if not sel.any():
        m2.RunLog(root, owner=owner, axis="early warning", images_dir=IMAGES).add_run(
            CFG, split, len(mt.load_split(split)), logged_seed_ids(root), started=started, notes=note)
        return refresh_seed_ids(root)
    secs = pd.to_numeric(mine.seconds, errors="coerce").dropna()
    runs.loc[sel, "n_images"] = str(len(mine))
    runs.loc[sel, "n_flagged"] = str(int(mine.flagged.sum()))
    runs.loc[sel, "sec_per_image"] = f"{secs.mean():.3f}" if len(secs) else ""
    runs.loc[sel, "finished"] = m2._now()
    runs.loc[sel, "notes"] = [n if note in n else (f"{n}; {note}" if n else note) for n in runs.loc[sel, "notes"]]
    runs.to_csv(root / "runs.csv", index=False, lineterminator="\n")
    return refresh_seed_ids(root)


def refresh_seed_ids(root: Path) -> pd.DataFrame:
    """Set every run row's seed_ids to the seed_ids its images were logged with. Earlier runs wrote
    only the last seed_id painted, while counting the images of all of them."""
    runs = pd.read_csv(Path(root) / "runs.csv", dtype=str, keep_default_na=False)
    for i, r in runs.iterrows():
        ids = logged_seed_ids(root, r.config_id, r.split)
        if ids:
            runs.loc[i, "seed_ids"] = " ".join(map(str, ids))
    runs.to_csv(Path(root) / "runs.csv", index=False, lineterminator="\n")
    return runs


# ------------------------------------------------------------------ phase 2: judge the finished prospective pictures

def judge(args, batch: int = 8) -> None:
    from PIL import Image
    assert args.part == "prospective", "development pictures were judged in the word-timing runs"
    j = mt.Judges(args.device)
    for seed_id in PARTS["prospective"]["seed_ids"]:
        out = run_dir("prospective", seed_id)
        objs, caps = pd.read_csv(out / "objects.csv"), pd.read_csv(out / "captions.csv")
        done = set(pd.read_csv(out / "judges.csv").caption_idx) if (out / "judges.csv").exists() else set()
        todo = caps[~caps.caption_idx.isin(done)]
        want = {k: sorted(set(g.category)) for k, g in objs.groupby("caption_idx")}
        t0 = time.time()
        for i in range(0, len(todo), batch):
            part = todo.iloc[i:i + batch]
            ims = [Image.open(image_path(c, seed_id)).convert("RGB") for c in part.caption_idx]
            rows = []
            for c, sc in zip(part.caption_idx, j.scores(ims)):
                for k in want.get(c, []):
                    f, r, o = sc[k]
                    rows.append({"caption_idx": int(c), "category": k, "frcnn": round(f, 4), "retina": round(r, 4),
                                 "owl": round(o, 4)})
            append_csv(out / "judges.csv", rows, mt.JUDGE_COLUMNS)
        print(f"judge seed_id {seed_id} done: {len(done) + len(todo)} pictures in {(time.time() - t0) / 60:.1f} min",
              flush=True)


# ------------------------------------------------------------------ phase 3: previews and drafts

PREVIEW_COLUMNS = ["caption_idx", "category", "kind", "step", "frcnn", "retina", "owl", "clip"]
CLIP_ID = "openai/clip-vit-large-patch14"


def draft_timesteps(t_start: int, n: int = DRAFT_LEN) -> list[int]:
    """n timesteps spaced evenly from t_start down to (not including) 0."""
    return [int(round(t_start * (1 - j / n))) for j in range(n)]


def draft_scheduler(pipe):
    """DPM-Solver++ (2M) from the model's own noise schedule, ending at noise level 0."""
    from diffusers import DPMSolverMultistepScheduler
    return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++",
                                                   solver_order=2, final_sigmas_type="zero")


def draft(pipe, sch, emb, lat, t_start: int, guidance: float = CFG.guidance, n: int = DRAFT_LEN):
    """Finish a painting quickly: from latent `lat` at noise level `t_start`, n DPM-Solver++ steps
    with classifier-free guidance down to noise level 0. Returns the finished latent."""
    import torch
    sch.set_timesteps(timesteps=draft_timesteps(t_start, n), device=lat.device)
    x = lat
    with torch.no_grad():
        for t in sch.timesteps:
            pred = pipe.unet(torch.cat([x] * 2), t, encoder_hidden_states=emb, return_dict=False)[0]
            uncond, cond = pred.chunk(2)
            x = sch.step(uncond + guidance * (cond - uncond), t, x, return_dict=True).prev_sample
    return x


def draft_many(pipe, scheds, emb, lats, t_starts, guidance: float = CFG.guidance, n: int = DRAFT_LEN):
    """Several drafts of one painting at once, from different steps: the same arithmetic as `draft`,
    with one network call per draft step for all of them (each sample carries its own timestep)."""
    import torch
    m = len(lats)
    for sch, t0 in zip(scheds, t_starts):
        sch.set_timesteps(timesteps=draft_timesteps(t0, n), device=lats[0].device)
    xs = list(lats)
    ne, pe = emb.chunk(2)
    emb_m = torch.cat([ne.expand(m, -1, -1), pe.expand(m, -1, -1)])
    with torch.no_grad():
        for j in range(n):
            ts = torch.stack([sch.timesteps[j] for sch in scheds[:m]])
            x = torch.cat(xs)
            pred = pipe.unet(torch.cat([x, x]), torch.cat([ts, ts]), encoder_hidden_states=emb_m, return_dict=False)[0]
            uncond, cond = pred.chunk(2)
            eps = uncond + guidance * (cond - uncond)
            xs = [sch.step(eps[i:i + 1], sch.timesteps[j], xs[i], return_dict=True).prev_sample
                  for i, sch in enumerate(scheds[:m])]
    return torch.cat(xs)


class ClipScorer:
    """100 times the cosine similarity between a picture and "a photo of a {class}", CLIP ViT-L/14."""

    def __init__(self, device: str, classes, model_id: str = CLIP_ID):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = CLIPModel.from_pretrained(model_id).to(device, self.dtype).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.classes = list(classes)
        with torch.no_grad():
            t = self.proc.tokenizer([f"a photo of a {c}" for c in self.classes], padding=True, return_tensors="pt")
            t = {k: v.to(device) for k, v in t.items()}
            e = self.model.text_projection(self.model.text_model(**t).pooler_output)
            self.text = e / e.norm(dim=-1, keepdim=True)

    def scores(self, ims) -> list[dict]:
        import torch
        with torch.no_grad():
            px = self.proc.image_processor(images=ims, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
            e = self.model.visual_projection(self.model.vision_model(pixel_values=px).pooler_output)
            e = e / e.norm(dim=-1, keepdim=True)
            sims = (100.0 * e @ self.text.T).float().cpu().numpy()
        return [dict(zip(self.classes, row.tolist())) for row in sims]


def draft_path(seed_id: int, cap: int) -> Path:
    return REC / "drafts" / f"s{seed_id}" / f"{cap}.npy"


def preview(args) -> None:
    """Decode every kept one-step guess, make and decode the drafts, and score all of them with the
    three judges and CLIP. Pictures the safety checker blacked out are skipped: their objects are
    left out of every analysis."""
    import torch
    pipe = m2.load_pipeline(args.device)
    scheds = [draft_scheduler(pipe) for _ in DRAFT_STEPS]
    vae = pipe.vae
    j = mt.Judges(args.device)
    clip = ClipScorer(args.device, j.classes)
    for seed_id in PARTS[args.part]["seed_ids"]:
        out = run_dir(args.part, seed_id)
        draft_path(seed_id, 0).parent.mkdir(parents=True, exist_ok=True)
        objs, caps = pd.read_csv(out / "objects.csv"), pd.read_csv(out / "captions.csv")
        caps = caps[caps.flagged == 0]
        done = set(pd.read_csv(out / "previews.csv").caption_idx) if (out / "previews.csv").exists() else set()
        todo = caps[~caps.caption_idx.isin(done)]
        if args.limit:
            todo = todo.head(max(0, args.limit - len(done)))
        want = {k: sorted(set(g.category)) for k, g in objs.groupby("caption_idx")}
        prompts = mt.load_split(SPLIT).set_index("caption_idx").caption
        batch = args.batch
        t0, n = time.time(), 0
        for cap in todo.caption_idx.astype(int):
            if cap not in want:
                continue
            rec = np.load(rec_path(seed_id, cap))
            pe, ne = pipe.encode_prompt(prompts[cap], args.device, 1, True)
            emb = torch.cat([ne, pe])
            starts = [torch.from_numpy(rec["latents"][DECISION_STEPS.index(k)][None]).to(args.device, pipe.unet.dtype)
                      for k in DRAFT_STEPS]
            drafts = draft_many(pipe, scheds, emb, starts, [int(rec["timesteps"][k + 1]) for k in DRAFT_STEPS])
            np.save(draft_path(seed_id, cap), drafts.float().cpu().numpy().astype(np.float16))
            lats = torch.cat([torch.from_numpy(rec["x0"]).to(args.device, vae.dtype), drafts.to(vae.dtype)])
            ims = []
            with torch.no_grad():
                for i in range(0, len(lats), batch):
                    img = vae.decode(lats[i:i + batch] / vae.config.scaling_factor, return_dict=False)[0]
                    ims += pipe.image_processor.postprocess(img, output_type="pil")
            judged, clipped = [], []
            for i in range(0, len(ims), 2 * batch):
                judged += j.scores(ims[i:i + 2 * batch])
                clipped += clip.scores(ims[i:i + 2 * batch])
            labels = [("x0", k) for k in DECISION_STEPS] + [("draft", k) for k in DRAFT_STEPS]
            rows = [{"caption_idx": cap, "category": c, "kind": kind, "step": k, "frcnn": round(sc[c][0], 4),
                     "retina": round(sc[c][1], 4), "owl": round(sc[c][2], 4), "clip": round(cs[c], 3)}
                    for (kind, k), sc, cs in zip(labels, judged, clipped) for c in want[cap]]
            append_csv(out / "previews.csv", rows, PREVIEW_COLUMNS)
            n += 1
            if n % 100 == 0:
                print(f"[{(time.time() - t0) / 60:5.1f} min] previews {args.part} seed_id {seed_id}: "
                      f"{len(done) + n}/{len(caps)}, {(time.time() - t0) / n:.2f} s per painting", flush=True)
        print(f"preview {args.part} seed_id {seed_id} done: {len(done) + n} paintings, {n} new, in "
              f"{(time.time() - t0) / 60:.1f} min ({(time.time() - t0) / max(n, 1):.2f} s per painting)", flush=True)


# ------------------------------------------------------------------ phase 4: features

def _gauss3(sigma: float = 0.5) -> np.ndarray:
    g = np.exp(-np.array([-1.0, 0.0, 1.0]) ** 2 / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def smooth3(m: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """3 x 3 Gaussian smoothing over the last two axes, reflected edges (as Attend-and-Excite)."""
    k = _gauss3(sigma)
    p = np.pad(m, [(0, 0)] * (m.ndim - 2) + [(1, 1), (1, 1)], mode="reflect")
    h, w = m.shape[-2:]
    return sum(k[dy, dx] * p[..., dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3))


def _spatial(m: np.ndarray) -> np.ndarray:
    """(steps, H, W) maps scaled to sum to 1 over positions, flattened to (steps, H*W)."""
    f = m.reshape(m.shape[0], -1).astype(np.float64)
    return f / (f.sum(1, keepdims=True) + 1e-12)


def attention_features(maps: np.ndarray, content, objects: list[dict]) -> list[np.ndarray]:
    """Per object, a (steps, 6) array of share, peak, focus, overlap, lead and steady at each step.

    maps: (steps, slots, H, W) with slot 0 the start token and the last slot the end token.
    objects: dicts with `slot` (the measured token) and `obj_slots` (all tokens of its words).
    The definitions are the pre-registration's."""
    maps = maps.astype(np.float64)
    steps, n_slots = maps.shape[:2]
    caption = list(range(1, n_slots - 1))
    content = [int(s) for s in content if 0 < int(s) < n_slots - 1]
    mass = maps.mean(axis=(2, 3))                                            # (steps, slots)
    c_tot = mass[:, content].sum(1) if content else np.full(steps, np.nan)

    def share_of(obj):
        idx = [s for s in obj["obj_slots"] if s in content]
        return mass[:, idx].sum(1) / c_tot if idx else np.full(steps, np.nan)

    shares = [share_of(o) for o in objects]
    per_pixel = maps[:, caption].sum(1) + 1e-12                              # (steps, H, W)
    dists = [_spatial(maps[:, o["slot"]]) for o in objects]
    n_pos = dists[0].shape[1] if dists else 1
    out = []
    for i, o in enumerate(objects):
        a = maps[:, o["slot"]]
        peak = smooth3(a / per_pixel).reshape(steps, -1).max(1)
        p = dists[i]
        focus = 1.0 - (-(p * np.log(p + 1e-12)).sum(1)) / np.log(n_pos)
        others = [j for j in range(len(objects)) if j != i and objects[j]["slot"] != o["slot"]]
        overlap = np.max([np.minimum(p, dists[j]).sum(1) for j in others], axis=0) if others else np.zeros(steps)
        rivals = [shares[j] for j in others if not np.isnan(shares[j]).all()]
        lead = shares[i] / (shares[i] + np.max(rivals, axis=0)) if rivals else np.where(np.isnan(shares[i]), np.nan, 1.0)
        z = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-12)
        steady = np.concatenate([[np.nan], (z[1:] * z[:-1]).mean(1)])
        out.append(np.stack([shares[i], peak, focus, overlap, lead, steady], axis=1))
    return out


def at_step(curve: np.ndarray, k: int) -> dict:
    """The attention features entering at decision step k: each measure's value at k and its mean
    over steps 0 to k (steady from step 1, and not at k = 0)."""
    row = {}
    for j, name in enumerate(ATT_NAMES):
        if name == "steady" and k == 0:
            continue
        row[f"{name}_now@{k:02d}"] = float(curve[k, j])
        row[f"{name}_mean@{k:02d}"] = float(np.nanmean(curve[:k + 1, j])) if not np.isnan(curve[:k + 1, j]).all() \
            else float("nan")
    return row


def logit(p, eps: float = 1e-4):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def scorer_value(name: str, v: float) -> float:
    """The three judges' scores enter as clipped logits, the CLIP score as it is."""
    if np.isnan(v):
        return float("nan")
    return float(logit(v)) if name in JUDGE_NAMES else float(v)


def preview_features(prev: pd.DataFrame, kind: str = "x0") -> dict:
    """prev: one object's scores on its one-step previews (kind "x0") or its drafts (kind "draft"),
    columns step and one per scorer. Each scorer's value after every decision step (or draft step)
    and the best it gave so far."""
    steps, prefix = (DECISION_STEPS, "") if kind == "x0" else (DRAFT_STEPS, "draft_")
    if "kind" in prev:
        prev = prev[prev.kind == kind]
    prev = prev.set_index("step").reindex(steps)
    row = {}
    for name in SCORERS:
        v = prev[name].to_numpy(float) if name in prev else np.full(len(steps), np.nan)
        best = np.fmax.accumulate(v)
        for k, now, mx in zip(steps, v, best):
            row[f"{prefix}{name}_now@{k:02d}"] = scorer_value(name, now)
            row[f"{prefix}{name}_max@{k:02d}"] = scorer_value(name, mx)
    return row


PROMPT_COLUMNS = ["category", "n_objects", "position", "n_words", "n_tokens", "is_person"]


def labels_for(part: str, seed_id: int, out: Path) -> pd.DataFrame:
    """Each object's label on the finished picture, by the group judges (missing when both miss),
    with RetinaNet's score for the ceiling."""
    if part == "development":
        j = pd.read_csv(STORED / f"{SPLIT}__{CFG.config_id}__s{seed_id}" / "objects_judged.csv")
        j = j[mt.OBJECT_KEY + ["frcnn", "retina", "owl", "judge_group"]]
    else:
        objs = pd.read_csv(out / "objects.csv")[mt.OBJECT_KEY]
        j = objs.merge(pd.read_csv(out / "judges.csv"), on=["caption_idx", "category"], how="left",
                       validate="many_to_one")
        assert j.frcnn.notna().all(), "some prospective objects have no judge scores: run the judge phase"
        j["judge_group"] = [mt.judge_group(f, o) for f, o in zip(j.frcnn, j.owl)]
    return j.rename(columns={"frcnn": "final_frcnn", "retina": "final_retina", "owl": "final_owl",
                             "judge_group": "judge"})


def features(args) -> pd.DataFrame:
    part, rows = args.part, []
    for seed_id in PARTS[part]["seed_ids"]:
        out = run_dir(part, seed_id)
        objs, caps = pd.read_csv(out / "objects.csv"), pd.read_csv(out / "captions.csv")
        prev = pd.read_csv(out / "previews.csv")
        prev_by = {k: g for k, g in prev.groupby(["caption_idx", "category"])}
        keep = caps[caps.flagged == 0]
        if part == "development":
            keep = keep[(keep.same_image == 1) & (keep.same_measures == 1)]
        for cap, g in objs[objs.caption_idx.isin(set(keep.caption_idx))].groupby("caption_idx", sort=False):
            rec = np.load(rec_path(seed_id, int(cap)))
            ob = [{"slot": int(r.slot), "obj_slots": [int(s) for s in str(r.obj_slots).split()]}
                  for r in g.itertuples(index=False)]
            curves = attention_features(rec["maps"], rec["content"], ob)
            for r, curve in zip(g.itertuples(index=False), curves):
                row = {"seed_id": seed_id, "caption_idx": int(cap), "category": r.category, "word_start": int(r.word_start),
                       "n_objects": int(r.n_objects), "position": r.word_start / r.n_words, "n_words": int(r.n_words),
                       "n_tokens": int(r.n_tokens), "is_person": int(r.category == "person"),
                       "commit_08": int(r.commit_08), "early": float(r.early)}
                for k in DECISION_STEPS:
                    row.update(at_step(curve, k))
                row.update(preview_features(prev_by[(cap, r.category)], "x0"))
                row.update(preview_features(prev_by[(cap, r.category)], "draft"))
                rows.append(row)
        print(f"features {part} seed_id {seed_id}: {sum(1 for x in rows if x['seed_id'] == seed_id)} objects", flush=True)
    df = pd.DataFrame(rows)
    lab = pd.concat([labels_for(part, s, run_dir(part, s)).assign(seed_id=s) for s in PARTS[part]["seed_ids"]])
    df = df.merge(lab, on=["seed_id"] + mt.OBJECT_KEY, how="left", validate="one_to_one")
    assert df.judge.notna().all(), "some objects have no label"
    df["missing"] = (df.judge == "missing").astype(int)
    path = OUT / f"features_{part}.csv"
    df.to_csv(path, index=False, lineterminator="\n")
    if part == "development":
        print(f"{path.name}: {len(df)} objects, {int(df.missing.sum())} missing ({df.missing.mean():.3f}), "
              f"{df.caption_idx.nunique()} captions", flush=True)
    else:                                       # the prospective labels stay unread until the test phase
        print(f"{path.name}: {len(df)} objects in {df.caption_idx.nunique()} captions", flush=True)
    return df


# ------------------------------------------------------------------ phase 5: the detectors

FAMILIES = {"prompt": ("prompt",), "attention": ("attention",), "preview": ("preview",), "draft": ("draft",),
            "prompt+attention": ("prompt", "attention"), "prompt+preview": ("prompt", "preview"),
            "prompt+draft": ("prompt", "draft"), "full": ("prompt", "attention", "preview"),
            "everything": ("prompt", "attention", "preview", "draft")}
GBM = "full, boosted trees"
PROMPT_NUMERIC = ["n_objects", "position", "n_words", "n_tokens", "is_person"]
N_FOLDS, FOLD_SEED = 5, 0
N_BOOT, BOOT_SEED = mt.N_BOOT, mt.BOOT_SEED
FLAG_SHARE = 0.10                          # catch rate: 10% of present objects flagged
H1_STEP, H2_STEPS, H3_STEPS, H4_STEPS, GOOD_AUROC = 2, (0, 10), (3, 10, 15), (0, 2), 0.90


def columns_for(groups, k: int) -> tuple[list[str], list[str]]:
    """(numeric, categorical) feature columns of a family at decision step k."""
    num, cat = [], []
    if "prompt" in groups:
        num += PROMPT_NUMERIC
        cat += ["category"]
    if "attention" in groups:
        num += [f"{n}_{w}@{k:02d}" for n in ATT_NAMES for w in ("now", "mean") if not (n == "steady" and k == 0)]
    if "preview" in groups:
        num += [f"{j}_{w}@{k:02d}" for j in SCORERS for w in ("now", "max")]
    if "draft" in groups:
        num += [f"draft_{j}_{w}@{k:02d}" for j in SCORERS for w in ("now", "max")]
    return num, cat


def draft_cost(groups, k: int) -> int:
    """Extra network evaluations a family spends by step k: 5 per draft made so far."""
    return DRAFT_LEN * sum(1 for d in DRAFT_STEPS if d <= k) if "draft" in groups else 0


def make_model(num: list[str], cat: list[str], kind: str = "lr"):
    """The pre-registered detector: median fill and standardizing for numbers, one-hot class
    (classes with fewer than 10 rows pooled), then L2 logistic regression with C = 1 (or, for the
    secondary detector, boosted trees)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    parts = [("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num)]
    if cat:
        parts.append(("cat", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist"), cat))
    clf = LogisticRegression(C=1.0, max_iter=5000) if kind == "lr" else \
        HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
    return Pipeline([("features", ColumnTransformer(parts, sparse_threshold=0)), ("model", clf)])


def detectors() -> list[tuple]:
    """(name, step, numeric columns, categorical columns, kind) for every detector."""
    out = []
    for k in DECISION_STEPS:
        for fam, groups in FAMILIES.items():
            if "draft" in groups and k not in DRAFT_STEPS:
                continue                              # drafts exist at draft steps only
            out.append((fam, k, *columns_for(groups, k), "lr"))
        out.append((GBM, k, *columns_for(FAMILIES["full"], k), "gbm"))
    return out


def col(name: str, k: int) -> str:
    return f"{name}@{k:02d}"


def caption_folds(captions, n_folds: int = N_FOLDS, seed: int = FOLD_SEED) -> np.ndarray:
    """Each row's fold: the captions, in a random order drawn with `seed`, dealt into the folds in
    turn, so every seed of a caption lands in the same fold."""
    uniq = np.unique(captions)
    order = np.random.default_rng(seed).permutation(uniq.size)
    fold_of = dict(zip(uniq[order].tolist(), (np.arange(uniq.size) % n_folds).tolist()))
    return np.array([fold_of[c] for c in np.asarray(captions).tolist()])


def load_features(part: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / f"features_{part}.csv")
    return df.sort_values(["seed_id", "caption_idx", "word_start"]).reset_index(drop=True)


def out_of_fold(df: pd.DataFrame) -> pd.DataFrame:
    folds = caption_folds(df.caption_idx.to_numpy())
    y = df.missing.to_numpy()
    scores = {}
    for name, k, num, cat, kind in detectors():
        s = np.zeros(len(df))
        for f in range(N_FOLDS):
            tr, te = folds != f, folds == f
            model = make_model(num, cat, kind).fit(df.loc[tr, num + cat], y[tr])
            s[te] = model.predict_proba(df.loc[te, num + cat])[:, 1]
        scores[col(name, k)] = s
    return pd.DataFrame(scores)


# ------------------------------------------------------------------ measures with caption-bootstrap intervals

def auroc_w(score, y, w) -> float:
    m, p = (y == 1) & (w > 0), (y == 0) & (w > 0)
    num, den = mt._wins(score[m], w[m], score[p], w[p])
    return num / den if den else float("nan")


def ap_w(score, y, w) -> float:
    """Average precision with row weights; tied scores form one threshold (as scikit-learn)."""
    keep = w > 0
    score, y, w = score[keep], y[keep], w[keep]
    order = np.argsort(-score, kind="mergesort")
    s, yy, ww = score[order], y[order], w[order]
    tp, fp = np.cumsum(ww * yy), np.cumsum(ww * (1 - yy))
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]
    tp, fp = tp[last], fp[last]
    if tp[-1] <= 0:
        return float("nan")
    recall, precision = tp / tp[-1], tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def catch_w(score, y, w, flag_share: float = FLAG_SHARE) -> float:
    """Share of missing objects flagged when the threshold is the (1 - flag_share) quantile of the
    present objects' scores; objects above it are flagged."""
    p, m = (y == 0) & (w > 0), (y == 1) & (w > 0)
    if not p.any() or not m.any():
        return float("nan")
    order = np.argsort(score[p], kind="mergesort")
    cum = np.cumsum(w[p][order]) / w[p].sum()
    thr = score[p][order][min(np.searchsorted(cum, 1 - flag_share - 1e-12), order.size - 1)]
    return float((w[m] * (score[m] > thr)).sum() / w[m].sum())


MEASURES = {"auroc": auroc_w, "ap": ap_w, "catch": catch_w}


def bootstrap(df: pd.DataFrame, scores: dict, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[dict, dict]:
    """Point values and caption-bootstrap draws of every measure for every score.
    Returns ({score: {measure: value}}, {score: (n_boot, n_measures) array})."""
    y = df.missing.to_numpy()
    uniq, code = np.unique(df.caption_idx.to_numpy(), return_inverse=True)
    ones = np.ones(len(df))
    point = {c: {m: f(s, y, ones) for m, f in MEASURES.items()} for c, s in scores.items()}
    rng = np.random.default_rng(seed)
    draws = {c: np.empty((n_boot, len(MEASURES))) for c in scores}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size)[code].astype(float)
        for c, s in scores.items():
            draws[c][b] = [f(s, y, w) for f in MEASURES.values()]
    return point, draws


def ci(x) -> list[float]:
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def within(df: pd.DataFrame, s: np.ndarray) -> dict:
    """The word-timing runs' within-object test on a detector's score (higher score = more likely missing)."""
    return mt.within_object_test(df.assign(judge=df.judge, risk=s), "risk", +1.0)


def evaluate(df: pd.DataFrame, scores: pd.DataFrame, n_boot: int = N_BOOT) -> tuple[dict, pd.DataFrame]:
    """Every measure for every detector at every step, the ceiling, and the four readings."""
    s = {c: scores[c].to_numpy(float) for c in scores.columns}
    s["ceiling"] = -df.final_retina.to_numpy(float)          # RetinaNet on the finished picture
    point, draws = bootstrap(df, s, n_boot)
    mi = {m: i for i, m in enumerate(MEASURES)}
    rows = []
    for name, k, *_ in detectors():
        c = col(name, k)
        w = within(df, s[c])
        groups = FAMILIES.get(name, FAMILIES["full"])
        row = {"detector": name, "step": k, "steps_done": k + 1, "extra_unet_calls": draft_cost(groups, k)}
        for m in MEASURES:
            row[m] = round(point[c][m], 4)
            row[f"{m}_lo"], row[f"{m}_hi"] = ci(draws[c][:, mi[m]])
        row.update(within_auroc=w["auroc"], within_lo=w["ci95"][0], within_hi=w["ci95"][1],
                   n_discordant=w["n_discordant"], within_reading=w["reading"])
        rows.append(row)
    curves = pd.DataFrame(rows)

    def diff(a: str, b: str, k: int, m: str = "auroc") -> dict:
        x = draws[col(a, k)][:, mi[m]] - draws[col(b, k)][:, mi[m]]
        return {"step": k, "difference": round(point[col(a, k)][m] - point[col(b, k)][m], 4), "ci95": ci(x)}

    def at(name, k, m="auroc"):
        return round(point[col(name, k)][m], 4)

    h1 = diff("full", "prompt", H1_STEP)
    h1_within = within(df, s[col("full", H1_STEP)])
    h2 = [diff("prompt+attention", "prompt+preview", k) for k in H2_STEPS]
    lead = ["attention ahead" if d["ci95"][0] > 0 else "preview ahead" if d["ci95"][1] < 0 else "no clear difference"
            for d in h2]
    crossover = next((k for k in DECISION_STEPS if at("prompt+preview", k) > at("prompt+attention", k)), None)
    good = next((k for k in DECISION_STEPS if at("full", k) >= GOOD_AUROC), None)
    drafted = [k for k in DRAFT_STEPS if k in DECISION_STEPS]
    good_all = next((k for k in drafted if at("everything", k) >= GOOD_AUROC), None)
    h4a = [diff("prompt+draft", "prompt+preview", k) for k in H4_STEPS]
    h4b = [diff("everything", "prompt+draft", k) for k in H4_STEPS]
    return {
        "rows": int(len(df)), "missing": int(df.missing.sum()), "captions": int(df.caption_idx.nunique()),
        "missing_rate": round(float(df.missing.mean()), 4),
        "ceiling": {"label": "RetinaNet v2 on the finished picture", "auroc": round(point["ceiling"]["auroc"], 4),
                    "ci95": ci(draws["ceiling"][:, mi["auroc"]]), "ap": round(point["ceiling"]["ap"], 4)},
        "H1_painting_adds": {"auroc_full_minus_prompt": h1, "reading": "ADDS" if h1["ci95"][0] > 0 else "NOT_SHOWN",
                             "within_object_full": h1_within, "within_reading": h1_within["reading"]},
        "H2_attention_first": {"prompt+attention_minus_prompt+preview": h2,
                               "by_step": dict(zip(map(str, H2_STEPS), lead)),
                               "reading": "ATTENTION_FIRST" if lead == ["attention ahead", "preview ahead"] else "NOT_SHOWN",
                               "first_step_preview_passes_attention": crossover},
        "H3_how_early": {"first_step_full_auroc_at_least_0.90": good,
                         "first_draft_step_everything_auroc_at_least_0.90": good_all,
                         "catch_rate": {str(k): {"full": at("full", k, "catch"), "prompt": at("prompt", k, "catch")}
                                        for k in H3_STEPS},
                         "catch_rate_with_drafts": {str(k): {"everything": at("everything", k, "catch"),
                                                             "prompt+draft": at("prompt+draft", k, "catch"),
                                                             "extra_unet_calls": draft_cost(FAMILIES["everything"], k)}
                                                    for k in drafted}},
        "H4_draft": {"prompt+draft_minus_prompt+preview": h4a,
                     "draft_reading": "DRAFT_AHEAD" if all(d["ci95"][0] > 0 for d in h4a) else "NOT_SHOWN",
                     "everything_minus_prompt+draft": h4b,
                     "still_adds_reading": "STILL_ADDS" if h4b[H4_STEPS.index(2)]["ci95"][0] > 0 else "NOT_SHOWN"},
    }, curves


def file_sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings() -> dict:
    import sklearn
    return {"split": SPLIT, "config_id": CFG.config_id, "parts": PARTS, "decision_steps": list(DECISION_STEPS),
            "families": {k: list(v) for k, v in FAMILIES.items()}, "secondary": GBM, "folds": N_FOLDS,
            "fold_seed": FOLD_SEED, "bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED, "flag_share": FLAG_SHARE,
            "sklearn": sklearn.__version__, "protocol_sha": m2.protocol_sha(),
            "script_sha": file_sha(Path(__file__))[:12], "preregistration": "docs/m2_detector_preregistration.md"}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def develop(args) -> dict:
    """The six families and the secondary detector under 5-fold caption-grouped cross-validation."""
    df = load_features("development")
    scores = out_of_fold(df)
    out = OUT / "development"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([df[["seed_id"] + mt.OBJECT_KEY + ["missing"]], scores], axis=1).to_csv(
        out / "scores.csv", index=False, lineterminator="\n")
    summary, curves = evaluate(df, scores, args.n_boot)
    summary = {"settings": settings(), **summary}
    curves.to_csv(out / "curves.csv", index=False, lineterminator="\n")
    write_json(out / "summary.json", summary)
    report(summary, curves, "development, out-of-fold")
    return summary


def lock(args) -> dict:
    """Fit every detector on all development rows and seal them, once."""
    import joblib
    path = OUT / "lock.json"
    if path.exists():
        raise SystemExit(f"{path} already exists: the detectors were locked at "
                         f"{json.loads(path.read_text())['locked_at']}. Locking twice is not allowed.")
    df = load_features("development")
    y = df.missing.to_numpy()
    models = {col(name, k): make_model(num, cat, kind).fit(df[num + cat], y) for name, k, num, cat, kind in detectors()}
    joblib.dump(models, OUT / "models.joblib")
    seal = {"locked_at": m2._now(), "models": "models.joblib", "models_sha256": file_sha(OUT / "models.joblib"),
            "features_development_sha256": file_sha(OUT / "features_development.csv"), "rows": int(len(df)),
            "missing": int(df.missing.sum()), "detectors": len(models), **settings()}
    write_json(path, seal)
    print(json.dumps({k: seal[k] for k in ("locked_at", "models_sha256", "rows", "missing", "detectors")}, indent=2))
    return seal


def prospective_test(args) -> dict:
    """Apply the locked detectors once to the prospective paintings. Every run is logged."""
    import joblib
    path = OUT / "lock.json"
    if not path.exists():
        raise SystemExit("no lock.json: run the lock phase first. The prospective labels stay unread until then.")
    seal = json.loads(path.read_text())
    if file_sha(OUT / seal["models"]) != seal["models_sha256"]:
        raise SystemExit("models.joblib differs from the locked one: stop and report")
    out = OUT / "prospective"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.log", "a", encoding="utf-8") as f:
        f.write(f"{m2._now()}  test run, lock {seal['models_sha256'][:12]} made {seal['locked_at']}, "
                f"script {file_sha(Path(__file__))[:12]}\n")
    if not (OUT / "features_prospective.csv").exists():
        features(argparse.Namespace(part="prospective"))
    df = load_features("prospective")
    models = joblib.load(OUT / seal["models"])
    scores = pd.DataFrame({col(name, k): models[col(name, k)].predict_proba(df[num + cat])[:, 1]
                           for name, k, num, cat, _ in detectors()})
    pd.concat([df[["seed_id"] + mt.OBJECT_KEY + ["missing"]], scores], axis=1).to_csv(
        out / "scores.csv", index=False, lineterminator="\n")
    summary, curves = evaluate(df, scores, args.n_boot)
    summary = {"settings": settings(), "lock": {k: seal[k] for k in ("locked_at", "models_sha256")},
               "runs_logged": sum(1 for _ in open(out / "runs.log", encoding="utf-8")), **summary}
    curves.to_csv(out / "curves.csv", index=False, lineterminator="\n")
    write_json(out / "summary.json", summary)
    report(summary, curves, "prospective test")
    return summary


def report(summary: dict, curves: pd.DataFrame, label: str) -> None:
    print(f"\n== {label}: {summary['rows']} objects, {summary['missing']} missing ({summary['missing_rate']}), "
          f"{summary['captions']} captions")
    print(f"ceiling: {summary['ceiling']}")
    show = curves[curves.detector.isin(["prompt", "prompt+attention", "prompt+preview", "prompt+draft", "full",
                                        "everything", GBM])]
    print(show.pivot(index="steps_done", columns="detector", values="auroc").to_string())
    for h in ("H1_painting_adds", "H2_attention_first", "H3_how_early", "H4_draft"):
        print(f"{h}: {json.dumps(summary[h])}")


# ------------------------------------------------------------------ phase 6: figures

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#ffffff"
SERIES = {"prompt": ("#8a8984", "--", "o"), "prompt+attention": ("#2a78d6", "-", "o"),
          "prompt+preview": ("#eb6834", "-", "o"), "prompt+draft": ("#1baf7a", "-", "s"), "full": (INK, "-", "o"),
          "everything": (INK, ":", "s")}
NAMES = {"prompt": "prompt only", "prompt+attention": "prompt + attention", "prompt+preview": "prompt + preview",
         "prompt+draft": "prompt + draft", "full": "prompt + attention + preview", "everything": "all four"}


def _axes_style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def plot_how_early(panels: list[tuple[str, pd.DataFrame, dict]], path: Path) -> None:
    """AUROC against the number of steps done, one panel per data part, with the ceiling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.7), sharey=True, squeeze=False)
    for ax, (title, curves, summary) in zip(axes[0], panels):
        _axes_style(ax)
        for fam, (color, ls, mk) in SERIES.items():
            c = curves[curves.detector == fam].sort_values("step")
            if c.empty:
                continue
            x = c.steps_done.to_numpy()
            if fam != "everything":
                ax.fill_between(x, c.auroc_lo, c.auroc_hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, c.auroc, color=color, ls=ls, lw=2.0 if fam in ("full", "everything") else 1.6, marker=mk,
                    ms=3.6 if mk == "s" else 3.2, solid_capstyle="round", label=NAMES[fam])
        top = summary["ceiling"]["auroc"]
        ax.axhline(top, color=INK2, lw=1.0, ls=":")
        ax.text(1, 0.52, f"dotted line: ceiling {top:.3f}\n(RetinaNet on the finished picture)", ha="left",
                va="bottom", fontsize=7, color=INK2)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("denoising steps done (of 30)", fontsize=8, color=INK2)
        ax.set_xticks([1, 3, 5, 9, 11, 16, 21])
    axes[0][0].set_ylabel("AUROC, missing vs present", fontsize=8, color=INK2)
    axes[0][0].set_ylim(0.5, 1.0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_within(panels: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    """Within-object AUROC against steps done: the prompt is the same in every pair, so only the
    painting can separate the seed that loses the object from the seed that keeps it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.5), sharey=True, squeeze=False)
    for ax, (title, curves) in zip(axes[0], panels):
        _axes_style(ax)
        for fam in ("prompt+attention", "prompt+preview", "prompt+draft", "full", "everything"):
            color, ls, mk = SERIES[fam]
            c = curves[curves.detector == fam].sort_values("step")
            if c.empty:
                continue
            if fam != "everything":
                ax.fill_between(c.steps_done, c.within_lo, c.within_hi, color=color, alpha=0.12, linewidth=0)
            ax.plot(c.steps_done, c.within_auroc, color=color, ls=ls, lw=2.0 if fam in ("full", "everything") else 1.6,
                    marker=mk, ms=3.6 if mk == "s" else 3.2, label=NAMES[fam])
        ax.axhline(0.5, color=INK2, lw=1.0, ls=":")
        ax.text(21, 0.49, "dotted line: prompt only, 0.5 by construction", fontsize=7, color=INK2, ha="right", va="top")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("denoising steps done (of 30)", fontsize=8, color=INK2)
        ax.set_xticks([1, 3, 5, 9, 11, 16, 21])
    axes[0][0].set_ylabel("within-object AUROC", fontsize=8, color=INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def pick_example(scores: pd.DataFrame, df: pd.DataFrame, k: int = 4, name: str = "everything") -> tuple:
    """The development object whose detector at step k separates its losing seed from its keeping
    seed by the widest margin (out-of-fold scores). Exploratory, chosen by rule."""
    d = df[mt.OBJECT_KEY + ["seed_id", "missing"]].assign(risk=scores[col(name, k)].to_numpy())
    best, key = -np.inf, None
    for obj, g in d.groupby(mt.OBJECT_KEY):
        if g.missing.nunique() < 2:
            continue
        m, p = g[g.missing == 1], g[g.missing == 0]
        gap = m.risk.max() - p.risk.min()
        if gap > best:
            best, key = gap, (obj, int(m.loc[m.risk.idxmax(), "seed_id"]), int(p.loc[p.risk.idxmin(), "seed_id"]))
    return key


def plot_filmstrip(args, df: pd.DataFrame, scores: pd.DataFrame, path: Path) -> dict:
    """Two paintings of one caption that differ only in seed: the quick drafts from steps 0, 2, 4
    and 8, then the finished picture, with the all-four detector's out-of-fold score under each
    draft. Every frame passes the safety checker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from w2w.rewind import Painter
    example = pick_example(scores, df)
    if example is None:
        print("filmstrip skipped: no development object is missing under one seed and present under another")
        return {}
    (cap, category, word_start), lost, kept = example
    caption = mt.load_split(SPLIT).set_index("caption_idx").caption[cap]
    pipe = m2.load_pipeline(args.device)
    painter = Painter(pipe, args.device, STEPS, CFG.guidance, spacing=CFG.schedule, cpu_noise=True)
    fig, axes = plt.subplots(2, len(DRAFT_STEPS) + 1, figsize=(1.6 * (len(DRAFT_STEPS) + 1), 3.7))
    for r, seed_id in enumerate((kept, lost)):
        drafts = np.load(draft_path(seed_id, cap))
        row = df[(df.seed_id == seed_id) & (df.caption_idx == cap) & (df.word_start == word_start)].index[0]
        for c, k in enumerate(DRAFT_STEPS):
            lat = torch.from_numpy(drafts[c][None]).to(args.device)
            axes[r, c].imshow(painter.decode(lat, safe=True))
            axes[r, c].set_title(f"draft after {k + 1} step{'s' if k else ''}" if r == 0 else "", fontsize=7, color=INK2)
            axes[r, c].set_xlabel(f"risk {scores.loc[row, col('everything', k)]:.2f}", fontsize=7, color=INK)
        axes[r, -1].imshow(Image.open(image_path(cap, seed_id)).convert("RGB"))
        axes[r, -1].set_title("finished, 30 steps" if r == 0 else "", fontsize=7, color=INK2)
        axes[r, -1].set_xlabel(f"{category}: {'missing' if seed_id == lost else 'present'}", fontsize=7, color=INK)
        axes[r, 0].set_ylabel(f"seed_id {seed_id}", fontsize=7, color=INK2)
        for a in axes[r]:
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
    fig.suptitle(f"\u201c{caption}\u201d", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return {"caption_idx": int(cap), "caption": caption, "object": category, "word_start": int(word_start),
            "seed_kept": kept, "seed_lost": lost}


def figures(args) -> None:
    dev = OUT / "development"
    panels, within_panels = [], []
    for title, d in (("development, out-of-fold", dev), ("prospective test", OUT / "prospective")):
        if (d / "curves.csv").exists():
            curves = pd.read_csv(d / "curves.csv")
            summary = json.loads((d / "summary.json").read_text())
            panels.append((title, curves, summary))
            within_panels.append((title, curves))
    plot_how_early(panels, OUT / "how_early.png")
    plot_within(within_panels, OUT / "within_object.png")
    if not args.no_film:
        df, scores = load_features("development"), pd.read_csv(dev / "scores.csv")
        keys = ["seed_id"] + mt.OBJECT_KEY
        assert (df[keys].to_numpy() == scores[keys].to_numpy()).all(), "scores.csv rows are out of step with the features"
        ex = plot_filmstrip(args, df, scores, OUT / "filmstrip.png")
        write_json(OUT / "filmstrip.json", ex)
        print(f"filmstrip: {ex}")
    print(f"figures in {OUT}")


# ------------------------------------------------------------------ main

PHASES = {"record": record, "judge": judge, "preview": preview, "features": features,
          "develop": develop, "lock": lock, "test": prospective_test, "figures": figures}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=list(PHASES))
    ap.add_argument("part", nargs="?", choices=list(PARTS), help="for record, judge, preview and features")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="only the first N captions of the part (for checks)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--batch", type=int, default=4, help="preview: pictures decoded together (judged in twice that)")
    ap.add_argument("--no-film", action="store_true", help="figures without the example filmstrip")
    args = ap.parse_args()
    if args.phase in ("record", "judge", "preview", "features") and args.part is None:
        ap.error(f"{args.phase} needs a part: development or prospective")
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
