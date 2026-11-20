"""Milestone 2 generation protocol, shared by the whole group.

Every M2 image is made through this file (or through code checked against it), so the four
members' numbers can sit in one table.

Anchor setting: Stable Diffusion 1.5 in half precision, 512 x 512, each prompt encoded by the
pipeline's own CLIP text encoder, DDIM, 30 steps, eta 0, guidance 7.5, leading timestep
spacing (Stable Diffusion's default). The safety checker stays on; flagged images are logged.

Seed rule: seed = 100000 * seed_id + caption_idx. Stress prompts use their row number in
stress_prompts.csv (0 to 199) in place of caption_idx. The starting noise is drawn on the CPU
and handed to the pipeline as `latents`, so a laptop and a Colab T4 start from identical noise.

A sweep changes one field of `GenConfig` away from the anchor:
  sampler   ddim | pndm | dpmpp2m (DPM-Solver++ 2M)
  schedule  leading | linspace | trailing   timestep spacing, any sampler
            karras | ays                      Karras sigmas, Align Your Steps: dpmpp2m only
  steps, guidance

The noise schedule is the list of noise levels the sampler visits; the sampler is the update
rule that moves between them. `visited_timesteps(cfg)` gives the list for any setting, and
`python m2_protocol.py` prints it for every planned setting.

Use (Colab or laptop):
    import m2_protocol as m2
    pipe = m2.load_pipeline()
    m2.smoke_check(pipe)                                        # Wednesday 6 PM check
    log = m2.RunLog("m2_out", owner="Tanmayi", axis="guidance")
    for g in (3, 5, 7.5, 10, 12):
        m2.run_config(pipe, m2.GenConfig(guidance=g), "dev_500", log)
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROTOCOL_VERSION = "m2-v1"
SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SPLITS_URL = "https://raw.githubusercontent.com/tanmayi123/from-words-to-worlds/main/milestone-2/splits/"

SAMPLERS = ("ddim", "pndm", "dpmpp2m")
SPACINGS = ("leading", "linspace", "trailing")
SCHEDULES = SPACINGS + ("karras", "ays")

# Align Your Steps for SD 1.5 (Sabour et al., 2024): NVIDIA's 10-step noise levels and the
# matching timesteps shipped in diffusers. Diffusers stores the last level as 0; the published
# value 0.029 is kept here because interpolating to more steps needs it.
AYS_SD15_TIMESTEPS = [999, 850, 736, 645, 545, 455, 343, 233, 124, 24]
AYS_SD15_SIGMAS = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]

# SD 1.5's training noise schedule (scaled_linear betas over 1,000 steps) as noise levels sigma(t).
_BETAS = np.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=np.float64) ** 2
_ALPHAS_CUM = np.cumprod(1.0 - _BETAS)
TRAIN_SIGMAS = np.sqrt((1 - _ALPHAS_CUM) / _ALPHAS_CUM)

# The Wednesday check: every machine must print exactly these.
SMOKE_CAPTION_IDX = 3669          # first caption of dev_500
EXPECTED_NOISE_SUM = 36.4438      # start_noise(seed_for(3669, 0)).double().sum(), 4 decimals
EXPECTED_ANCHOR_TIMESTEPS = list(range(958, 0, -33))


# ------------------------------------------------------------------ settings

@dataclass(frozen=True)
class GenConfig:
    sampler: str = "ddim"
    schedule: str = "leading"
    steps: int = 30
    guidance: float = 7.5

    def __post_init__(self):
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {self.sampler!r}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        if self.schedule not in SPACINGS and self.sampler != "dpmpp2m":
            raise ValueError("Karras sigmas and Align Your Steps are run on dpmpp2m only")
        if self.steps < 2 or self.guidance < 1.0:
            raise ValueError("steps must be at least 2 and guidance at least 1")

    @property
    def config_id(self) -> str:
        return f"{self.sampler}-{self.schedule}-s{self.steps}-g{self.guidance:g}"

    @property
    def unet_calls(self) -> int:
        """Network evaluations per image (each one runs the prompt and the empty prompt together).
        PNDM's PLMS start evaluates one timestep twice, so 30 steps cost 31 calls."""
        return self.steps + 1 if self.sampler == "pndm" else self.steps


ANCHOR = GenConfig()


def sd15_scheduler_config() -> dict:
    """SD 1.5's scheduler config as shipped with the model, so schedules can be inspected offline."""
    return {"_class_name": "PNDMScheduler", "beta_end": 0.012, "beta_schedule": "scaled_linear",
            "beta_start": 0.00085, "num_train_timesteps": 1000, "set_alpha_to_one": False,
            "skip_prk_steps": True, "steps_offset": 1, "trained_betas": None, "clip_sample": False}


def make_scheduler(base_config: dict, cfg: GenConfig):
    """A fresh scheduler for `cfg`, built from the model's own scheduler config."""
    from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, PNDMScheduler
    base = dict(base_config)
    if cfg.sampler == "ddim":
        return DDIMScheduler.from_config(base, timestep_spacing=cfg.schedule)
    if cfg.sampler == "pndm":  # PLMS only, as in SD 1.5's own config
        return PNDMScheduler.from_config(base, timestep_spacing=cfg.schedule, skip_prk_steps=True)
    kw = dict(algorithm_type="dpmsolver++", solver_order=2, use_karras_sigmas=(cfg.schedule == "karras"))
    if cfg.schedule in SPACINGS:
        kw["timestep_spacing"] = cfg.schedule
    return DPMSolverMultistepScheduler.from_config(base, **kw)


def ays_timesteps(steps: int) -> list[int]:
    """Align Your Steps for SD 1.5 at `steps` steps.

    At 10 steps: the published timesteps. Otherwise NVIDIA's recipe: interpolate the 11
    published noise levels log-linearly to steps + 1 levels; the sampler visits the first
    `steps` of them and finishes at zero noise. Each level goes to the training timestep with the
    nearest noise level (this rounding would give 125 where the published 10-step list has 124)."""
    if steps == len(AYS_SD15_TIMESTEPS):
        return list(AYS_SD15_TIMESTEPS)
    s = np.log(np.asarray(AYS_SD15_SIGMAS)[::-1])
    levels = np.exp(np.interp(np.linspace(0, 1, steps + 1), np.linspace(0, 1, len(s)), s))[::-1][:steps]
    ts = [int(np.abs(np.log(TRAIN_SIGMAS) - np.log(v)).argmin()) for v in levels]
    if len(set(ts)) != steps or ts != sorted(ts, reverse=True):
        raise ValueError(f"Align Your Steps at {steps} steps maps two noise levels to one timestep")
    return ts


def custom_timesteps(cfg: GenConfig) -> list[int] | None:
    return ays_timesteps(cfg.steps) if cfg.schedule == "ays" else None


def visited_timesteps(cfg: GenConfig, base_config: dict | None = None) -> list[int]:
    """The timesteps the network is evaluated at, in order (PNDM repeats one)."""
    sch = make_scheduler(base_config or sd15_scheduler_config(), cfg)
    ts = custom_timesteps(cfg)
    if ts is not None:
        sch.set_timesteps(timesteps=ts)
    else:
        sch.set_timesteps(cfg.steps)
    return [int(t) for t in sch.timesteps]


# ------------------------------------------------------------------ seeds and noise

def seed_for(key: int, seed_id: int = 0) -> int:
    """The image seed: 100000 * seed_id + caption_idx (or the stress prompt's row number)."""
    key, seed_id = int(key), int(seed_id)
    if not (0 <= key < 100000 and seed_id >= 0):
        raise ValueError(f"key must be in [0, 100000) and seed_id >= 0, got {key}, {seed_id}")
    return 100000 * seed_id + key


def start_noise(seed: int, size: int = 64) -> torch.Tensor:
    """The starting latent, drawn on the CPU in float32 so every machine gets the same numbers."""
    g = torch.Generator("cpu").manual_seed(int(seed))
    return torch.randn((1, 4, size, size), generator=g, dtype=torch.float32)


# ------------------------------------------------------------------ generation

def load_pipeline(device: str = "cuda", dtype=torch.float16):
    """SD 1.5 with its safety checker, progress bar off."""
    from diffusers import StableDiffusionPipeline
    try:
        pipe = StableDiffusionPipeline.from_pretrained(SD_ID, dtype=dtype)
    except TypeError:  # diffusers before `dtype` was accepted
        pipe = StableDiffusionPipeline.from_pretrained(SD_ID, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def configure(pipe, cfg: GenConfig) -> None:
    """Put `cfg`'s scheduler on the pipeline, always built from the model's original config."""
    if not hasattr(pipe, "_m2_base_config"):
        pipe._m2_base_config = dict(pipe.scheduler.config)
    pipe.scheduler = make_scheduler(pipe._m2_base_config, cfg)


@torch.no_grad()
def generate(pipe, prompt: str, cfg: GenConfig, seed: int):
    """(PIL image, flagged by the safety checker, seconds) for one prompt."""
    configure(pipe, cfg)
    lat = start_noise(seed, pipe.unet.config.sample_size).to(pipe.device, pipe.unet.dtype)
    kw = dict(prompt=prompt, guidance_scale=cfg.guidance, latents=lat)
    ts = custom_timesteps(cfg)
    if ts is not None:
        kw["timesteps"] = ts
    else:
        kw["num_inference_steps"] = cfg.steps
    cuda = pipe.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe(**kw)
    if cuda:
        torch.cuda.synchronize()
    flags = out.nsfw_content_detected
    return out.images[0], bool(flags[0]) if flags is not None else False, time.perf_counter() - t0


# ------------------------------------------------------------------ splits

def load_split(name: str, root=None):
    """eval_2000, eval_500, dev_500, ref_2000_v2, groundtruth_eval_2000, groundtruth_dev_500 or
    stress_prompts. Looks in `root`, then in the splits folder next to this file's folder (as in
    milestone-2/protocol and milestone-2/splits), then ./milestone-2/splits and ./splits, and
    finally downloads it from the group repo. stress_prompts gets a `seed_key` column: its row
    number, used by the seed rule."""
    import pandas as pd
    df = None
    beside = Path(__file__).resolve().parent.parent / "splits"
    for base in ([Path(root)] if root else []) + [beside, Path("milestone-2/splits"), Path("splits")]:
        if (base / f"{name}.csv").exists():
            df = pd.read_csv(base / f"{name}.csv")
            break
    if df is None:
        df = pd.read_csv(SPLITS_URL + f"{name}.csv")
    if "prompt_id" in df.columns:
        df["seed_key"] = np.arange(len(df))
    return df


def split_columns(df) -> tuple[str, str, str]:
    """(key column, prompt column, seed column) for a COCO split or the stress prompts."""
    if "prompt_id" in df.columns:
        if "seed_key" not in df.columns:
            raise ValueError("load stress_prompts with load_split() so each prompt gets its seed_key")
        return "prompt_id", "prompt", "seed_key"
    return "caption_idx", "caption", "caption_idx"


# ------------------------------------------------------------------ the generation log

RUN_COLUMNS = ["config_id", "owner", "axis", "split", "n_prompts", "seed_ids", "seed_rule", "sampler", "schedule",
               "steps", "unet_calls", "guidance", "dtype", "gpu", "diffusers_version", "torch_version",
               "protocol_version", "protocol_sha", "git_commit", "sec_per_image", "n_images", "n_flagged",
               "started", "finished", "notes"]
IMAGE_COLUMNS = ["config_id", "split", "key", "seed_id", "seed", "file", "flagged", "seconds"]
SEED_RULE = "m2: 100000*seed_id + key, CPU float32 noise"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def protocol_sha() -> str:
    """First 12 hex digits of this file's sha256, so every row says which protocol made it."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def environment() -> dict:
    import diffusers
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                timeout=5, cwd=Path(__file__).parent).stdout.strip() or "n/a"
    except Exception:
        commit = "n/a"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"gpu": gpu, "diffusers_version": diffusers.__version__, "torch_version": torch.__version__,
            "protocol_version": PROTOCOL_VERSION, "protocol_sha": protocol_sha(), "git_commit": commit}


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


class RunLog:
    """<root>/runs.csv: one row per setting run (rewritten when a setting is resumed).
    <root>/images.csv: one row per image, appended as each image is saved.
    Images go to <images_dir>/<split>/<config_id>/<key>_s<seed_id>.png (images_dir defaults to
    <root>/images); the `file` column holds the part after <images_dir>/."""

    def __init__(self, root, owner: str, axis: str, images_dir=None):
        self.root = Path(root)
        self.images_dir = Path(images_dir) if images_dir is not None else self.root / "images"
        self.root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.owner, self.axis, self.env = owner, axis, environment()
        if not (self.root / "images.csv").exists():
            _write(self.root / "images.csv", IMAGE_COLUMNS, [])
        if not (self.root / "runs.csv").exists():
            _write(self.root / "runs.csv", RUN_COLUMNS, [])

    def image(self, cfg: GenConfig, split: str, key, seed_id: int, seed: int, file: str, flagged: bool,
              seconds: float | None) -> None:
        row = {"config_id": cfg.config_id, "split": split, "key": key, "seed_id": seed_id, "seed": seed,
               "file": file, "flagged": int(flagged), "seconds": "" if seconds is None else round(seconds, 3)}
        with open(self.root / "images.csv", "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=IMAGE_COLUMNS, lineterminator="\n").writerow(row)

    def add_run(self, cfg: GenConfig, split: str, n_prompts: int, seed_ids, started: str = "",
                seed_rule: str = SEED_RULE, dtype: str = "float16", notes: str = "", n_images: int | None = None,
                n_flagged: int | None = None, sec_per_image: float | None = None) -> dict:
        """Write (or rewrite) the runs.csv row for `cfg` on `split`, counting its images from
        images.csv. A run made before this protocol is logged the same way, with its own
        `seed_rule` (for example "own: seeds 0 1 2 for every prompt, CUDA generator") and its
        counts given directly, since its images have no rows in images.csv."""
        mine = [r for r in _read(self.root / "images.csv") if r["config_id"] == cfg.config_id and r["split"] == split]
        secs = [float(r["seconds"]) for r in mine if r["seconds"]]
        if sec_per_image is not None:
            secs = [sec_per_image]
        same = lambda r: r["config_id"] == cfg.config_id and r["split"] == split and r["owner"] == self.owner  # noqa: E731
        earlier = [r["started"] for r in _read(self.root / "runs.csv") if same(r) and r["started"]]
        started = earlier[0] if earlier else started  # a resumed run keeps its first start time
        row = {"config_id": cfg.config_id, "owner": self.owner, "axis": self.axis, "split": split,
               "n_prompts": n_prompts, "seed_ids": " ".join(str(s) for s in seed_ids), "seed_rule": seed_rule,
               "sampler": cfg.sampler, "schedule": cfg.schedule, "steps": cfg.steps, "unet_calls": cfg.unet_calls,
               "guidance": cfg.guidance, "dtype": dtype, **self.env,
               "sec_per_image": round(float(np.mean(secs)), 3) if secs else "",
               "n_images": len(mine) if n_images is None else n_images,
               "n_flagged": sum(int(r["flagged"]) for r in mine) if n_flagged is None else n_flagged,
               "started": started, "finished": _now(), "notes": notes}
        runs = [r for r in _read(self.root / "runs.csv") if not same(r)]
        _write(self.root / "runs.csv", RUN_COLUMNS, runs + [row])
        return row


def run_config(pipe, cfg: GenConfig, split: str, log: RunLog, items=None, seed_ids=(0,), first: int | None = None,
               every: int = 50, notes: str = "") -> Path:
    """Generate every prompt of `split` (or the DataFrame `items`, or its first `first` rows) under
    `cfg`, for each seed_id. Resumable: images already on disk are kept and not remade."""
    items = load_split(split) if items is None else items
    if first is not None:
        items = items.head(first)
    key_col, prompt_col, seed_col = split_columns(items)
    out = log.images_dir / split / cfg.config_id
    out.mkdir(parents=True, exist_ok=True)
    started, times = _now(), []
    for i, r in enumerate(items.itertuples(index=False)):
        key, prompt, seed_key = getattr(r, key_col), str(getattr(r, prompt_col)), int(getattr(r, seed_col))
        for sid in seed_ids:
            path = out / f"{key}_s{sid}.png"
            if path.exists():
                continue
            seed = seed_for(seed_key, sid)
            img, flagged, dt = generate(pipe, prompt, cfg, seed)
            img.save(path)
            log.image(cfg, split, key, sid, seed, path.relative_to(log.images_dir).as_posix(), flagged, dt)
            times.append(dt)
        if every and (i + 1) % every == 0:
            rate = f"{np.mean(times):.2f} s/image" if times else "resumed"
            print(f"{cfg.config_id} on {split}: {i + 1}/{len(items)} prompts, {rate}", flush=True)
    row = log.add_run(cfg, split, len(items), seed_ids, started=started,
                      dtype=str(pipe.unet.dtype).replace("torch.", ""), notes=notes)
    print(f"{cfg.config_id} on {split}: done, {row['n_images']} images, {row['n_flagged']} flagged, "
          f"{row['sec_per_image']} s/image", flush=True)
    return out


# ------------------------------------------------------------------ the Wednesday check

def smoke_check(pipe, dev=None, out_dir="m2_smoke", n: int = 10) -> dict:
    """Prints the starting-noise sum for the first dev caption and the anchor's timesteps (both
    must match on every machine), then paints the anchor for the first `n` dev captions. The
    pictures should look the same on every machine; tiny pixel differences across GPUs are normal."""
    dev = load_split("dev_500") if dev is None else dev
    first = int(dev.caption_idx.iloc[0])
    noise = start_noise(seed_for(first, 0))
    noise_sum = round(float(noise.double().sum()), 4)
    ts = visited_timesteps(ANCHOR, dict(getattr(pipe, "_m2_base_config", pipe.scheduler.config)))
    ok_noise = first == SMOKE_CAPTION_IDX and noise_sum == EXPECTED_NOISE_SUM
    ok_ts = ts == EXPECTED_ANCHOR_TIMESTEPS
    print(f"protocol {PROTOCOL_VERSION} ({protocol_sha()}), torch {torch.__version__}")
    print(f"first dev caption_idx {first}, noise sum {noise_sum:.4f}: {'OK' if ok_noise else 'MISMATCH'}")
    print(f"anchor timesteps {ts[:3]} ... {ts[-2:]} ({len(ts)}): {'OK' if ok_ts else 'MISMATCH'}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in dev.head(n).itertuples(index=False):
        img, flagged, dt = generate(pipe, r.caption, ANCHOR, seed_for(r.caption_idx, 0))
        img.save(out / f"{r.caption_idx}_s0.png")
        mean = float(np.asarray(img, dtype=np.float32).mean())
        rows.append({"caption_idx": int(r.caption_idx), "mean_pixel": round(mean, 2), "flagged": flagged,
                     "seconds": round(dt, 2)})
        print(f"  {r.caption_idx}: mean pixel {mean:.2f}, flagged {flagged}, {dt:.2f} s")
    return {"noise_ok": ok_noise, "timesteps_ok": ok_ts, "noise_sum": noise_sum, "images": rows}


if __name__ == "__main__":
    print(f"protocol {PROTOCOL_VERSION} ({protocol_sha()})")
    planned = [GenConfig()] + [GenConfig(schedule=s) for s in ("linspace", "trailing")] + \
              [GenConfig(sampler="pndm"), GenConfig(sampler="dpmpp2m")] + \
              [GenConfig(sampler="dpmpp2m", schedule=s, steps=n) for n in (10, 30)
               for s in ("leading", "trailing", "karras", "ays")]
    for cfg in planned:
        ts = visited_timesteps(cfg)
        print(f"{cfg.config_id:28s} {cfg.unet_calls:3d} calls  {ts}")
