"""Milestone 2: one generation log for the whole group.

Merges every member's image log into Results/m2_log/:
  images_all.csv   one row per image
  runs_all.csv     one row per member, model, setting and prompt set
  runs_all.md      the same table, for the report's appendix

Inputs, one folder per member in Results/m2_team_logs/<member>/:
  runs.csv and images.csv        written by the shared protocol (m2_protocol.RunLog)
  mukul_generation_log.csv       Mukul's first run, made before the protocol
  praniti_generation_log.csv     Praniti's runs, made before the protocol
This laptop's own protocol log (Results/m2_runs/) is added as Santhosh's.

Run from the project folder:  python scripts/m2_merge_logs.py
"""
import re
from pathlib import Path

import pandas as pd

from w2w.paths import RESULTS

COLUMNS = ["owner", "source", "model", "config_id", "sampler", "schedule", "beta_schedule", "steps", "guidance",
           "prompt_set", "key", "seed_id", "seed", "seed_rule", "flagged", "seconds", "gpu", "file"]
SAMPLER_NAMES = {"dpmpp": "dpmpp2m", "dpm++": "dpmpp2m", "dpmsolver": "dpmpp2m"}


def config_id(sampler, schedule, steps, guidance, beta="scaled_linear") -> str:
    cid = f"{sampler}-{schedule}-s{int(steps)}-g{float(guidance):g}"
    return cid if beta == "scaled_linear" else f"{cid}-beta_{beta}"


def from_protocol(folder: Path, owner: str) -> pd.DataFrame:
    """A RunLog folder: images.csv joined to runs.csv for each image's settings."""
    images, runs = pd.read_csv(folder / "images.csv"), pd.read_csv(folder / "runs.csv")
    runs = runs.drop_duplicates(["config_id", "split"], keep="last")
    m = images.merge(runs, on=["config_id", "split"], how="left", validate="many_to_one")
    return pd.DataFrame({
        "owner": owner, "source": "protocol", "model": "base", "config_id": m.config_id, "sampler": m.sampler,
        "schedule": m.schedule, "beta_schedule": "scaled_linear", "steps": m.steps, "guidance": m.guidance,
        "prompt_set": m.split, "key": m.key, "seed_id": m.seed_id, "seed": m.seed, "seed_rule": m.seed_rule, "flagged": m.flagged,
        "seconds": m.seconds, "gpu": m.gpu, "file": m.file})


def from_mukul(path: Path) -> pd.DataFrame:
    """Mukul's first run: 20 hand-written prompts, seeds 0 and 1, batches of 4 on a free Colab T4."""
    g = pd.read_csv(path)
    return pd.DataFrame({
        "owner": "Mukul", "source": "own log", "model": g.model, "sampler": g.sampler, "schedule": g.spacing,
        "beta_schedule": g.beta, "steps": g.steps, "guidance": g.cfg,
        "config_id": [config_id(s, sp, n, c, b) for s, sp, n, c, b in zip(g.sampler, g.spacing, g.steps, g.cfg, g.beta)],
        "prompt_set": "mukul_prompts_20", "key": g.prompt_idx, "seed_id": g.seed, "seed": g.seed,
        "seed_rule": "own: seeds 0 and 1 for every prompt, GPU generator, batches of 4",
        "flagged": g.blank.astype(str).str.lower().eq("true").astype(int), "seconds": g.sec_per_img,
        "gpu": "Tesla T4 (free Colab)", "file": g.path.map(lambda p: "/".join(Path(str(p)).parts[-2:]))})


def from_praniti(path: Path) -> pd.DataFrame:
    """Praniti's stress-prompt runs. Her schedulers came from SD's config via from_config, so DDIM and
    PNDM used leading spacing and DPM-Solver++ its own default, linspace."""
    g = pd.read_csv(path)
    sampler = g.scheduler.str.lower().map(lambda s: SAMPLER_NAMES.get(s, s))
    schedule = sampler.map(lambda s: "linspace" if s == "dpmpp2m" else "leading")
    return pd.DataFrame({
        "owner": "Praniti", "source": "own log", "model": "base", "sampler": sampler, "schedule": schedule,
        "beta_schedule": "scaled_linear", "steps": g.steps, "guidance": g.guidance,
        "config_id": [config_id(s, sp, n, c) for s, sp, n, c in zip(sampler, schedule, g.steps, g.guidance)],
        "prompt_set": "stress_prompts", "key": g.prompt_id, "seed_id": g.seed, "seed": g.seed,
        "seed_rule": "own: seeds 0, 1, 2 for every prompt, GPU generator",
        "flagged": g.blacked_out.astype(str).str.lower().eq("true").astype(int), "seconds": g.seconds, "gpu": g.gpu,
        "file": g.path.map(lambda p: "/".join(Path(str(p)).parts[-2:]))})


def collect(root: Path, own: Path | None) -> pd.DataFrame:
    parts = []
    if own is not None and (own / "images.csv").exists():
        parts.append(from_protocol(own, "Santhosh"))
    for member in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        owner = member.name.capitalize()
        if (member / "images.csv").exists():
            parts.append(from_protocol(member, owner))
        for f in member.glob("*generation_log*.csv"):
            name = f.name.lower()
            if name.startswith("mukul"):
                parts.append(from_mukul(f))
            elif name.startswith("praniti"):
                parts.append(from_praniti(f))
            else:
                raise ValueError(f"don't know the format of {f}")
    if not parts:
        raise FileNotFoundError("no logs found")
    return pd.concat(parts, ignore_index=True)[COLUMNS]


def summarise(images: pd.DataFrame) -> pd.DataFrame:
    keys = ["owner", "source", "model", "config_id", "prompt_set"]
    rows = []
    for k, g in images.groupby(keys, sort=False):
        rows.append({**dict(zip(keys, k)), "sampler": g.sampler.iloc[0], "schedule": g.schedule.iloc[0],
                     "steps": int(g.steps.iloc[0]), "guidance": float(g.guidance.iloc[0]),
                     "prompts": int(g.key.nunique()), "seed_ids": " ".join(map(str, sorted(g.seed_id.unique()))),
                     "images": int(len(g)),
                     "flagged": int(g.flagged.sum()), "sec_per_image": round(float(g.seconds.mean()), 2),
                     "gpu": g.gpu.iloc[0], "seed_rule": g.seed_rule.iloc[0]})
    return pd.DataFrame(rows)


def to_markdown(runs: pd.DataFrame) -> str:
    cols = ["owner", "model", "config_id", "prompt_set", "prompts", "seed_ids", "images", "flagged", "sec_per_image", "gpu"]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(re.sub(r"\|", "/", str(r[c])) for c in cols) + " |\n" for _, r in runs.iterrows())
    return head + body


def main():
    out = RESULTS / "m2_log"
    out.mkdir(parents=True, exist_ok=True)
    images = collect(RESULTS / "m2_team_logs", RESULTS / "m2_runs")
    runs = summarise(images)
    images.to_csv(out / "images_all.csv", index=False, lineterminator="\n")
    runs.to_csv(out / "runs_all.csv", index=False, lineterminator="\n")
    (out / "runs_all.md").write_text(to_markdown(runs), encoding="utf-8")
    print(f"{len(images)} images in {len(runs)} runs from {images.owner.nunique()} members")
    print(runs[["owner", "model", "config_id", "prompt_set", "images", "flagged", "sec_per_image"]].to_string(index=False))


if __name__ == "__main__":
    main()
