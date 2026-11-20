"""Milestone 2, the development grid: how do spacing, guidance and the seed move when a word
commits and whether its object appears? Exploratory (not pre-registered).

Reads the seven recorded runs of the development grid on the first 200 dev captions and compares each with
the anchor (DDIM, leading spacing, guidance 7.5, seed_id 0), object by object:

  missing rate     the share of objects both judges miss, and its paired change from the anchor
  commit step      which of the 30 steps the word commits at, paired change
  commit timestep  the noise level it commits at (the timestep of that step), paired change.
                   Spacings put the 30 steps at different noise levels, so if a word commits at
                   the same noise level whatever the spacing, the step moves but the timestep
                   doesn't: the measure then tracks the noise level, not the step count.
  agreement        Spearman correlation of commit steps between the setting and the anchor

Seeds 1 and 2 also give a development-split version of the within-object test (the
pre-registered word-timing test on eval_2000). All intervals: 2,000 bootstrap resamples of captions.

Run from the project folder:  python scripts/m2_grid.py
Writes Results/m2_grid/summary.json and Results/m2_grid/comparison.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m2_timing as mt  # noqa: E402
from w2w.paths import RESULTS  # noqa: E402

SPLIT, FIRST = "dev_500", 200
ANCHOR = ("ddim-leading-s30-g7.5", 0)
SETTINGS = [("ddim-linspace-s30-g7.5", 0, "spacing"), ("ddim-trailing-s30-g7.5", 0, "spacing"),
            ("ddim-leading-s30-g3", 0, "guidance"), ("ddim-leading-s30-g12", 0, "guidance"),
            ("ddim-leading-s30-g7.5", 1, "seed"), ("ddim-leading-s30-g7.5", 2, "seed")]
KEY = mt.OBJECT_KEY
N_BOOT, SEED = mt.N_BOOT, mt.BOOT_SEED


def run_path(config_id: str, seed_id: int) -> Path:
    return RESULTS / "m2_timing" / f"{SPLIT}_first{FIRST}__{config_id}__s{seed_id}" / "objects_judged.csv"


def load(config_id: str, seed_id: int) -> pd.DataFrame:
    d = pd.read_csv(run_path(config_id, seed_id))
    d = d[d.flagged == 0].copy()
    d["missing"] = (d.judge_group == "missing").astype(float)
    return d


def describe(d: pd.DataFrame) -> dict:
    settled = d[d.commit_t >= 0]
    return {"objects": int(len(d)), "captions": int(d.caption_idx.nunique()),
            "missing_rate": round(float(d.missing.mean()), 4),
            "median_commit_step": float(d.commit_08.median()),
            "median_commit_timestep": float(settled.commit_t.median()) if len(settled) else float("nan"),
            "never_settles": round(float(d.never_settles.mean()), 4),
            "mean_early_share": round(float(d.early.mean()), 4)}


def paired(anchor: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Paired changes (other minus anchor) over objects present in both runs, with caption-level
    bootstrap intervals."""
    m = anchor.merge(other, on=KEY, suffixes=("_a", "_b"))
    both = (m.commit_t_a >= 0) & (m.commit_t_b >= 0)
    cols = {"missing": m.missing_b - m.missing_a, "commit_step": (m.commit_08_b - m.commit_08_a).astype(float),
            "commit_timestep": (m.commit_t_b - m.commit_t_a).where(both).astype(float)}
    caps = m.caption_idx.to_numpy()
    uniq = np.unique(caps)
    code = np.searchsorted(uniq, caps)
    rng = np.random.default_rng(SEED)
    draws = [np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size)[code].astype(float)
             for _ in range(N_BOOT)]
    out = {"pairs": int(len(m))}
    for name, v in cols.items():
        v = v.to_numpy(float)
        ok = ~np.isnan(v)
        if not ok.any():
            out[name] = {"mean_change": float("nan"), "ci95": [float("nan")] * 2}
            continue
        boot = [(w[ok] * v[ok]).sum() / w[ok].sum() for w in draws if w[ok].sum() > 0]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {"mean_change": round(float(v[ok].mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "n": int(ok.sum())}
    out["commit_step_spearman"] = round(float(m.commit_08_a.corr(m.commit_08_b, method="spearman")), 3)
    return out


def main():
    out = RESULTS / "m2_grid"
    out.mkdir(parents=True, exist_ok=True)
    anchor = load(*ANCHOR)
    rows, summary = [], {"anchor": {"config_id": ANCHOR[0], "seed_id": ANCHOR[1], **describe(anchor)}, "settings": []}
    for cfg, s, axis in SETTINGS:
        d = load(cfg, s)
        res = {"config_id": cfg, "seed_id": s, "axis": axis, **describe(d), "vs_anchor": paired(anchor, d)}
        summary["settings"].append(res)
        p = res["vs_anchor"]
        rows.append({"axis": axis, "config_id": cfg, "seed_id": s, "objects": res["objects"],
                     "missing_rate": res["missing_rate"], "median_commit_step": res["median_commit_step"],
                     "median_commit_timestep": res["median_commit_timestep"],
                     "change_missing": p["missing"]["mean_change"], "change_missing_ci": p["missing"]["ci95"],
                     "change_commit_step": p["commit_step"]["mean_change"], "change_commit_step_ci": p["commit_step"]["ci95"],
                     "change_commit_timestep": p["commit_timestep"]["mean_change"],
                     "change_commit_timestep_ci": p["commit_timestep"]["ci95"], "spearman": p["commit_step_spearman"]})
    seeds = pd.concat([load(ANCHOR[0], s).assign(seed_id=s) for s in (0, 1, 2)], ignore_index=True)
    seeds["judge"] = seeds.judge_group
    summary["within_object_dev"] = {"commit": mt.within_object_test(seeds, mt.PRIMARY, +1.0),
                                    "early": mt.within_object_test(seeds, "early", -1.0)}
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False, lineterminator="\n")
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(pd.DataFrame(rows).drop(columns=[c for c in rows[0] if c.endswith("_ci")]).to_string(index=False))
    print(json.dumps(summary["within_object_dev"], indent=2))


if __name__ == "__main__":
    main()
