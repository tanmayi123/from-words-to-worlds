"""The development-grid comparison (scripts/m2_grid.py) on made-up runs with known differences."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m2_grid as mg  # noqa: E402


def _run(rng, n_caps=60, shift=0, flip=0.0):
    rows = []
    for c in range(n_caps):
        for k in range(2):
            commit = int(rng.integers(0, 25))
            rows.append({"caption_idx": 5000 + c, "category": ["cat", "dog"][k], "word_start": k + 1,
                         "commit_08": commit, "commit_t": 958 - 33 * commit, "never_settles": False,
                         "early": 0.3, "flagged": 0, "judge_group": "missing" if commit > 18 else "present"})
    d = pd.DataFrame(rows)
    d["commit_08"] = d.commit_08 + shift
    d["commit_t"] = 958 - 33 * d.commit_08
    if flip:
        idx = d.sample(frac=flip, random_state=0).index
        d.loc[idx, "judge_group"] = "missing"
    return d


def test_grid_recovers_known_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "RESULTS", tmp_path)
    base = _run(np.random.default_rng(1))
    later = base.copy()
    later["commit_08"] += 2
    later["commit_t"] = 958 - 33 * later.commit_08
    runs = {mg.ANCHOR: base}
    for cfg, s, _ in mg.SETTINGS:
        runs[(cfg, s)] = later if (cfg, s) == mg.SETTINGS[0][:2] else base
    for (cfg, s), d in runs.items():
        p = mg.run_path(cfg, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(p, index=False)
    mg.main()
    summary = json.loads((tmp_path / "m2_grid" / "summary.json").read_text())
    first = summary["settings"][0]["vs_anchor"]
    assert first["commit_step"]["mean_change"] == 2 and first["commit_step"]["ci95"] == [2, 2]
    assert first["commit_timestep"]["mean_change"] == -66
    assert first["missing"]["mean_change"] == 0 and first["commit_step_spearman"] == 1.0
    same = summary["settings"][2]["vs_anchor"]
    assert same["commit_step"]["mean_change"] == 0 and same["missing"]["ci95"] == [0, 0]
    assert summary["within_object_dev"]["commit"]["n_discordant"] == 0       # identical seeds: nothing discordant
    table = pd.read_csv(tmp_path / "m2_grid" / "comparison.csv")
    assert len(table) == len(mg.SETTINGS)
