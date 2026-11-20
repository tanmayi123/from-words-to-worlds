"""The group log merge (scripts/m2_merge_logs.py) on one tiny log of each format."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m2_merge_logs as ml  # noqa: E402


def test_every_format_lands_in_one_table(tmp_path):
    own = tmp_path / "m2_runs"
    own.mkdir()
    pd.DataFrame([{"config_id": "ddim-leading-s30-g7.5", "split": "eval_2000", "key": 5, "seed_id": 1, "seed": 100005,
                   "file": "eval_2000/x/5_s1.png", "flagged": 0, "seconds": 3.6}]).to_csv(own / "images.csv", index=False)
    pd.DataFrame([{"config_id": "ddim-leading-s30-g7.5", "owner": "Santhosh", "split": "eval_2000", "sampler": "ddim",
                   "schedule": "leading", "steps": 30, "guidance": 7.5, "gpu": "RTX 5070",
                   "seed_rule": "m2: rule"}]).to_csv(own / "runs.csv", index=False)
    team = tmp_path / "m2_team_logs"
    (team / "mukul").mkdir(parents=True)
    (team / "praniti").mkdir(parents=True)
    pd.DataFrame([{"run_id": "r", "model": "base", "sampler": "ddim", "beta": "linear", "spacing": "leading", "cfg": 7.5,
                   "steps": 30, "prompt_idx": 0, "prompt": "two dogs", "seed": 1, "sec_per_img": 4.3, "blank": False,
                   "path": "/content/a/b/p00_s1.png", "clipscore": 25.0}]).to_csv(
        team / "mukul" / "mukul_generation_log.csv", index=False)
    pd.DataFrame([{"setting": "dpmpp_g3_st30", "prompt_id": "counting_00", "family": "counting", "prompt": "two cats",
                   "seed": 0, "guidance": 3.0, "scheduler": "dpmpp", "steps": 30, "seconds": 5.0, "blacked_out": True,
                   "path": "/content/r/s/counting_00_s0.png", "gpu": "Tesla T4", "time": "t"}]).to_csv(
        team / "praniti" / "praniti_generation_log.csv", index=False)
    images = ml.collect(team, own)
    assert list(images.columns) == ml.COLUMNS and len(images) == 3
    ids = dict(zip(images.owner, images.config_id))
    assert ids == {"Santhosh": "ddim-leading-s30-g7.5", "Mukul": "ddim-leading-s30-g7.5-beta_linear",
                   "Praniti": "dpmpp2m-linspace-s30-g3"}
    assert images.set_index("owner").flagged.to_dict() == {"Santhosh": 0, "Mukul": 0, "Praniti": 1}
    runs = ml.summarise(images)
    assert runs.set_index("owner").seed_ids.to_dict() == {"Santhosh": "1", "Mukul": "1", "Praniti": "0"}
    assert ml.to_markdown(runs).count("\n") == 2 + len(runs)
