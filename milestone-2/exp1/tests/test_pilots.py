"""The judging and reading rules of pilots 4 and 5, on small hand-made tables (CPU, no models)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pilot_recall as rc  # noqa: E402
import pilot_rescue as rs  # noqa: E402


def test_verdict_and_auroc():
    assert rc.verdict(0.6, 0.0) == "present" and rc.verdict(0.1, 0.55) == "present"
    assert rc.verdict(0.29, 0.1) == "missing" and rc.verdict(0.4, 0.2) == "ambiguous"
    assert rc.auroc([3, 4], [1, 2]) == 1.0 and rc.auroc([1, 2], [3, 4]) == 0.0
    assert rc.auroc([2], [2]) == 0.5


def test_within_class_auroc_ignores_differences_between_classes():
    # Knives commit late and go missing; cars commit early and appear. Within each class,
    # commit step says nothing, so the overall AUROC is high but the within-class AUROC is 0.5.
    rows = [("knife", 20, "missing"), ("knife", 20, "present"), ("car", 5, "missing"), ("car", 5, "present"),
            ("knife", 20, "missing"), ("car", 5, "present")]
    df = pd.DataFrame([{"caption_idx": i, "object": o, "commit": c, "judge": j} for i, (o, c, j) in enumerate(rows)])
    assert rc.auroc(df[df.judge == "missing"].commit, df[df.judge == "present"].commit) > 0.5
    assert rc.within_class_auroc(df, "commit") == 0.5


def test_reading_rules():
    good = {"auroc": 0.7, "ci95": [0.6, 0.8], "within_class_auroc": 0.66, "within_class_ci95": [0.55, 0.75]}
    assert rc.read(good, 40) == "PREDICTS" and rc.read(good, 10) == "TOO_FEW"
    assert rc.read({**good, "within_class_ci95": [0.45, 0.7]}, 40) == "CLASS_ONLY"
    assert rc.read({**good, "auroc": 0.6}, 40) == "WEAK"
    assert rc.read({**good, "ci95": [0.48, 0.7]}, 40) == "NONE"


def _tries():
    """Two failures. Regeneration always fixes but changes everything; rescue from step 6 fixes
    failure 0 and keeps the scene; rescue from the other steps never fixes."""
    rows = []
    for fid in (0, 1):
        for cond in rs.CONDITIONS:
            for t in range(rs.T):
                fixed = cond in ("regen", "regen_eta1") or (cond == "rescue_06" and fid == 0)
                lp = 0.6 if cond.startswith("regen") else 0.1
                rows.append({"fid": fid, "cond": cond, "t": t, "fixed": fixed, "lpips": lp,
                             "clipscore": 30.0, "sharpness": 100.0})
    return pd.DataFrame(rows)


def test_rescue_analysis():
    pf, s = rs.analyse(_tries())
    assert s["tau"] == 0.3
    tab = s["by_condition"]
    assert tab["regen"]["fix_at_T"] == 1.0 and tab["regen"]["kept_at_T"] == 0.0
    assert tab["rescue_06"]["fix_at_T"] == 0.5 and tab["rescue_06"]["kept_at_T"] == 0.5
    assert s["best_rescue"] == "rescue_06"
    assert s["fix_gap_regen_minus_best"]["diff"] == 0.5
    assert s["sampler"]["verdict"] == "SAME"
    assert np.isnan(pf[(pf.fid == 1) & (pf.cond == "rescue_06")].best_lpips_fixed).all()


def test_rescue_reading():
    tab = {c: {"kept_at_T": 0.0} for c in rs.CONDITIONS}
    assert rs.read(tab, {"ci95": [0.1, 0.3]}, {"diff": 0.0}) == "NO_RESCUE"
    tab["rescue_09"]["kept_at_T"] = tab["rescue_12"]["kept_at_T"] = 0.5
    assert rs.best_rescue(tab) == "rescue_12"
    assert rs.read(tab, {"ci95": [0.1, 0.3]}, {"diff": 0.05}) == "BOTH"
    assert rs.read(tab, {"ci95": [0.1, 0.3]}, {"diff": 0.2}) == "TRADE_OFF"
    assert rs.read(tab, {"ci95": [-0.1, 0.3]}, {"diff": 0.05}) == "UNCLEAR"
