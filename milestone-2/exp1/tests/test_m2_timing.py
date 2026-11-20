"""Word timing at scale (scripts/m2_timing.py): word-to-token mapping, per-object measures,
judging rules, and the fast bootstrap against pilot 4's own implementation (CPU, no image model)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m2_timing as mt  # noqa: E402
import pilot_recall as rc  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from transformers import CLIPTokenizer
    try:
        return CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    except Exception as e:  # offline
        pytest.skip(f"CLIP tokenizer unavailable: {e}")


def test_object_slots_follow_whitespace_words(tok):
    cap = "A group of people, mostly children, are gathering around a tent."
    slots, slot, token = mt.object_slots(tok, cap, 3, 1)          # "people,"
    assert token == "people" and len(slots) == 2 and slot == slots[0]
    ids = tok(cap)["input_ids"]
    assert tok.convert_ids_to_tokens(ids[slot]) == "people</w>"
    _, slot, token = mt.object_slots(tok, "A teddy bear sitting on a bed.", 1, 2)   # "teddy bear"
    assert token == "bear"
    _, slot, token = mt.object_slots(tok, "a skateboarder rides down the ramp", 1, 1)
    assert token == "boarder" or token.endswith("er")                # a split word is measured on its last piece


def test_every_labelled_object_maps_to_a_token(tok):
    root = mt.SPLITS
    files = [root / f"groundtruth_{s}.csv" for s in ("eval_2000", "dev_500")]
    if not all(f.exists() for f in files):
        pytest.skip("label files not in Results/m2_splits")
    for f in files:
        g = pd.read_csv(f)
        g = g[g.target.astype(bool)]
        for r in g.itertuples(index=False):
            slots, slot, token = mt.object_slots(tok, r.caption, int(r.word_start), len(str(r.words).split()))
            assert 1 <= slot < 77 and token


def _maps(steps=30, settle=5, slot=2, never=False, seed=0):
    rng = np.random.default_rng(seed)
    maps = rng.random((steps, 77, 8, 8)).astype(np.float32) * 0.01
    layout = rng.random((8, 8)).astype(np.float32)
    for t in range(steps):
        maps[t, slot] = layout if (t >= settle and not never) else rng.random((8, 8))
    return maps.astype(np.float16)


def test_measure_objects_on_made_up_attention(tok):
    cap = "a dog on a sofa"
    labels = pd.DataFrame([{"caption_idx": 1, "image_id": 9, "category": "dog", "words": "dog", "word_start": 1,
                            "n_instances": 1, "cover_after_crop": 0.2}])
    ts = list(range(958, 0, -33))
    row = mt.measure_objects(tok, cap, _maps(settle=5, slot=2), ts, labels)[0]
    assert row["slot"] == 2 and row["token"] == "dog"
    assert row["commit_08"] == 5 and row["commit_t"] == ts[5] and not row["never_settles"]
    assert 0 < row["early"] <= 1
    curves = row["curves"]
    assert len(curves) == 2 * mt.STEPS and set(mt.CURVE_COLUMNS[4:]) == set(curves)
    assert all(curves[f"r_{t:02d}"] >= 0.8 for t in range(5, mt.STEPS - 1))
    row = mt.measure_objects(tok, cap, _maps(slot=2, never=True), ts, labels)[0]
    assert row["commit_08"] == mt.STEPS and row["commit_t"] == -1 and row["never_settles"]


def test_the_primary_test_leaves_out_captions_painted_before():
    if not (mt.SPLITS / "eval_2000.csv").exists():
        pytest.skip("splits not in Results/m2_splits")
    ev = pd.read_csv(mt.SPLITS / "eval_2000.csv")
    unseen = mt.unseen_captions("eval_2000")
    assert len(unseen) == 2000 - 432 and not unseen & set(ev.caption_idx.head(432))
    assert mt.unseen_captions("dev_500") == set(pd.read_csv(mt.SPLITS / "dev_500.csv").caption_idx)


def test_judging_rules():
    assert mt.judge_group(0.6, 0.0) == "present" and mt.judge_group(0.1, 0.2) == "present"
    assert mt.judge_group(0.49, 0.09) == "missing"
    assert mt.judge_frcnn(0.49, 0.9) == "missing"
    assert mt.judge_agree(0.6, 0.2) == "present" and mt.judge_agree(0.6, 0.05) == "ambiguous"
    assert mt.judge_agree(0.1, 0.05) == "missing"


def _table(n_caps=120, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_caps):
        for _ in range(rng.integers(1, 4)):
            obj = rng.choice(["cat", "dog", "car", "knife"])
            commit = int(rng.integers(0, 31))
            p_miss = 0.1 + 0.6 * (commit / 30) + (0.2 if obj == "knife" else 0)
            judge = rng.choice(["missing", "present", "ambiguous"], p=[p_miss * 0.8, 1 - p_miss, p_miss * 0.2])
            rows.append({"caption_idx": 1000 + c, "object": obj, "commit": commit,
                         "early": round(float(rng.random()), 2), "judge": judge})
    return pd.DataFrame(rows)


def _seeds(rows):
    return pd.DataFrame([{"caption_idx": c, "category": o, "word_start": 1, "seed_id": s, "commit_08": k,
                          "early": e, "judge": j} for c, o, s, k, e, j in rows])


def test_within_object_test_compares_seeds_of_the_same_object_only():
    # Object A: the missing seed commits later (a win). Object B: missing seed commits earlier (a loss).
    # Object C: missing in every seed and very late; object D present in every seed and early. C and D
    # differ hugely, but they are not discordant, so they must not count.
    rows = [(1, "cat", 0, 15, 0.1, "missing"), (1, "cat", 1, 5, 0.3, "present"), (1, "cat", 2, 6, 0.3, "present"),
            (2, "dog", 0, 4, 0.2, "missing"), (2, "dog", 1, 9, 0.2, "present"),
            (3, "kite", 0, 29, 0.0, "missing"), (3, "kite", 1, 28, 0.0, "missing"),
            (4, "bus", 0, 1, 0.9, "present"), (4, "bus", 1, 2, 0.9, "present")]
    r = mt.within_object_test(_seeds(rows), "commit_08", +1.0, n_boot=200)
    assert r["n_discordant"] == 2 and r["reading"] == "TOO_FEW"
    assert r["auroc"] == pytest.approx(2 / 3, abs=1e-3)                  # cat: 2 wins of 2 pairs; dog: 0 of 1
    assert r["mean_difference"] == pytest.approx(np.mean([15 - 5.5, 4 - 9]), abs=1e-3)
    e = mt.within_object_test(_seeds(rows), "early", -1.0, n_boot=200)
    assert e["auroc"] == pytest.approx((2 + 0.5) / 3, abs=1e-3)          # cat: lower share when missing; dog: a tie


def test_within_object_test_reads_a_clear_signal():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(80):
        late = int(rng.integers(0, 3))
        for s in range(3):
            rows.append((c, "cat", s, 20 if s == late else 5, 0.1, "missing" if s == late else "present"))
    r = mt.within_object_test(_seeds(rows), "commit_08", +1.0, n_boot=200)
    assert r["n_discordant"] == 80 and r["auroc"] == 1.0 and r["reading"] == "SEEN_COMING"


@pytest.mark.parametrize("col,sign", [("commit", 1.0), ("early", -1.0)])
def test_fast_bootstrap_gives_pilot_4s_numbers(col, sign):
    df = _table()
    ours = mt.test(df, col, sign, n_boot=300)
    theirs = rc.test_measure(df, col, sign, n_boot=300)
    for k in ("auroc", "ci95", "within_class_auroc", "within_class_ci95"):
        assert ours[k] == theirs[k], k
