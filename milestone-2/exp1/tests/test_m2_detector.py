"""The early-warning features (scripts/m2_detector.py) on made-up attention, previews and logs (CPU, no image model)."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import m2_detector as md  # noqa: E402


def _two_objects(steps=4, n=16):
    """Slots: 0 start, 1 to 4 caption tokens, 5 end. Object A (slot 2) fills a 3 x 3 block,
    object B (slot 4) is spread evenly, and slots 1 and 3 get as little as B everywhere."""
    maps = np.full((steps, 6, n, n), 1e-4)
    maps[:, 0] = 0.9 / (n * n)
    maps[:, 2] = 0.0
    maps[:, 2, 4:7, 6:9] = 0.05
    objects = [{"slot": 2, "obj_slots": [2]}, {"slot": 4, "obj_slots": [4]}]
    return maps, objects


def test_pooling_keeps_each_token_mean():
    rng = np.random.default_rng(0)
    maps = rng.random((3, 77, 64, 64)).astype(np.float16)
    pooled = md.pool_maps(maps, keep=9)
    assert pooled.shape == (3, 9, 16, 16) and pooled.dtype == np.float16
    np.testing.assert_allclose(pooled.astype(np.float32).mean(axis=(2, 3)),
                               maps[:, :9].astype(np.float32).mean(axis=(2, 3)), atol=2e-3)


def test_attention_features_on_made_up_maps():
    maps, objects = _two_objects()
    a, b = md.attention_features(maps, content=[1, 2, 3, 4], objects=objects)
    assert a.shape == (4, len(md.ATT_NAMES))
    col = {n: i for i, n in enumerate(md.ATT_NAMES)}
    mass = maps.mean(axis=(2, 3))[0]
    share_a, share_b = mass[2] / mass[1:5].sum(), mass[4] / mass[1:5].sum()
    assert a[0, col["share"]] == pytest.approx(share_a) and b[0, col["share"]] == pytest.approx(share_b)
    assert a[0, col["focus"]] == pytest.approx(1 - np.log(9) / np.log(256), abs=1e-6)
    assert b[0, col["focus"]] == pytest.approx(0.0, abs=1e-6)
    assert a[0, col["peak"]] == pytest.approx(0.05 / 0.0503)        # inside A's block, after smoothing
    assert b[0, col["peak"]] == pytest.approx(1 / 3)                 # B, slot 1 and slot 3 share each position
    assert a[0, col["overlap"]] == pytest.approx(9 / 256) and b[0, col["overlap"]] == pytest.approx(9 / 256)
    assert a[0, col["lead"]] == pytest.approx(share_a / (share_a + share_b))
    assert np.isnan(a[0, col["steady"]]) and a[1:, col["steady"]] == pytest.approx(1.0)


def test_one_object_has_no_rival_and_no_overlap():
    maps, objects = _two_objects()
    (a,) = md.attention_features(maps, content=[1, 2, 3, 4], objects=objects[:1])
    col = {n: i for i, n in enumerate(md.ATT_NAMES)}
    assert (a[:, col["overlap"]] == 0).all() and (a[:, col["lead"]] == 1).all()


def test_features_at_a_decision_step_use_only_steps_so_far():
    curve = np.arange(30 * 6, dtype=float).reshape(30, 6)
    curve[0, 5] = np.nan                                           # steady has no value at step 0
    row = md.at_step(curve, 2)
    assert row["share_now@02"] == curve[2, 0] and row["share_mean@02"] == curve[:3, 0].mean()
    assert row["steady_mean@02"] == curve[1:3, 5].mean()
    assert "steady_now@00" not in md.at_step(curve, 0) and len(md.at_step(curve, 0)) == 10
    assert len(row) == 12


def test_preview_features_keep_the_best_score_so_far():
    steps = list(md.DECISION_STEPS)
    owl = np.linspace(0.05, 0.5, len(steps))
    owl[3] = 0.9                                                   # one good preview at step 3
    prev = pd.DataFrame({"step": steps, "frcnn": 0.0, "retina": 1.0, "owl": owl})
    row = md.preview_features(prev)
    assert row["owl_now@03"] == pytest.approx(md.logit(0.9)) and row["owl_max@20"] == pytest.approx(md.logit(0.9))
    assert row["owl_now@20"] == pytest.approx(md.logit(0.5))
    assert row["frcnn_now@00"] == pytest.approx(np.log(1e-4 / (1 - 1e-4)))   # clipped, not infinite
    assert np.isfinite(row["retina_max@20"])


def test_repaint_check_against_stored_measures():
    stored = pd.DataFrame([{"caption_idx": 7, "category": "dog", "word_start": 1, "commit_08": 5, "early": 0.31}])
    stored = stored.set_index(["caption_idx", "category", "word_start"])
    o = {"category": "dog", "word_start": 1, "commit_08": 5, "early": 0.3104}
    assert md.same_as_stored(stored, 7, o)
    assert not md.same_as_stored(stored, 7, {**o, "commit_08": 6})
    assert not md.same_as_stored(stored, 8, o)


def test_run_log_rows_get_their_real_seed_ids(tmp_path):
    cfg = md.CFG.config_id
    pd.DataFrame([{"config_id": cfg, "owner": "Santhosh", "axis": "word timing", "split": "eval_2000", "seed_ids": "2",
                   "n_images": "6", "n_flagged": "0", "sec_per_image": "3.3", "started": "a", "finished": "b",
                   "notes": "made with Painter"},
                  {"config_id": cfg, "owner": "Santhosh", "axis": "dev grid", "split": "dev_500", "seed_ids": "2",
                   "n_images": "3", "n_flagged": "0", "sec_per_image": "3.0", "started": "c", "finished": "d",
                   "notes": ""}]).to_csv(tmp_path / "runs.csv", index=False)
    im = [{"config_id": cfg, "split": "eval_2000", "key": k, "seed_id": s, "seed": 0, "file": "f", "flagged": 0,
           "seconds": 3.0} for k in (1, 2) for s in (0, 1, 2, 3)]
    im += [{"config_id": cfg, "split": "dev_500", "key": 5, "seed_id": s, "seed": 0, "file": "f", "flagged": 0,
            "seconds": 3.0} for s in (0, 1, 2)]
    pd.DataFrame(im).to_csv(tmp_path / "images.csv", index=False)
    runs = md.update_run_row(tmp_path, "the note", started="x").set_index("split")
    assert runs.loc["eval_2000", "seed_ids"] == "0 1 2 3" and runs.loc["eval_2000", "n_images"] == "8"
    assert runs.loc["eval_2000", "notes"] == "made with Painter; the note" and runs.loc["eval_2000", "started"] == "a"
    assert runs.loc["dev_500", "seed_ids"] == "0 1 2" and runs.loc["dev_500", "finished"] == "d"
    again = md.update_run_row(tmp_path, "the note", started="x").set_index("split")
    assert again.loc["eval_2000", "notes"] == "made with Painter; the note"


def test_features_phase_on_made_up_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "OUT", tmp_path / "out")
    monkeypatch.setattr(md, "REC", tmp_path / "rec")
    monkeypatch.setattr(md, "STORED", tmp_path / "stored")
    monkeypatch.setattr(md, "PARTS", {"development": {"rows": (0, 2), "seed_ids": (0, 1)}})
    maps, _ = _two_objects(steps=30)
    stored = tmp_path / "stored"
    for s in (0, 1):
        out = md.run_dir("development", s)
        (md.REC / f"s{s}").mkdir(parents=True)
        np.savez(md.rec_path(s, 11), maps=maps.astype(np.float16), content=np.array([1, 2, 3, 4]))
        objs = pd.DataFrame([{"seed_id": s, "caption_idx": 11, "category": c, "words": c, "word_start": w, "slot": sl,
                              "token": c, "n_tokens": 1, "commit_08": 4, "early": 0.2, "obj_slots": str(sl),
                              "n_words": 5, "n_objects": 2} for c, w, sl in (("cat", 1, 2), ("dog", 3, 4))])
        objs.to_csv(out / "objects.csv", index=False)
        pd.DataFrame([{"caption_idx": 11, "flagged": 0, "same_image": 1, "same_measures": 1}]).to_csv(
            out / "captions.csv", index=False)
        pd.DataFrame([{"caption_idx": 11, "category": c, "kind": kind, "step": k, "frcnn": 0.1,
                       "retina": 0.2, "owl": 0.3 if kind == "x0" else 0.8, "clip": 21.0}
                      for c in ("cat", "dog") for kind, steps in (("x0", md.DECISION_STEPS), ("draft", md.DRAFT_STEPS))
                      for k in steps]).to_csv(out / "previews.csv", index=False)
        d = stored / f"{md.SPLIT}__{md.CFG.config_id}__s{s}"
        d.mkdir(parents=True)
        pd.DataFrame([{"caption_idx": 11, "category": c, "word_start": w, "frcnn": 0.9, "retina": 0.9, "owl": 0.2,
                       "judge_group": "missing" if (c == "dog" and s == 1) else "present"}
                      for c, w in (("cat", 1), ("dog", 3))]).to_csv(d / "objects_judged.csv", index=False)
    df = md.features(types.SimpleNamespace(part="development"))
    assert len(df) == 4 and df.missing.tolist() == [0, 0, 0, 1]
    assert {"share_now@00", "steady_mean@20", "owl_max@20", "clip_now@03", "draft_owl_max@08",
            "final_retina"} <= set(df.columns)
    assert df["draft_owl_now@02"].iloc[0] == pytest.approx(md.logit(0.8)) and df["owl_now@02"].iloc[0] == pytest.approx(md.logit(0.3))
    assert df["clip_max@20"].iloc[0] == 21.0                        # CLIP enters as it is, not as a logit
    assert df.position.tolist() == [0.2, 0.6] * 2
    assert (md.OUT / "features_development.csv").exists()


# ------------------------------------------------------------------ the detectors and their measures

def test_measures_match_scikit_learn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(1)
    y = (rng.random(400) < 0.1).astype(int)
    s = np.round(rng.random(400) + 0.8 * y, 2)                      # rounded: ties on purpose
    ones = np.ones(400)
    assert md.auroc_w(s, y, ones) == pytest.approx(roc_auc_score(y, s))
    assert md.ap_w(s, y, ones) == pytest.approx(average_precision_score(y, s))
    present = np.arange(100.0)                                      # 100 present scores 0..99, 10 missing at 95
    s2, y2 = np.r_[present, np.full(10, 95.0)], np.r_[np.zeros(100), np.ones(10)].astype(int)
    assert md.catch_w(s2, y2, np.ones(110)) == 1.0                  # threshold 89: the 10 above it are flagged
    assert md.catch_w(np.r_[present, np.full(10, 50.0)], y2, np.ones(110)) == 0.0


def test_caption_folds_keep_a_caption_together():
    caps = np.repeat(np.arange(40), 3)
    folds = md.caption_folds(caps)
    assert all(len(set(folds[caps == c])) == 1 for c in range(40))
    assert sorted(np.bincount(folds).tolist()) == [24] * 5
    assert (md.caption_folds(caps) == folds).all()


def _made_up_features(n_caps, first_cap, seed):
    """Rows like features_<part>.csv: three seeds per caption; the preview and the attention share
    carry a signal about the label, the class carries another."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(first_cap, first_cap + n_caps):
        cats = ["person", "dog", "clock", "spoon", "car"]
        objs = [(cats[(c + j) % 5], 1 + 3 * j) for j in range(1 + c % 2)]
        for s in (0, 1, 2):
            for cat, w in objs:
                risk = rng.normal() + (1.0 if cat == "spoon" else 0.0)
                missing = int(risk > 1.4)
                row = {"seed_id": s, "caption_idx": c, "category": cat, "word_start": w, "n_objects": len(objs),
                       "position": w / 8, "n_words": 8, "n_tokens": 1, "is_person": int(cat == "person"),
                       "missing": missing, "judge": "missing" if missing else "present",
                       "final_retina": float(np.clip(0.9 - 0.8 * missing + 0.1 * rng.normal(), 0, 1))}
                for k in md.DECISION_STEPS:
                    for n in md.ATT_NAMES:
                        if not (n == "steady" and k == 0):
                            v = rng.normal() - (0.5 * risk if n == "share" else 0.0)
                            row[f"{n}_now@{k:02d}"], row[f"{n}_mean@{k:02d}"] = v, v
                    for j in md.SCORERS:
                        v = rng.normal() - risk * min(1.0, k / 10)
                        row[f"{j}_now@{k:02d}"], row[f"{j}_max@{k:02d}"] = v, v
                        if k in md.DRAFT_STEPS:
                            v = rng.normal() - 1.5 * risk
                            row[f"draft_{j}_now@{k:02d}"], row[f"draft_{j}_max@{k:02d}"] = v, v
                rows.append(row)
    return pd.DataFrame(rows)


def test_develop_lock_test_and_figures_on_made_up_features(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "OUT", tmp_path)
    monkeypatch.setattr(md, "DECISION_STEPS", (0, 2, 10))
    monkeypatch.setattr(md, "H3_STEPS", (2, 10))
    monkeypatch.setattr(md, "within", lambda df, s: md.mt.within_object_test(
        df.assign(judge=df.judge, risk=s), "risk", +1.0, n_boot=50))
    _made_up_features(150, 0, 0).to_csv(tmp_path / "features_development.csv", index=False)
    _made_up_features(100, 1000, 1).to_csv(tmp_path / "features_prospective.csv", index=False)
    args = types.SimpleNamespace(n_boot=40, no_film=True, device="cpu")
    with pytest.raises(SystemExit, match="lock"):
        md.prospective_test(args)                                  # nothing is tested before the lock
    dev = md.develop(args)
    curves = pd.read_csv(tmp_path / "development" / "curves.csv")
    assert len(curves) == len(md.detectors()) == 3 * 7 + 2 * 3          # draft families at steps 0 and 2 only
    assert set(curves[curves.detector == "everything"].step) == {0, 2}
    assert curves.set_index(["detector", "step"]).extra_unet_calls[("everything", 2)] == 10
    at = curves.set_index(["detector", "step"]).auroc
    assert at[("full", 10)] > at[("prompt", 10)] and at[("prompt+preview", 10)] > at[("prompt+attention", 10)]
    assert at[("prompt+draft", 0)] > at[("prompt+preview", 0)]          # the made-up drafts carry more signal
    assert dev["H4_draft"]["draft_reading"] in ("DRAFT_AHEAD", "NOT_SHOWN")
    assert dev["H1_painting_adds"]["reading"] in ("ADDS", "NOT_SHOWN") and "ceiling" in dev
    md.lock(args)
    with pytest.raises(SystemExit, match="already exists"):
        md.lock(args)
    res = md.prospective_test(args)
    assert res["lock"]["models_sha256"] == md.json.loads((tmp_path / "lock.json").read_text())["models_sha256"]
    assert res["H2_attention_first"]["reading"] in ("ATTENTION_FIRST", "NOT_SHOWN") and res["runs_logged"] == 1
    md.figures(args)
    assert (tmp_path / "how_early.png").exists() and (tmp_path / "within_object.png").exists()
    with open(tmp_path / "models.joblib", "ab") as f:              # a changed model file is refused
        f.write(b"x")
    with pytest.raises(SystemExit, match="differs"):
        md.prospective_test(args)


# ------------------------------------------------------------------ drafts and the CLIP scorer

def test_draft_timesteps_run_evenly_to_zero():
    assert md.draft_timesteps(925, 5) == [925, 740, 555, 370, 185]
    assert md.draft_timesteps(661, 5)[0] == 661 and len(md.draft_timesteps(100, 3)) == 3


@pytest.fixture(scope="module")
def tiny_painter():
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
    try:
        tok = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    except Exception as e:  # offline
        pytest.skip(f"tiny tokenizer unavailable: {e}")
    torch.manual_seed(0)
    te = CLIPTextModel(CLIPTextConfig(vocab_size=tok.vocab_size, hidden_size=32, intermediate_size=37,
                                      num_attention_heads=4, num_hidden_layers=2, max_position_embeddings=77))
    unet = UNet2DConditionModel(block_out_channels=(32, 64), layers_per_block=1, sample_size=8, in_channels=4,
                                out_channels=4, down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=32,
                                attention_head_dim=8)
    vae = AutoencoderKL(block_out_channels=(32, 64), in_channels=3, out_channels=3, latent_channels=4,
                        down_block_types=["DownEncoderBlock2D"] * 2, up_block_types=["UpDecoderBlock2D"] * 2)
    sched = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet, scheduler=sched,
                                   safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    from w2w.rewind import Painter
    return Painter(pipe, device="cpu", steps=10)


def test_a_draft_from_later_lands_nearer_the_finished_painting(tiny_painter):
    import torch
    p = tiny_painter
    tl = p.paint("a cat on a bench", 3, previews=False, record=False)
    pe, ne = p.embed("a cat on a bench")
    emb = torch.cat([ne, pe])
    sch, ts = md.draft_scheduler(p.pipe), p.scheduler.timesteps
    early = md.draft(p.pipe, sch, emb, tl.latents[1], int(ts[1]))
    late = md.draft(p.pipe, sch, emb, tl.latents[8], int(ts[8]))
    assert torch.equal(late, md.draft(p.pipe, sch, emb, tl.latents[8], int(ts[8])))     # deterministic
    final = tl.latents[10]
    assert (late - final).abs().mean() < (early - final).abs().mean()
    assert torch.isfinite(early).all() and early.shape == final.shape
    together = md.draft_many(p.pipe, [md.draft_scheduler(p.pipe) for _ in range(2)], emb,
                             [tl.latents[1], tl.latents[8]], [int(ts[1]), int(ts[8])])
    assert torch.allclose(together, torch.cat([early, late]), atol=1e-4)     # batched drafts match one by one


def test_clip_scorer_reads_a_stop_sign():
    from PIL import Image, ImageDraw
    try:
        from transformers import CLIPModel
        CLIPModel.from_pretrained(md.CLIP_ID, local_files_only=True)
    except Exception as e:
        pytest.skip(f"CLIP ViT-L/14 not in the local cache: {e}")
    classes = ["stop sign", "dog", "pizza", "bicycle", "clock"]
    im = Image.new("RGB", (224, 224), "white")
    d = ImageDraw.Draw(im)
    d.regular_polygon((112, 112, 90), 8, rotation=22.5, fill=(200, 20, 20))
    d.text((70, 100), "STOP", fill="white", font_size=36)
    (sc,) = md.ClipScorer("cpu", classes).scores([im])
    assert set(sc) == set(classes) and max(sc, key=sc.get) == "stop sign"
    assert all(-100 <= v <= 100 for v in sc.values())
