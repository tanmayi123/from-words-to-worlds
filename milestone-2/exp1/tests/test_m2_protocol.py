"""The Milestone 2 protocol: seeds, noise, schedules, the log, and agreement with Painter.

Schedule and seed tests need no model. The generation tests use a tiny random Stable Diffusion
on the CPU (a few seconds) and are skipped when its tokenizer can't be downloaded.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from w2w import m2_protocol as m2


# ------------------------------------------------------------------ no model needed

def test_seed_rule():
    assert m2.seed_for(3669) == 3669 and m2.seed_for(3669, 2) == 203669
    with pytest.raises(ValueError):
        m2.seed_for(100000)
    with pytest.raises(ValueError):
        m2.seed_for(5, -1)


def test_starting_noise_is_cpu_float32_and_repeatable():
    a, b, c = m2.start_noise(3669), m2.start_noise(3669), m2.start_noise(3670)
    assert a.dtype == torch.float32 and a.device.type == "cpu" and a.shape == (1, 4, 64, 64)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert round(float(a.double().sum()), 4) == m2.EXPECTED_NOISE_SUM


def test_config_ids_and_network_calls():
    assert m2.ANCHOR.config_id == "ddim-leading-s30-g7.5" and m2.ANCHOR.unet_calls == 30
    assert m2.GenConfig(guidance=3).config_id == "ddim-leading-s30-g3"
    assert m2.GenConfig(sampler="pndm").unet_calls == 31
    assert m2.GenConfig("dpmpp2m", "ays", 10).config_id == "dpmpp2m-ays-s10-g7.5"
    for bad in (dict(sampler="ddim", schedule="karras"), dict(sampler="euler"), dict(schedule="cosine"), dict(guidance=0.5)):
        with pytest.raises(ValueError):
            m2.GenConfig(**bad)


def test_schedules():
    assert m2.visited_timesteps(m2.ANCHOR) == m2.EXPECTED_ANCHOR_TIMESTEPS
    assert m2.visited_timesteps(m2.GenConfig(schedule="trailing"))[0] == 999
    pndm = m2.visited_timesteps(m2.GenConfig(sampler="pndm"))
    assert len(pndm) == 31 and len(set(pndm)) == 30
    assert m2.visited_timesteps(m2.GenConfig("dpmpp2m", "ays", 10)) == m2.AYS_SD15_TIMESTEPS
    leading10 = m2.visited_timesteps(m2.GenConfig("dpmpp2m", "leading", 10))
    assert leading10[0] < 950          # leading never starts from pure noise (Lin et al., 2024)
    for n in (10, 20, 30, 50):
        ts = m2.ays_timesteps(n)
        assert len(ts) == n and len(set(ts)) == n and ts == sorted(ts, reverse=True) and ts[0] == 999


def test_ays_follows_the_published_noise_levels():
    # the 30-step schedule passes through the published levels at every third step
    ts = m2.ays_timesteps(30)
    sig = m2.TRAIN_SIGMAS[ts]
    for k, s in enumerate(m2.AYS_SD15_SIGMAS[:10]):
        assert abs(np.log(sig[3 * k]) - np.log(s)) < 0.02


def test_a_run_made_before_the_protocol_can_be_logged(tmp_path):
    log = m2.RunLog(tmp_path, owner="Praniti", axis="sampler x guidance")
    log.add_run(m2.GenConfig("pndm", "leading", 30, 7.5), "stress_prompts", 200, (0, 1, 2),
                seed_rule="own: seeds 0 1 2 for every prompt, CUDA generator", n_images=600, n_flagged=2,
                sec_per_image=5.3)
    runs = pd.read_csv(tmp_path / "runs.csv")
    assert runs.n_images.iloc[0] == 600 and runs.n_flagged.iloc[0] == 2 and runs.sec_per_image.iloc[0] == 5.3
    assert runs.seed_rule.iloc[0].startswith("own") and runs.unet_calls.iloc[0] == 31


def test_stress_prompts_get_a_seed_key(tmp_path):
    pd.DataFrame({"prompt_id": ["word_order_00", "counting_00"], "family": ["word_order", "counting"],
                  "prompt": ["a bear following a sheep", "two dogs"]}).to_csv(tmp_path / "stress_prompts.csv", index=False)
    df = m2.load_split("stress_prompts", root=tmp_path)
    assert df.seed_key.tolist() == [0, 1]
    assert m2.split_columns(df) == ("prompt_id", "prompt", "seed_key")
    with pytest.raises(ValueError):
        m2.split_columns(df.drop(columns="seed_key"))


# ------------------------------------------------------------------ tiny pipeline

@pytest.fixture(scope="module")
def pipe():
    from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
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
    sched = PNDMScheduler.from_config(m2.sd15_scheduler_config())     # SD 1.5's own scheduler, as loaded
    p = StableDiffusionPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet, scheduler=sched,
                                safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    p.set_progress_bar_config(disable=True)
    return p


def _arr(img):
    return np.asarray(img, dtype=np.float32)


@pytest.mark.parametrize("cfg", [m2.GenConfig(steps=10), m2.GenConfig(schedule="trailing", steps=10),
                                 m2.GenConfig(sampler="pndm", steps=10), m2.GenConfig("dpmpp2m", "leading", 10),
                                 m2.GenConfig("dpmpp2m", "karras", 10), m2.GenConfig("dpmpp2m", "ays", 10),
                                 m2.GenConfig("dpmpp2m", "ays", 12, 3.0)], ids=lambda c: c.config_id)
def test_every_setting_generates_and_repeats(pipe, cfg):
    a, fa, _ = m2.generate(pipe, "a cat on a bench", cfg, 7)
    b, fb, _ = m2.generate(pipe, "a cat on a bench", cfg, 7)
    c, _, _ = m2.generate(pipe, "a cat on a bench", cfg, 8)
    assert np.array_equal(_arr(a), _arr(b)) and not np.array_equal(_arr(a), _arr(c))
    assert fa is False and fb is False
    assert [int(t) for t in pipe.scheduler.timesteps] == m2.visited_timesteps(cfg, pipe._m2_base_config)


def test_settings_do_not_leak_into_each_other(pipe):
    first = _arr(m2.generate(pipe, "a red bus", m2.GenConfig(steps=10), 1)[0])
    m2.generate(pipe, "a red bus", m2.GenConfig("dpmpp2m", "karras", 10), 1)
    again = _arr(m2.generate(pipe, "a red bus", m2.GenConfig(steps=10), 1)[0])
    assert np.array_equal(first, again)


@pytest.mark.parametrize("spacing", ["leading", "linspace", "trailing"])
def test_painter_paints_the_same_picture_as_the_protocol(pipe, spacing):
    from w2w.rewind import Painter
    cfg = m2.GenConfig(schedule=spacing, steps=10, guidance=7.5)
    ref, _, _ = m2.generate(pipe, "a dog under a tree", cfg, m2.seed_for(42))
    painter = Painter(pipe, device="cpu", steps=10, guidance=7.5, spacing=spacing, cpu_noise=True)
    tl = painter.paint("a dog under a tree", m2.seed_for(42), record=True, previews=False, safe=True)
    assert np.array_equal(_arr(ref), _arr(tl.final))
    assert tl.timesteps == m2.visited_timesteps(cfg, pipe._m2_base_config) and tl.flagged is False
    assert tl.maps.shape[0] == 10


def test_a_second_painter_does_not_nest_recorders(pipe):
    from w2w.attention import RecordingCrossAttnProcessor
    from w2w.rewind import Painter
    a = Painter(pipe, device="cpu", steps=10, spacing="leading", cpu_noise=True)
    a.paint("a cat", 1, record=True, previews=False)            # leaves recorders in place
    b = Painter(pipe, device="cpu", steps=10, spacing="trailing", cpu_noise=True)
    assert not any(isinstance(p, RecordingCrossAttnProcessor) for p in b.original.values())
    tl = b.paint("a cat", 1, record=True, previews=False)
    assert tl.maps.shape[0] == 10


def test_run_config_logs_resumes_and_counts(pipe, tmp_path):
    items = pd.DataFrame({"image_id": [1, 2, 3], "caption_idx": [10, 11, 12],
                          "caption": ["a cat", "a dog on a sofa", "two buses"]})
    log = m2.RunLog(tmp_path / "log", owner="Test", axis="guidance", images_dir=tmp_path / "img")
    cfg = m2.GenConfig(steps=10, guidance=3)
    out = m2.run_config(pipe, cfg, "dev_500", log, items=items, seed_ids=(0, 1), every=0)
    assert sorted(p.name for p in out.glob("*.png")) == ["10_s0.png", "10_s1.png", "11_s0.png", "11_s1.png",
                                                         "12_s0.png", "12_s1.png"]
    images = pd.read_csv(tmp_path / "log" / "images.csv")
    assert len(images) == 6 and set(images.seed) == {10, 11, 12, 100010, 100011, 100012}
    assert (tmp_path / "img" / images.file.iloc[0]).exists()
    first_start = pd.read_csv(tmp_path / "log" / "runs.csv").started.iloc[0]
    m2.run_config(pipe, cfg, "dev_500", log, items=items, seed_ids=(0, 1), every=0)     # nothing left to do
    runs = pd.read_csv(tmp_path / "log" / "runs.csv")
    assert len(runs) == 1 and runs.n_images.iloc[0] == 6 and runs.config_id.iloc[0] == "ddim-leading-s10-g3"
    assert runs.started.iloc[0] == first_start and runs.dtype.iloc[0] == "float32"
    assert runs.protocol_sha.iloc[0] == m2.protocol_sha() and runs.seed_rule.iloc[0] == m2.SEED_RULE
    assert len(pd.read_csv(tmp_path / "log" / "images.csv")) == 6
