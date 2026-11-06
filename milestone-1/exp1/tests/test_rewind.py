"""Rewind correctness on a tiny random Stable Diffusion (CPU, a few seconds).

Skipped when the tiny tokenizer can't be downloaded.
"""
import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def painter():
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


def _arr(img):
    return np.asarray(img, dtype=np.float32)


def test_matches_the_stock_pipeline(painter):
    pe, ne = painter.embed("a cat on a bench")
    ref = painter.pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=painter.noise(3),
                       num_inference_steps=10, guidance_scale=7.5).images[0]
    ours = painter.paint("a cat on a bench", 3, record=False, previews=False).final
    assert np.abs(_arr(ref) - _arr(ours)).max() == 0


def test_branching_with_the_same_prompt_reproduces_the_painting(painter):
    a = painter.paint("a cat on a bench", 1)
    b = painter.paint("a cat on a bench", 1, branch_from=a, at=4)
    assert b.start == 4 and sorted(b.previews) == list(range(4, 10))
    assert np.abs(_arr(a.final) - _arr(b.final)).max() == 0


def test_branching_with_a_new_prompt_changes_only_what_follows(painter):
    a = painter.paint("a cat on a bench", 1)
    b = painter.paint("a dog on a bench", 1, branch_from=a, at=4)
    assert torch.equal(a.latents[4], b.latents[4])
    assert not torch.equal(a.latents[10], b.latents[10])
    assert b.maps.shape[0] == 6                      # recorded steps 4 to 9 only


def test_stochastic_repaints_differ_and_repeat(painter):
    a = painter.paint("a cat on a bench", 1, previews=False)
    x = painter.paint("a cat on a bench", 1, branch_from=a, at=3, eta=1.0, noise_seed=7, previews=False, record=False)
    y = painter.paint("a cat on a bench", 1, branch_from=a, at=3, eta=1.0, noise_seed=7, previews=False, record=False)
    z = painter.paint("a cat on a bench", 1, branch_from=a, at=3, eta=1.0, noise_seed=8, previews=False, record=False)
    assert torch.equal(x.latents[10], y.latents[10])
    assert not torch.equal(x.latents[10], z.latents[10])


def test_local_blend_keeps_outside_the_mask(painter):
    a = painter.paint("a cat on a bench", 1, previews=False)
    b = painter.paint("a dog on a bench", 1, branch_from=a, at=3, previews=False,
                      blend={"orig": a, "slots_new": [2], "slots_orig": [2], "th": 0.6})
    m = b.masks[9]
    assert m.any() and not m.all()
    outside = ~m
    assert torch.equal(b.latents[10][..., outside], a.latents[10][..., outside])
