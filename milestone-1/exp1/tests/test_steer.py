import math

import pytest
import torch

from w2w.steer import SteerController, SteerCrossAttnProcessor, box_mask, build_bias

try:
    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
except ImportError:  # newer Diffusers layouts
    from diffusers.models.attention import Attention
    from diffusers.models.attention_processor import AttnProcessor2_0


def _attn():
    torch.manual_seed(0)
    return Attention(query_dim=16, cross_attention_dim=12, heads=2, dim_head=8).eval()


def test_box_mask_right_half():
    m = box_mask(8, (0.5, 0.0, 1.0, 1.0))
    assert m[:, 4:].all() and not m[:, :4].any()


def test_box_mask_rows_are_y():
    m = box_mask(8, (0.0, 0.0, 1.0, 0.25))
    assert m[:2].all() and not m[2:].any()


def test_bias_only_on_conditional_slot():
    s = 5.0
    b = build_bias(64, 77, 2, [3], (0.5, 0.0, 1.0, 1.0), s)
    assert b.shape == (2, 1, 64, 77)
    assert b[0].abs().sum() == 0                      # unconditional half untouched
    inside = box_mask(8, (0.5, 0.0, 1.0, 1.0)).flatten()
    assert torch.allclose(b[1, 0, inside, 3], torch.full((int(inside.sum()),), math.log(s)))
    assert torch.allclose(b[1, 0, ~inside, 3], torch.full((int((~inside).sum()),), -math.log(s)))
    assert b[1, 0, :, [0, 1, 2, 4]].abs().sum() == 0   # other slots untouched


def test_controller_switches_on_at_start():
    c = SteerController()
    c.configure(slots=[2], strength=4.0, start=3)
    c.eval_idx = 2
    assert not c.active()
    c.eval_idx = 3
    assert c.active()
    c.configure(slots=[2], strength=1.0, start=0)
    c.eval_idx = 5
    assert not c.active()


@torch.no_grad()
def test_zero_bias_matches_stock_processor():
    a = _attn()
    hs, enc = torch.randn(2, 64, 16), torch.randn(2, 77, 12)
    ref = AttnProcessor2_0()(a, hs, enc)
    out = SteerCrossAttnProcessor(SteerController())(a, hs, enc)
    assert torch.allclose(out, ref, atol=1e-5)


@torch.no_grad()
def test_steering_changes_only_the_conditional_half():
    a = _attn()
    hs, enc = torch.randn(2, 64, 16), torch.randn(2, 77, 12)
    ref = AttnProcessor2_0()(a, hs, enc)
    c = SteerController()
    c.configure(slots=[4], box=(0.5, 0.0, 1.0, 1.0), strength=10.0, start=0)
    c.eval_idx = 0
    out = SteerCrossAttnProcessor(c)(a, hs, enc)
    assert torch.allclose(out[0], ref[0], atol=1e-5)
    assert not torch.allclose(out[1], ref[1], atol=1e-3)


def test_box_loss_zero_when_all_attention_inside():
    from w2w.steer import box_loss
    box = (0.5, 0.0, 1.0, 1.0)
    inside = box_mask(8, box).flatten().float()
    a_in = (inside / inside.sum())[None, :, None]
    a_out = ((1 - inside) / (1 - inside).sum())[None, :, None]
    assert float(box_loss([a_in], box)) < 1e-6
    assert abs(float(box_loss([a_out], box)) - 1.0) < 1e-6
