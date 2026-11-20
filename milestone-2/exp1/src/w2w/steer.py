"""Steer a word's position partway through generation.

The steering processor replaces each cross-attention layer's processor. It computes the
same thing as Diffusers' scaled-dot-product processor, plus an additive bias on the
attention scores of chosen text slots: +log(s) for latent pixels inside a target box and
-log(s) outside it. Adding log(s) to a score multiplies that slot's unnormalised attention
weight by s, so the word is pulled into the box and pushed out of everywhere else. Only
the conditional half of the guidance batch is biased; the unconditional half reads an
empty prompt and has nothing to steer.

Every run in an experiment, the unsteered baseline included, goes through this one code
path, with a zero bias wherever steering is off. The only difference between two
conditions is therefore the bias itself, not a change of attention kernel.

Placing objects through attention is established work (Directed Diffusion, Ma et al.
2023; training-free layout guidance, Chen et al., WACV 2024). What we vary is the
evaluation at which the bias switches on.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


class StopForward(Exception):
    """Raised inside the UNet once every attention map a guidance loss needs is collected."""


def box_mask(res: int, box: tuple) -> torch.Tensor:
    """(res, res) bool grid, True where a pixel centre lies in box = (x0, y0, x1, y1) in [0, 1].

    Rows are y (top to bottom) and columns are x, matching both the latent layout and the
    decoded image.
    """
    x0, y0, x1, y1 = box
    c = (torch.arange(res, dtype=torch.float32) + 0.5) / res
    in_x = (c >= x0) & (c < x1)
    in_y = (c >= y0) & (c < y1)
    return in_y[:, None] & in_x[None, :]


def build_bias(n_pixels: int, n_tokens: int, batch: int, slots, box, strength: float) -> torch.Tensor:
    """Additive score bias of shape (batch, 1, n_pixels, n_tokens), float32.

    With guidance the batch is [unconditional, conditional]; only the last element is biased.
    """
    res = int(round(math.sqrt(n_pixels)))
    assert res * res == n_pixels, f"expected a square latent grid, got {n_pixels} pixels"
    inside = box_mask(res, box).flatten()
    assert inside.any(), f"box {box} covers no pixel at {res}x{res}"
    v = math.log(strength)
    col = torch.where(inside, torch.tensor(v), torch.tensor(-v))
    bias = torch.zeros(batch, 1, n_pixels, n_tokens)
    for s in slots:
        bias[batch - 1, 0, :, s] = col
    return bias


@dataclass
class SteerController:
    """Shared state for every steering processor in the UNet.

    `eval_idx` counts UNet evaluations within one generation (a forward pre-hook on the
    UNet advances it; reset it to -1 before each run). Steering is on for evaluations
    `start` and later.
    """
    slots: list = field(default_factory=list)
    box: tuple = (0.0, 0.0, 1.0, 1.0)
    strength: float = 1.0
    start: int | None = None
    eval_idx: int = -1
    # Gradient guidance (see collect below): when `collect` is on, layers with at most
    # `collect_max_pixels` latent pixels store the attention of `slots`, with gradients,
    # in `collected`. After `collect_stop` layers the forward pass is cut short.
    collect: bool = False
    collect_max_pixels: int = 256
    collect_stop: int | None = None
    collected: list = field(default_factory=list, repr=False)
    _cache: dict = field(default_factory=dict, repr=False)

    def configure(self, slots=(), box=(0.0, 0.0, 1.0, 1.0), strength: float = 1.0, start: int | None = None):
        self.slots, self.box, self.strength, self.start = list(slots), tuple(box), float(strength), start
        self.eval_idx = -1
        self._cache.clear()

    def active(self) -> bool:
        return (self.start is not None and self.strength != 1.0 and len(self.slots) > 0
                and self.eval_idx >= self.start)

    def mask(self, n_pixels, n_tokens, batch, device, dtype) -> torch.Tensor:
        on = self.active()
        key = (on, n_pixels, n_tokens, batch, str(device), dtype)
        if key not in self._cache:
            if on:
                m = build_bias(n_pixels, n_tokens, batch, self.slots, self.box, self.strength)
            else:
                m = torch.zeros(batch, 1, n_pixels, n_tokens)
            self._cache[key] = m.to(device=device, dtype=dtype)
        return self._cache[key]


class SteerCrossAttnProcessor:
    """Cross-attention with an additive score bias (zero when steering is off)."""

    def __init__(self, ctrl: SteerController):
        self.ctrl = ctrl

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 temb=None, *args, **kwargs):
        assert encoder_hidden_states is not None, "steering processor is for cross-attention only"
        assert attention_mask is None, "SD 1.5 cross-attention uses no mask; refusing an unexpected one"
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        batch, n, _ = hidden_states.shape
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if getattr(attn, "norm_cross", None):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        if self.ctrl.collect and n <= self.ctrl.collect_max_pixels:
            scores = torch.matmul(query.float(), key.float().transpose(-1, -2)) * attn.scale
            probs = scores.softmax(dim=-1)                       # (batch, heads, n, tokens)
            self.ctrl.collected.append(probs[..., self.ctrl.slots].mean(1))   # (batch, n, len(slots))
            if self.ctrl.collect_stop is not None and len(self.ctrl.collected) >= self.ctrl.collect_stop:
                raise StopForward

        bias = self.ctrl.mask(n, key.shape[2], batch, query.device, query.dtype)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=bias, dropout_p=0.0,
                                             is_causal=False, scale=attn.scale)
        out = out.transpose(1, 2).reshape(batch, -1, attn.heads * head_dim).to(query.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(b, c, h, w)
        if getattr(attn, "residual_connection", False):
            out = out + residual
        return out / getattr(attn, "rescale_output_factor", 1.0)


def steering_processors(original: dict, ctrl: SteerController) -> dict:
    """Processor dict with every cross-attention layer replaced by a steering processor.

    Diffusers' set_attn_processor empties the dict it is given, so pass it a copy.
    """
    return {name: (SteerCrossAttnProcessor(ctrl) if "attn2" in name else proc)
            for name, proc in original.items()}


def attach_eval_counter(unet, ctrl: SteerController):
    """Forward pre-hook that advances ctrl.eval_idx once per UNet evaluation. Returns the handle."""
    def _hook(module, args):
        ctrl.eval_idx += 1
    return unet.register_forward_pre_hook(_hook)


def box_loss(collected: list, box: tuple) -> torch.Tensor:
    """Layout-guidance loss (Chen et al., WACV 2024): mean over layers of (1 - share of the
    word's attention that falls inside the box) squared. Expects one steered slot."""
    losses = []
    for a in collected:
        a = a[0, :, 0]                                     # (n,) for batch 1, slot 0
        res = int(round(math.sqrt(a.shape[0])))
        m = box_mask(res, box).flatten().to(a.device, a.dtype)
        losses.append((1.0 - (a * m).sum() / (a.sum() + 1e-8)) ** 2)
    return torch.stack(losses).mean()


def guide_latents(unet, ctrl: SteerController, latents, t, cond, slot: int, box, eta: float, iters: int):
    """Nudge the latents so the word's attention moves into the box (backward guidance).

    Runs `iters` partial UNet passes on the conditional branch only, each followed by one
    gradient step of RMS size `eta` on the latents. Returns (new latents, last loss).
    The first call counts how many low-resolution layers exist; later calls stop the
    forward pass as soon as those layers are collected.
    """
    ctrl.configure(slots=[slot], strength=1.0, start=None)
    x = latents.detach()
    loss_val = float("nan")
    try:
        ctrl.collect = True
        with torch.enable_grad():
            for _ in range(iters):
                ctrl.collected = []
                x = x.detach().requires_grad_(True)
                try:
                    unet(x, t, encoder_hidden_states=cond)
                except StopForward:
                    pass
                if ctrl.collect_stop is None:
                    ctrl.collect_stop = len(ctrl.collected)
                assert ctrl.collected, "no low-resolution cross-attention layer was collected"
                loss = box_loss(ctrl.collected, box)
                (g,) = torch.autograd.grad(loss, x)
                g = g.float()
                rms = g.pow(2).mean().sqrt()
                if not torch.isfinite(rms) or rms == 0:
                    break
                x = (x.float() - eta * g / rms).to(latents.dtype)
                loss_val = float(loss.detach())
    finally:
        ctrl.collect = False
        ctrl.collected = []
    return x.detach(), loss_val
