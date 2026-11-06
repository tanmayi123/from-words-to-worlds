"""Record how much attention each word receives, where, and at which denoising step.

Stable Diffusion's UNet has 16 cross-attention layers (named `...attn2`). In each one,
every latent pixel distributes attention over the 77 text slots. We replace those
layers' processors with one that computes the same output but also keeps the
attention probabilities of the conditional branch.

Aggregation follows DAAM (Tang et al., ACL 2023): per step, average over heads,
upsample every layer's map to the 64 x 64 latent grid, and sum over layers. Unlike
DAAM, we keep the steps separate, because when a word is attended to is the point.

Only cross-attention is touched; self-attention keeps its original processor.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

GRID = 64


@dataclass
class AttentionStore:
    """Holds per-step token maps: step -> tensor (77, 64, 64) float16 on CPU.

    A "step" here is one UNet evaluation. Most schedulers make one per inference step;
    PNDM, the SD 1.5 default, makes one extra, so N steps give N + 1 entries.
    """
    keep_steps: bool = True
    maps: dict = field(default_factory=dict)
    step: int = 0
    _acc: torch.Tensor | None = None
    _layers: int = 0

    def add(self, probs_cond: torch.Tensor) -> None:
        """probs_cond: (heads, pixels, 77) attention of the conditional branch for one layer."""
        heads, n, t = probs_cond.shape
        res = int(round(n ** 0.5))
        m = probs_cond.mean(0).float().T.reshape(t, res, res)  # (77, res, res)
        if res != GRID:
            m = F.interpolate(m[None], size=(GRID, GRID), mode="bilinear", align_corners=False)[0]
        self._acc = m if self._acc is None else self._acc + m
        self._layers += 1

    def end_step(self) -> None:
        if self._acc is not None:
            self.maps[self.step] = (self._acc / self._layers).half().cpu()
        self._acc, self._layers = None, 0
        self.step += 1

    def reset(self) -> None:
        self.maps, self.step, self._acc, self._layers = {}, 0, None, 0

    def stacked(self) -> np.ndarray:
        """All steps as one array, shape (steps, 77, 64, 64), float16."""
        return np.stack([self.maps[s].numpy() for s in sorted(self.maps)])


class RecordingCrossAttnProcessor:
    """Wraps the stock cross-attention processor: the image comes from the stock code path,
    and the attention probabilities are recomputed on the side for recording only.

    A first version computed the layer output itself, with a float32 softmax. On CPU it
    matched the stock output to 5e-6, but in float16 on the GPU the small differences
    compounded over 20 steps into a visibly different image (mean pixel difference 5.2).
    Delegating the output to the original processor makes the recorded image identical to
    the stock one; the probabilities we record never feed back into the generation.
    """

    def __init__(self, store: AttentionStore, inner, cond_only: bool = True):
        self.store = store
        self.inner = inner
        self.cond_only = cond_only

    @torch.no_grad()
    def _record(self, attn, hidden_states, encoder_hidden_states, attention_mask, temb):
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if hidden_states.ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)
        batch, n, _ = hidden_states.shape
        # One prompt at a time: batch 1 without guidance, 2 with it ([unconditional, conditional]).
        # A larger batch would average several prompts into one map, so it is refused.
        assert batch in (1, 2), f"record one prompt at a time (got batch {batch})"
        assert attention_mask is None, "SD 1.5 cross-attention uses no mask; refusing an unexpected one"
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if getattr(attn, "norm_cross", None):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # The pipeline stacks [unconditional, conditional] when guidance is on; record only
        # the conditional half, which is the branch that reads the prompt.
        sl = slice(batch // 2, batch) if (self.cond_only and batch % 2 == 0 and batch > 1) else slice(0, batch)
        hs, enc = hidden_states[sl], encoder_hidden_states[sl]
        heads = attn.heads
        q = attn.to_q(hs)
        k = attn.to_k(enc)
        d = q.shape[-1] // heads
        q = q.view(q.shape[0], n, heads, d).transpose(1, 2).float()
        k = k.view(k.shape[0], -1, heads, d).transpose(1, 2).float()
        scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            scores = scores + attention_mask[sl].float()
        for p in scores.softmax(dim=-1):
            self.store.add(p)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 temb=None, *args, **kwargs):
        if encoder_hidden_states is not None:
            self._record(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        return self.inner(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)


def install_recorder(unet, store: AttentionStore) -> dict:
    """Swap every cross-attention processor for the recorder. Returns the originals."""
    original = dict(unet.attn_processors)
    new = {name: (RecordingCrossAttnProcessor(store, inner=proc) if "attn2" in name else proc)
           for name, proc in original.items()}
    unet.set_attn_processor(new)
    return original


def remove_recorder(unet, original: dict) -> None:
    unet.set_attn_processor(original)


def step_callback(store: AttentionStore):
    """For `callback_on_step_end`: closes the current step's accumulation."""
    def _cb(pipe, i, t, kwargs):
        store.end_step()
        return kwargs
    return _cb


# ---------------------------------------------------------------- analysis

STOPWORDS = {"a", "an", "the", "of", "on", "in", "at", "to", "with", "and", "is", "are", "its",
             "it", "for", "by", "some", "very", "while", "into", "that", "this", "there", "his",
             "her", "their", "from", "next", "up", "down", "near", "over", "through", "each", "other",
             "around", "another", "be", "being", "has", "have"}


def is_content(word: str) -> bool:
    """Alphabetic and not a function word. Punctuation tokens like '.' are excluded."""
    return word.isalpha() and word.lower() not in STOPWORDS


def word_tokens(tokenizer, prompt: str) -> list[tuple[int, str]]:
    """(slot index, token text) for the prompt's real tokens, skipping start, end and padding."""
    ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True)["input_ids"]
    out = []
    for i, tid in enumerate(ids):
        if tid in (tokenizer.bos_token_id, tokenizer.eos_token_id):
            if tid == tokenizer.eos_token_id:
                break
            continue
        out.append((i, tokenizer.convert_ids_to_tokens(tid).replace("</w>", "")))
    return out


def attention_share(maps: np.ndarray, slots: list[int]) -> np.ndarray:
    """Share of word-directed attention each word gets per step, renormalised over real words.

    The start token absorbs most attention in CLIP-conditioned models (an attention sink),
    so raw probabilities understate every word. Renormalising over the prompt's own tokens
    gives the split among words. maps: (steps, 77, 64, 64). Returns (steps, len(slots)).
    """
    mass = maps[:, slots].astype(np.float32).mean(axis=(2, 3))
    return mass / mass.sum(axis=1, keepdims=True)


def commit_step(maps: np.ndarray, slot: int, ref_steps: int = 5, threshold: float = 0.8,
                ignore_last: int = 1) -> int | None:
    """First step after which the word's heatmap stays correlated with its final layout.

    Reference = mean map over the `ref_steps` evaluations before the last `ignore_last` ones.
    Returns the earliest step t such that the Pearson correlation with the reference is at
    least `threshold` at t and at every later step up to (not including) the ignored tail.
    None if it never settles.

    Why ignore the final evaluation: at the lowest noise level, almost every word's map
    decorrelates sharply from its layout (r drops from about 0.98 to about 0.75 in our
    first runs) while the image barely changes. Left in, that single step turns real
    commits into "never". The drop itself is kept as a separate observation
    (`final_step_drop`). The threshold is a parameter, not a constant: analyses report how
    conclusions move as it varies.
    """
    corr = correlation_curve(maps, slot, ref_steps=ref_steps, ignore_last=ignore_last)
    body = corr[: len(corr) - ignore_last] if ignore_last else corr
    ok = body >= threshold
    for t in range(len(ok)):
        if ok[t:].all():
            return t
    return None


def final_step_drop(maps: np.ndarray, slot: int, ref_steps: int = 5) -> float:
    """Correlation at the second-to-last evaluation minus at the last one."""
    corr = correlation_curve(maps, slot, ref_steps=ref_steps, ignore_last=1)
    return float(corr[-2] - corr[-1])


def correlation_curve(maps: np.ndarray, slot: int, ref_steps: int = 5, ignore_last: int = 1) -> np.ndarray:
    """Pearson r between the word's map at each step and its reference layout (see commit_step)."""
    if maps.shape[0] < ref_steps + ignore_last + 1:
        raise ValueError(f"need at least {ref_steps + ignore_last + 1} steps, got {maps.shape[0]}")
    seq = maps[:, slot].astype(np.float32).reshape(maps.shape[0], -1)
    end = len(seq) - ignore_last if ignore_last else len(seq)
    ref = seq[end - ref_steps:end].mean(0)
    ref = (ref - ref.mean()) / (ref.std() + 1e-8)
    z = (seq - seq.mean(1, keepdims=True)) / (seq.std(1, keepdims=True) + 1e-8)
    return (z * ref).mean(1)
