"""Text to embedding: the CLIP ViT-L/14 text encoder that Stable Diffusion 1.5 reads.

Two outputs per caption:
  * token states, shape (77, 768): what the UNet cross-attends to. Cached as a
    float16 memmap so later milestones never recompute them.
  * pooled, projected vector, shape (768,): CLIP's sentence vector, used for
    retrieval checks and CLIPScore. The UNet never sees it.
"""
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from .paths import CACHE

CLIP_ID = "openai/clip-vit-large-patch14"
MAX_LEN = 77
HIDDEN = 768


class TextEncoder:
    def __init__(self, device: str = "cuda", dtype=torch.float16):
        self.device, self.dtype = device, dtype
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_ID)
        self.model = CLIPModel.from_pretrained(CLIP_ID, dtype=dtype).to(device).eval()

    @torch.no_grad()
    def encode(self, captions: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (token_states [B,77,768], pooled [B,768], attention_mask [B,77])."""
        tok = self.tokenizer(captions, padding="max_length", max_length=MAX_LEN,
                             truncation=True, return_tensors="pt").to(self.device)
        # No attention mask, on purpose: SD 1.5's pipeline encodes prompts without one,
        # so padding positions carry context the UNet attends to. Matching that exactly
        # is what makes our cached states interchangeable with the pipeline's own.
        out = self.model.text_model(input_ids=tok.input_ids)
        states = out.last_hidden_state
        pooled = self.model.text_projection(out.pooler_output)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        return states, pooled, tok.attention_mask

    @torch.no_grad()
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.model.vision_model(pixel_values=pixel_values.to(self.device, self.dtype))
        feats = self.model.visual_projection(out.pooler_output)
        return feats / feats.norm(dim=-1, keepdim=True)


def cache_paths(name: str = "coco_val") -> dict[str, Path]:
    return {
        "states": CACHE / f"{name}_token_states_f16.npy",
        "pooled": CACHE / f"{name}_pooled_f16.npy",
        "mask": CACHE / f"{name}_mask_u8.npy",
        "null": CACHE / "null_prompt_states_f16.npy",
        "meta": CACHE / f"{name}_meta.json",
    }


def build_cache(encoder: TextEncoder, captions: list[str], name: str = "coco_val", batch: int = 256) -> dict[str, Path]:
    p = cache_paths(name)
    n = len(captions)
    states = np.lib.format.open_memmap(p["states"], mode="w+", dtype=np.float16, shape=(n, MAX_LEN, HIDDEN))
    pooled = np.lib.format.open_memmap(p["pooled"], mode="w+", dtype=np.float16, shape=(n, HIDDEN))
    mask = np.lib.format.open_memmap(p["mask"], mode="w+", dtype=np.uint8, shape=(n, MAX_LEN))
    for i in tqdm(range(0, n, batch), desc="encode captions", mininterval=30):
        s, v, m = encoder.encode(captions[i:i + batch])
        states[i:i + batch] = s.cpu().numpy().astype(np.float16)
        pooled[i:i + batch] = v.cpu().numpy().astype(np.float16)
        mask[i:i + batch] = m.cpu().numpy().astype(np.uint8)
    for arr in (states, pooled, mask):
        arr.flush()
    null, _, _ = encoder.encode([""])
    np.save(p["null"], null[0].cpu().numpy().astype(np.float16))
    json.dump({"n": n, "max_len": MAX_LEN, "hidden": HIDDEN, "model": CLIP_ID, "batch": batch}, open(p["meta"], "w", encoding="utf-8"))
    return p


def load_cache(name: str = "coco_val") -> dict[str, np.ndarray]:
    p = cache_paths(name)
    return {
        "states": np.load(p["states"], mmap_mode="r"),
        "pooled": np.load(p["pooled"], mmap_mode="r"),
        "mask": np.load(p["mask"], mmap_mode="r"),
        "null": np.load(p["null"]),
    }
