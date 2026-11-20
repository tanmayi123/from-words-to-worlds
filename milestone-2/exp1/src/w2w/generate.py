"""Generation from cached embeddings.

Milestone 1 uses the stock Diffusers pipeline but feeds it our saved token states
through `prompt_embeds`, so the text-to-embedding step we built is the one that
drives the image. Later milestones replace the pipeline with our own loop.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def load_pipeline(device: str = "cuda"):
    pipe = StableDiffusionPipeline.from_pretrained(SD_ID, dtype=torch.float16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def generate_from_states(pipe, states: np.ndarray, null_states: np.ndarray, seed: int,
                         steps: int = 30, guidance: float = 7.5, device: str = "cuda"):
    """states: (77,768) float16 for one caption. Returns (PIL image, nsfw_flag, seconds)."""
    pe = torch.from_numpy(np.asarray(states)).to(device, torch.float16).unsqueeze(0)
    ne = torch.from_numpy(np.asarray(null_states)).to(device, torch.float16).unsqueeze(0)
    g = torch.Generator(device).manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, num_inference_steps=steps,
               guidance_scale=guidance, generator=g)
    torch.cuda.synchronize()
    flag = bool(out.nsfw_content_detected[0]) if out.nsfw_content_detected is not None else False
    return out.images[0], flag, time.perf_counter() - t0


def save_sample(img, out_dir: Path, name: str, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / f"{name}.png")
    json.dump(meta, open(out_dir / f"{name}.json", "w", encoding="utf-8"), indent=2)
