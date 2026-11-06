"""Caption-to-image retrieval over the whole validation set.

A sanity check with teeth: for each caption, does its own image rank first (or in
the top five) among all 5,000? Chance is 0.02 percent, so a healthy encoder
separates itself from a broken one by orders of magnitude.
"""
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor

from .data import IMAGES_DIR
from .encode import CLIP_ID, TextEncoder


@torch.no_grad()
def image_embeddings(encoder: TextEncoder, file_names: list[str], batch: int = 64) -> np.ndarray:
    proc = CLIPImageProcessor.from_pretrained(CLIP_ID)
    out = []
    for i in tqdm(range(0, len(file_names), batch), desc="encode images"):
        ims = [Image.open(IMAGES_DIR / f).convert("RGB") for f in file_names[i:i + batch]]
        px = proc(images=ims, return_tensors="pt").pixel_values
        out.append(encoder.encode_images(px).float().cpu().numpy())
    return np.concatenate(out)


def recall_at_k(text_emb: np.ndarray, image_emb: np.ndarray, caption_image_row: np.ndarray,
                ks=(1, 5, 10), chunk: int = 2048) -> dict[str, float]:
    """text_emb [C,768], image_emb [I,768], caption_image_row[c] = row of c's image in image_emb."""
    hits = {k: 0 for k in ks}
    ranks = []
    img = torch.from_numpy(np.array(image_emb, dtype=np.float32))
    for i in range(0, len(text_emb), chunk):
        t = torch.from_numpy(np.asarray(text_emb[i:i + chunk], dtype=np.float32))
        sims = t @ img.T
        true = torch.from_numpy(caption_image_row[i:i + chunk])
        true_sim = sims[torch.arange(len(t)), true].unsqueeze(1)
        rank = (sims > true_sim).sum(1)  # number of images scoring above the true one
        ranks.append(rank.numpy())
        for k in ks:
            hits[k] += int((rank < k).sum())
    ranks = np.concatenate(ranks)
    res = {f"recall@{k}": hits[k] / len(text_emb) for k in ks}
    res["median_rank"] = float(np.median(ranks) + 1)
    res["n_captions"] = int(len(text_emb))
    res["n_images"] = int(len(image_emb))
    return res
