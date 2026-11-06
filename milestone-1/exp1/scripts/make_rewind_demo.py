"""Build the data for the rewind viewer.

For each example, paint the original prompt once, then branch from it with the edited
prompt at several rewind points. Every timeline keeps the model's guess of the finished
picture at every step, plus where each word's attention falls, so the viewer can play
the painting as it forms and show the fork at the rewind point.

Output (open index.html in a browser; it works from the file system):
  data/rewind_demo/index.html   the viewer (copied from viewer/rewind.html)
  data/rewind_demo/demo.js      window.DEMO = {...}: prompts, words, attention, scores
  data/rewind_demo/sprites/     one JPEG per timeline: its frames in a 6-column grid
  data/rewind_demo/finals/      the finished pictures at 512 x 512

Scores shown in the viewer (estimates, labelled as such there):
  edit   CLIP: (s(branch) - s(original)) / (s(edit from the start) - s(original)), where
         s = cos(image, edited prompt) - cos(image, original prompt). 1 = as strong as
         editing from the start, 0 = no trace of the edit. Task 10 found this generous
         for small colour edits.
  scene  How much of the original picture survives: 1 - (mean absolute difference from the
         original at 64 x 64 greyscale) / (the same for the edit-from-the-start picture).

Run from the project folder (milestone-1/exp1):  python scripts/make_rewind_demo.py
"""
import argparse
import base64
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from w2w.attention import attention_share, is_content, word_tokens
from w2w.paths import DATA, ROOT
from w2w.rewind import Painter

STEPS, GUIDANCE = 30, 7.5
BRANCH_AT = (0, 3, 6, 9, 12, 15)
FRAME, COLS, MAP = 256, 6, 32
# Rewind point each example opens on, chosen by eye in task 11 (the automatic scores were too generous).
PICKS = {"bicycle": 6, "sunglasses": 9, "snow": 12, "hat": 6, "catdog": 9}

# id, original prompt, edited prompt, seed, kind (seeds and edits taken from task 10's units)
EXAMPLES = [
    ("bicycle", "a bicycle parked next to a tree", "a motorcycle parked next to a tree", 0, "swap"),
    ("sunglasses", "a dog lying on a couch", "a dog wearing sunglasses lying on a couch", 0, "accessory"),
    ("snow", "a car parked on a quiet street", "a car covered in snow parked on a quiet street", 1, "texture"),
    ("hat", "a cat sitting on a wooden bench", "a cat wearing a red hat sitting on a wooden bench", 1, "accessory"),
    ("catdog", "a cat sitting on a wooden bench", "a dog sitting on a wooden bench", 0, "swap"),
]


def sprite(tl, path):
    steps = sorted(tl.previews)
    rows = (len(steps) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * FRAME, rows * FRAME), "black")
    for j, s in enumerate(steps):
        sheet.paste(tl.previews[s].resize((FRAME, FRAME), Image.BICUBIC), ((j % COLS) * FRAME, (j // COLS) * FRAME))
    sheet.save(path, quality=82, optimize=True)
    return {"cols": COLS, "rows": rows, "frame": FRAME, "steps": steps}


def words(tok, prompt, maps):
    """Content words with their per-step attention share and per-step heatmaps (uint8, base64)."""
    toks = [(i, w) for i, w in word_tokens(tok, prompt) if is_content(w)]
    if not toks:
        return []
    share = attention_share(maps, [i for i, _ in toks])            # (steps, words)
    out = []
    for j, (slot, w) in enumerate(toks):
        m = torch.from_numpy(maps[:, slot].astype(np.float32))[:, None]
        m = torch.nn.functional.interpolate(m, size=(MAP, MAP), mode="area")[:, 0].numpy()
        m = m - m.min(axis=(1, 2), keepdims=True)
        m = m / (m.max(axis=(1, 2), keepdims=True) + 1e-8)
        out.append({"text": w, "slot": int(slot), "share": [round(float(x), 4) for x in share[:, j]],
                    "maps": base64.b64encode((m * 255).round().astype(np.uint8).tobytes()).decode()})
    return out


def grey64(img):
    return np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="testing only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-clip", action="store_true", help="testing only")
    ap.add_argument("--limit", type=int, default=None, help="testing only")
    ap.add_argument("--out", default="rewind_demo")
    args = ap.parse_args()

    from diffusers import StableDiffusionPipeline
    from w2w.generate import SD_ID
    dtype = torch.float16 if args.device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(args.model or SD_ID, dtype=dtype).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    painter = Painter(pipe, args.device, STEPS, GUIDANCE)
    tok = pipe.tokenizer

    clip = None
    if not args.no_clip:
        sys.path.insert(0, str(ROOT / "scripts"))
        from pilot_editing import Clip
        clip = Clip(args.device)

    out = DATA / args.out
    (out / "sprites").mkdir(parents=True, exist_ok=True)
    (out / "finals").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Parity: our loop against the stock pipeline on the first example.
    pid, src, _, seed, _ = EXAMPLES[0]
    pe, ne = painter.embed(src)
    ref = pipe(prompt_embeds=pe, negative_prompt_embeds=ne, latents=painter.noise(seed),
               num_inference_steps=STEPS, guidance_scale=GUIDANCE).images[0]
    ours = painter.paint(src, seed, record=False, previews=False).final
    parity = float(np.abs(np.asarray(ref, np.float32) - np.asarray(ours, np.float32)).mean())
    print(f"parity: mean |pipeline - our loop| = {parity:.4f}", flush=True)

    demo = {"steps": STEPS, "guidance": GUIDANCE, "map": MAP, "parity": parity, "examples": []}
    for ex_id, src, edited, seed, kind in (EXAMPLES[: args.limit] if args.limit else EXAMPLES):
        orig = painter.paint(src, seed)
        o_name = f"{ex_id}_original"
        entry = {"id": ex_id, "kind": kind, "seed": seed, "pick": PICKS.get(ex_id), "original": {
            "prompt": src, "start": 0, "sprite": f"sprites/{o_name}.jpg", "final": f"finals/{o_name}.jpg",
            "grid": sprite(orig, out / "sprites" / f"{o_name}.jpg"), "words": words(tok, src, orig.maps)},
            "edited": edited, "branches": []}
        orig.final.save(out / "finals" / f"{o_name}.jpg", quality=90)

        branches = {k: painter.paint(edited, seed, branch_from=orig, at=k) for k in BRANCH_AT}
        g0 = grey64(orig.final)
        full_d = float(np.abs(grey64(branches[0].final) - g0).mean())
        s0 = clip.gap(orig.final, edited, src) if clip else None
        s_full = clip.gap(branches[0].final, edited, src) if clip else None
        for k, tl in branches.items():
            name = f"{ex_id}_k{k:02d}"
            tl.final.save(out / "finals" / f"{name}.jpg", quality=90)
            d = float(np.abs(grey64(tl.final) - g0).mean())
            edit = None
            if clip and s_full is not None and s_full - s0 > 1e-6:
                edit = round((clip.gap(tl.final, edited, src) - s0) / (s_full - s0), 3)
            entry["branches"].append({
                "at": k, "prompt": edited, "start": k, "sprite": f"sprites/{name}.jpg",
                "final": f"finals/{name}.jpg", "grid": sprite(tl, out / "sprites" / f"{name}.jpg"),
                "words": words(tok, edited, tl.maps), "edit": edit,
                "scene": round(1 - d / full_d, 3) if full_d > 0 else None})
        demo["examples"].append(entry)
        print(f"[{(time.time() - t0) / 60:4.1f} min] {ex_id}: " + ", ".join(
            f"k={b['at']} edit {b['edit']} scene {b['scene']}" for b in entry["branches"]), flush=True)

    (out / "demo.js").write_text("window.DEMO = " + json.dumps(demo) + ";\n", encoding="utf-8")
    shutil.copy(ROOT / "viewer" / "rewind.html", out / "index.html")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"done in {(time.time() - t0) / 60:.1f} min; {size:.1f} MB in {out}", flush=True)


if __name__ == "__main__":
    main()
