"""Prepare executed notebooks for publishing: drop stderr streams (progress bars, warnings
that print local file paths), drop embedded images from the timelines notebook (its figures
are saved under Results/), and blank any remaining absolute local paths.

Run from the project root:  python scripts/clean_notebooks.py
"""
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
PATH_RE = re.compile(r"[A-Za-z]:\\\\?(?:[^\\\s'\"]+\\\\?)+")

for name, keep_images in [("01_data_and_embeddings.ipynb", True), ("02_word_timelines.ipynb", False)]:
    p = ROOT / "notebooks" / name
    nb = nbformat.read(p, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        kept = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                continue
            if not keep_images and out.get("output_type") in ("display_data", "execute_result"):
                data = out.get("data", {})
                if "image/png" in data:
                    out["data"] = {"text/plain": "[figure saved under Results/m1_word_timelines/]"}
            if out.get("output_type") == "stream":
                out["text"] = PATH_RE.sub("<local path>", out["text"])
            kept.append(out)
        cell["outputs"] = kept
    nbformat.write(nb, p)
    print(name, "cleaned,", round(p.stat().st_size / 1e6, 2), "MB")
