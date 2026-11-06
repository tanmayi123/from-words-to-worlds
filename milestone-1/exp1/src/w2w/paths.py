from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
COCO = DATA / "coco"
CACHE = DATA / "cache"
RESULTS = ROOT / "Results"

for p in (DATA, COCO, CACHE, RESULTS):
    p.mkdir(parents=True, exist_ok=True)
