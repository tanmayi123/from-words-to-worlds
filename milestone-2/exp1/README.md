# milestone-2 / exp1: When words are painted, and seeing a missing object coming

**Author:** Santhoshkumar Chandrasekar · IE 7615 Deep Learning for AI · Northeastern University

Part of [From Words to Worlds](../../README.md). Continues [`milestone-1/exp1`](../../milestone-1/exp1), where the attention recorder, the commit measure and the rewind loop were built and piloted. All paintings use the shared protocol in [`../protocol`](../protocol) and the captions in [`../splits`](../splits).

Stable Diffusion often leaves out an object the caption names. This experiment asks three questions:

1. **When is each word painted, and does a late start predict a missing object?** Word timing on all 2,000 test captions under three seeds, pre-registered.
2. **Can a missing object be seen coming, early in the painting?** Nine detectors built from the prompt, the attention, a one-step preview and a quick 5-step draft; developed on one half of the test captions, sealed, then tested once on the other half with new seeds. Pre-registered.
3. **Are the automatic judges right?** 240 judge calls labelled by the four of us ([`human-check/`](human-check)).

| Milestone 2 requirement | Here |
|---|---|
| Text encoder integrated with the diffusion model | Stable Diffusion 1.5 with its CLIP ViT-L/14 encoder, through our own sampling loop (`src/w2w/rewind.py`) that records cross-attention at every step |
| Baseline conditional generation, first set of images | the anchor on `eval_2000` under seed ids 0 to 5: 8,553 images, all logged in `Results/m2_runs/` |
| Guidance and noise-schedule sensitivity | an exploratory grid of guidance (3, 7.5, 12) and timestep spacing (leading, linspace, trailing) on 200 development captions (`Results/m2_grid/`) |

---

## Results

### 1. When words are painted (pre-registered: `docs/m2_preregistration.md`)

A word **commits** at the first step after which its cross-attention map keeps a Pearson correlation of at least 0.8 with its final layout. Primary population: `eval_2000` rows 432 to 1999 (rows 0 to 431 were used in a Milestone 1 pilot).

| Test | Result |
|---|---|
| Commit step separates missing from found objects, seed ids 0 / 1 / 2 | AUROC 0.678 [0.631, 0.724] / 0.667 / 0.653: **PREDICTS** under every seed |
| Leaving out *person* (present people commit late but are rarely missed) | 0.82 / 0.81 / 0.80 |
| Same object, found under one seed and lost under another (173 objects) | the losing painting commits 3.3 steps later [2.5, 4.1]; AUROC 0.704 [0.647, 0.761]: **SEEN_COMING** |
| Objects lost under 0 / 1 / 2 / 3 of the seeds (1,785 objects) | 1,589 / 115 / 55 / 26 |

Most losses come from the starting noise, not the caption. The commit step is measured against the painting's own final layout, so it describes a loss after the fact; it is not a forecast. Question 2 asks for the forecast.

### 2. Seeing a missing object coming (pre-registered: `docs/m2_detector_preregistration.md`)

**Design.** Development: `eval_2000` rows 0 to 999 under seed ids 0 to 2 (2,523 paintings, each repainted pixel for pixel identical to its stored picture). Test: rows 1000 to 1999 under new seed ids 3 to 5 (2,553 new paintings, 3,433 objects, 190 missing). After 11 steps from the first to the twenty-first we record each word's attention, the **preview** (DDIM's one-step guess of the finished picture) and the **draft** (the painting finished roughly with five DPM-Solver++ steps, made after steps 0, 2, 4 and 8, at 5 extra network calls each). Nine families of logistic-regression detectors were fitted on development data, sealed by file hash (`Results/m2_detector/lock.json`), and run once on the test paintings. The test code refuses to run without the seal, and its run log has exactly one line (`Results/m2_detector/prospective/runs.log`).

| Reading | Development | Test |
|---|---|---|
| H1: the painting adds to the prompt (after 3 steps) | +0.107: ADDS | +0.059 [0.027, 0.090]: **ADDS** |
| H1: same object across seeds | 0.583: SEEN_COMING | 0.693 [0.623, 0.762]: **SEEN_COMING** |
| H2: attention knows before the picture | ATTENTION_FIRST | **NOT_SHOWN** (tied at the first step, preview ahead from the second) |
| H3: the detector without drafts reaches AUROC 0.90 | after 7 steps | after 7 steps |
| H4: the draft looks ahead of the preview (after 1 step) | +0.225: DRAFT_AHEAD | +0.144 [0.112, 0.180]: **DRAFT_AHEAD** |
| H4: attention and preview still add to the draft | not shown | not shown |

| Test, AUROC (95% interval) | After 1 step | After 3 steps | After 9 steps |
|---|---|---|---|
| prompt alone | 0.788 | 0.788 | 0.788 |
| prompt + attention + preview | 0.814 | 0.847 | 0.961 |
| prompt + draft | **0.947** [0.927, 0.963] | 0.962 | 0.991 |
| ceiling: RetinaNet on the finished picture | 0.986 | | |

With 10% of present objects flagged, the draft made after the first step catches 86% of missing objects; the prompt alone catches 35%. Three of the four pre-registered predictions held on the test; "attention knows first" did not.

### 3. Human check of the judges

See [`human-check/README.md`](human-check/README.md). People agree with each other 95% of the time (kappa 0.90); the rule's "missing" is right 97% of the time but misses some real losses, so missing rates are lower bounds. Against people, the draft after one step still scores AUROC 0.86 [0.79, 0.93], so its lead is not the judges agreeing with themselves.

### Exploratory grid (not pre-registered)

On the first 200 development captions, guidance shifts when words commit (+0.9 steps [0.5, 1.3] at guidance 3, −0.8 [−1.1, −0.5] at 12); the timestep spacing does not move the commit step. No setting changed the missing rate detectably.

---

## Folder contents

```
milestone-2/exp1/
├── README.md
├── requirements.txt, pyproject.toml
├── src/w2w/              ← the package: data, encoding, Painter (sampling loop with attention recording, rewind, keep_x0), steering, the live viewer, the protocol
├── scripts/
│   ├── m2_check.py       ← the protocol and Painter paint the same pictures (run first, on the GPU)
│   ├── m2_timing.py      ← word timing: generate, judge, analyze, seeds
│   ├── m2_grid.py        ← the exploratory development grid
│   ├── m2_detector.py    ← early warning: record, judge, preview, features, develop, lock, test, figures
│   ├── m2_merge_logs.py  ← builds ../logs/ from every member's log
│   └── pilot_*.py        ← the Milestone 1 pilots, kept because the tests and later scripts import them
├── tests/                ← 72 tests (pytest)
├── docs/
│   ├── m2_preregistration.md            ← word timing, written before the run
│   ├── m2_detector_preregistration.md   ← early warning, written before any recording
│   └── research_log.md                  ← dated entries, development and test values side by side
├── Results/
│   ├── m2_runs/          ← runs.csv, images.csv: every image this folder generated
│   ├── m2_timing/        ← one folder per split, setting and seed: summary.json, commit_vs_recall.png
│   ├── m2_grid/          ← summary.json, comparison.csv
│   └── m2_detector/      ← lock.json, models.joblib, features, development/ and prospective/ results, figures
└── human-check/          ← the 240-picture human check
```

Not in the repo (too large; on Drive): the generated images, the recorded attention maps, previews and drafts.

## How to run

On a GPU with the packages in `requirements.txt` (we used an RTX 5070 Laptop, PyTorch 2.11, diffusers 0.42.1), from this folder:

```
pip install -e .
pytest                                     # 72 tests
python scripts/m2_check.py                 # protocol check: must report a pixel difference of 0
python scripts/m2_timing.py generate --split eval_2000 --seed-id 0
python scripts/m2_timing.py judge --split eval_2000 --seed-id 0
python scripts/m2_timing.py analyze --split eval_2000 --seed-id 0
python scripts/m2_timing.py seeds          # after seed ids 1 and 2
python scripts/m2_detector.py record development     # then the same for: prospective
python scripts/m2_detector.py judge development      # then: prospective
python scripts/m2_detector.py preview development    # then: prospective
python scripts/m2_detector.py features development
python scripts/m2_detector.py develop                # development results only
python scripts/m2_detector.py lock                   # seals the detectors; run once
python scripts/m2_detector.py test                   # the one-time test; refuses to run without the seal
python scripts/m2_detector.py figures
```

Word timing takes about 2 hours per seed on 2,000 captions; recording, previews and drafts for the early-warning study about 9 hours; the test itself about 6 minutes.
