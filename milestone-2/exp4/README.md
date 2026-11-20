# milestone-2 / exp4: Guidance and step-count sweeps, and the group's shared scoring code

**Author:** Tanmayi Shurpali · IE 7615 Deep Learning for AI · Northeastern University

Part of [From Words to Worlds](../../README.md). All images use the shared
protocol in [`milestone-2/protocol`](../protocol) (m2-v1) and the captions in
[`milestone-2/splits`](../splits).

**Result in one line:** on 500 unseen test captions with three seeds each, 10 DDIM steps keep object recall level with the 30-step anchor (+0.0 points, 95% CI −1.0 to +1.0), are detectably closer to real photos on KID (−0.85, CI −1.44 to −0.32), and take 1.7 s per image instead of 4.7 s. The setting was chosen on dev captions only, by rules written before any result was seen.

**My part:** I ran the guidance and step-count sweeps, wrote `m2_scoring.py`, which the whole group uses to score its images, and ran the follow-ups the proposal promised: the real-photo floor, the Inception Score, three seeds on the headline comparison, the early-samples figure and the HEaD+ related-work check.

## What this tests

How classifier-free guidance (3, 5, 7.5, 10, 12) and the number of DDIM steps (10, 20, 30, 50) change four things in Stable Diffusion 1.5: whether the objects named in the caption appear (object recall), how close the images are to real photos (KID, FID, Inception Score), how well they match the caption (CLIPScore), and how much two seeds of one caption differ (diversity). The question behind it is whether the missing-object failures the group studies come from the sampler settings, or have to be fixed in the model or the text encoder.

## What I did

**Generation (`code/1_m2_generate.ipynb`)**
- Used the group's shared protocol unchanged: SD 1.5 in half precision, DDIM with eta 0, leading timestep spacing, safety checker on, and seed = 100000 × seed_id + caption_idx with the starting noise drawn on the CPU. Every setting therefore paints each caption from identical noise, so settings are compared caption by caption.
- Ran the smoke check before any run: noise sum 36.4438 and anchor timesteps 958 … 1 both matched the group's.
- Changed one setting at a time from the anchor (guidance 7.5, 30 steps):
  - guidance sweep: all 500 `dev_500` captions, 2,500 images;
  - diversity pairs: a second seed on the first 100 captions at every guidance value, 500 images;
  - step sweep: the first 200 captions at 10, 20 and 50 steps, 600 images; the 30-step point reuses the guidance-7.5 images, which share their captions and noise.
- Painted `eval_500` only after the choice was made: the anchor and the chosen setting, seeds 0, 1 and 2, 3,000 images.
- In all: 6,600 images, 27 blacked out by the safety checker (20 dev, 7 test) and left out of every comparison, and about 8 GPU-hours on a free Colab T4. Every run resumes from the last image saved on Drive.

**Shared scoring code (`code/m2_scoring.py`, used by the whole group)**
- **Object recall:** each labelled object (the `target` rows of the groundtruth files, 559 objects in `dev_500`) is judged by two detectors, Faster R-CNN v2 (class score ≥ 0.5) and OWL-ViT with "a photo of a {class}" (score ≥ 0.1). An object counts as present if either finds it. Both judges' raw scores are kept, so thresholds can be checked later without rescoring.
- **Realism:** KID (× 1000), FID and the Inception Score from the standard Inception-v3 network, against `ref_2000_v2`, 2,000 COCO photos that share no photo with any caption set.
- **Alignment and diversity:** CLIPScore with ViT-L/14; LPIPS and CLIP image distance between the two seeds of a caption.
- **Statistics:** 95% bootstrap intervals over captions (2,000 resamples, seed 0), with a caption's seeds resampled together. Settings are compared paired, on the same captions and noise. KID uses the unbiased estimator, keeping repeated images out of the resampled pairs.
- Reads any member's `images.csv`, skips blacked-out images, and scores incrementally, so adding seeds never redoes old work.

**Quality checks**
- **Judges' ceiling:** the judges find 98.9% [98.0, 99.6] of the labelled objects in the real dev photos, the best recall any generator could get. They agree on 93% of objects (kappa 0.20 on real photos, where almost everything is present; 0.61 on generated images).
- **Real-photo floor:** the real photos behind the same captions, scored at the same number of images, sit next to every FID, KID and Inception Score. FID and the Inception Score depend on sample size; real photos get FID 78.0 with 500 images and 132.8 with 200.
- **Inception Score intervals:** a single set's interval did not contain its own score (real photos: 31.12, interval [31.60, 36.33]), because a bootstrap resample repeats images, which the score reads as lost variety. Single-set Inception Scores are therefore reported without an interval, like FID; paired differences keep theirs.

**Selection, using dev captions only (`code/3_m2_analysis.ipynb`)**
- Rules written before any result was seen.
- Guidance: keep 7.5 unless another value raises recall without a detectable KID cost.
- Steps: the fewest of 10, 20 and 30 that lose less than 5 points of recall without a detectable KID cost; 50 only if it raises recall.
- Result: guidance 7.5, 10 steps. It was then tested once on `eval_500`, which no choice had used.

**Follow-ups**
- **Figure 4:** eight anchor test images with captions and the judges' verdicts, chosen by a fixed rule rather than by hand: the first five captions where every object is found, and the first three where exactly one is missing.
- **Related work:** checked HEaD+ (Betti et al., 2026) against its PDF. It predicts missing objects early, but its ablation (Table 3) has no text-only baseline, and its split keeps all seeds of a prompt together, so it never tests whether the predictor separates seeds of one prompt.
- **VQAScore:** not run. The model fits a T4, but its reference implementation (`t2v_metrics` 3.0) pins torch 2.5, transformers 4.49 and numpy 1.26, which are incompatible with the Colab runtime (Python 3.13, torch 2.14, transformers 5.22, numpy 2.4).

## How to run

Free Colab Tesla T4, diffusers 0.42.1. Put the group's `M2.zip` (protocol and splits) and `code/m2_scoring.py` in `MyDrive/m2/`, then run:

| Notebook | Runtime | What it does | Time |
|---|---|---|---|
| `code/1_m2_generate.ipynb` | T4 | Smoke check, guidance sweep, diversity pairs, step sweep; section 7 paints `eval_500` | about 8 GPU-hours in all |
| `code/2_m2_score.ipynb` | T4 | All measures, the real-photo floor and the judges' ceiling | about 40 minutes |
| `code/3_m2_analysis.ipynb` | CPU | Paired comparisons, selection rules, figures, tables | about 5 minutes |

Order: notebook 1 (sections 1 to 6), notebook 2, notebook 3, which writes `results/chosen.json`; then notebook 1 section 7, notebook 2 section 6, and notebook 3 again. If Colab disconnects, Run all again: finished images and scores are skipped. To score your own images with the shared code, see the last cell of `code/2_m2_score.ipynb`.

## Results

**Guidance** (500 dev captions, 30 steps, one seed):

| Guidance | Object recall | CLIPScore | KID × 1000 | FID | IS |
|---|---|---|---|---|---|
| 3 | 91.8% [89.2, 94.1] | 26.00 | 1.68 [1.09, 2.47] | 81.8 | 32.66 |
| 5 | 93.9% [91.9, 95.9] | 26.73 | 3.54 [2.50, 4.80] | 83.3 | 33.48 |
| 7.5 (anchor) | 93.7% [91.7, 95.7] | 26.76 | 4.35 [3.38, 5.50] | 85.0 | 31.85 |
| 10 | 92.8% [90.7, 94.8] | 26.87 | 5.64 [4.58, 6.88] | 86.2 | 32.09 |
| 12 | 93.7% [91.6, 95.6] | 26.85 | 6.03 [5.01, 7.23] | 86.9 | 30.64 |
| Real photos (floor) | 98.9% (judges' ceiling) | – | 0.05 [−0.26, 0.47] | 78.0 | 31.12 |

Lower guidance looks more like real photos but loses objects: against 7.5, guidance 3 lowers KID by 2.71 [1.96, 3.55] but loses 2.0 points of recall [−4.0, −0.2] and 0.75 of CLIPScore [−0.97, −0.52]. Above 7.5, KID rises (+1.63 [0.95, 2.31] at 12) with no detectable change in recall or CLIPScore. Higher guidance makes a caption's two seeds more alike in content (CLIP distance 0.223 → 0.151) but slightly more different in detail (LPIPS 0.747 → 0.773). See `figures/fig1_guidance_tradeoff.png` and `figures/fig3_guidance_examples.png`.

**Steps** (first 200 dev captions, guidance 7.5, paired against 30 steps):

| Steps | Recall change | KID change | Seconds per image (T4) |
|---|---|---|---|
| 10 | +0.9 points [−1.4, +3.6] | −0.42 [−1.96, +0.80] | 1.9 |
| 20 | +0.5 [−0.9, +1.9] | +0.42 [−0.43, +1.31] | 3.5 |
| 30 | – | – | 5.5 |
| 50 | +0.5 [−0.9, +1.9] | +0.61 [−0.36, +1.61] | 8.4 |

No step count changes recall or KID detectably; the 10-step interval rules out a recall loss of more than 1.4 points. See `figures/fig2_steps.png`.

**Test captions** (`eval_500`, three seeds per caption):

| | Object recall | KID × 1000 | FID (per seed) | IS | CLIPScore | Seconds per image |
|---|---|---|---|---|---|---|
| Anchor, 30 steps | 94.4% [93.0, 95.6] | 4.66 [4.01, 5.92] | 84.9 | 35.73 | 26.70 | 4.7 |
| Chosen, 10 steps | 94.3% [93.0, 95.6] | 3.82 [3.48, 4.73] | 85.1 | 32.56 | 26.70 | 1.7 |
| Paired difference | +0.0 points [−1.0, +1.0] | −0.85 [−1.44, −0.32] | – | −3.30 [−5.54, −1.90] | +0.02 [−0.10, +0.14] | – |
| Real photos (floor) | – | −0.16 | 75.1 | 33.58 | – | – |

The 10-step setting's lower Inception Score is not a loss: the anchor scores above the real photos and 10 steps sits closer to them. Per seed, the recall change was −0.9, +0.3 and +0.5 points. `figures/fig4_early_samples.png` shows eight anchor test images with the judges' verdicts.

**What it means for the project:** no sampler setting fixes missing objects; recall stays at 92 to 94% across every guidance value and step count, against a 98.9% ceiling. The fix has to come from the model or the text encoder, which the LoRA and connector arms test.

## Files

```
exp4/
├── README.md
├── code/
│   ├── m2_scoring.py                     shared scoring code (m2-score-v2.1), used by the whole group
│   ├── 1_m2_generate.ipynb               generation, outputs kept
│   ├── 2_m2_score.ipynb                  scoring, outputs kept
│   └── 3_m2_analysis.ipynb               analysis and figures, outputs kept
├── results/
│   ├── runs.csv                          one row per run setting
│   ├── images.csv                        one row per image (6,600; 27 flagged)
│   ├── guidance_dev_500.csv              guidance sweep with paired differences
│   ├── steps_dev_500_first200.csv        step sweep with paired differences
│   ├── summary_dev_500_all_captions.csv  one row per dev setting
│   ├── final_eval_500.csv                test captions, three seeds pooled
│   ├── final_eval_500_per_seed.csv       test captions, each seed
│   ├── summary_eval_500.csv              one row per test setting (seed 0)
│   ├── real_photo_floor.csv              FID, KID, IS of real photos at 500 and 200 images
│   ├── chosen.json                       the setting chosen on dev_500
│   ├── fig4_early_samples.csv            captions shown in Figure 4
│   └── scores/
│       ├── objects/                      both judges' verdicts and scores per labelled object
│       ├── images/                       CLIPScore and timing per image
│       └── diversity/                    LPIPS and CLIP distance per seed pair
└── figures/
    ├── fig1_guidance_tradeoff.png        six measures against guidance
    ├── fig2_steps.png                    paired changes against 30 steps, and time per image
    ├── fig3_guidance_examples.png        captions whose object is judged differently at guidance 3 and 12
    └── fig4_early_samples.png            eight anchor test images with captions and verdicts
```

## Not in the repo

On Google Drive ([link](https://drive.google.com/drive/folders/1MUwxPldZdxKljHDbp_7I3vxb9YVf6wes?usp=sharing), anyone with the link can view):

- `images/`: the 6,600 generated images, named `<split>/<setting>/<caption_idx>_s<seed_id>.png` as listed in `results/images.csv`;
- `scores/features/` and `scores/logits/`: Inception and CLIP features and Inception logits (`.npz`), needed to recompute KID, FID and the Inception Score without rescoring.

The COCO photos are not redistributed; the scoring notebook downloads them from the COCO server by image id.
