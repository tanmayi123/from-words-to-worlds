# milestone-1 / exp4: Combined evaluation: image quality, calibrated object judges and word timing under LoRA

**Author:** Tanmayi Shurpali · IE 7615 Deep Learning for AI · Northeastern University

Part of From Words to Worlds. This experiment puts the team's Milestone 1 work into one fair setup: Santhosh's pipeline and commit measure (exp1), Mukul's LoRA checkpoints (exp2), and Praniti's splits, labels and stress prompts (exp3). It adds the pieces the project was missing: image-quality scores against a real-photo floor, two object judges calibrated on real photos, LoRA checkpoint selection on the development split, and word timing before and after fine-tuning.

## Folder contents

| Folder | Contents |
|---|---|
| `notebooks/` | `Tanmayi_exp.ipynb`: the combined evaluation, run end to end on a Colab T4 with all outputs saved · `Tanmayi_Generative_Project_Milestone1.ipynb`: my initial Milestone 1 run (training subset and text → embedding pipeline) |
| `results/` | `results.json` (every number), `headline.csv`, object recall per model, `word_timing.csv`, `stress_results.csv`, splits and labels |
| `samples/` | real vs base vs LoRA image grid, word-timing plots, encoder comparison plots |
| `reports/` | write-up (to come) |

## Setup

- **Data:** COCO val2017, with the group's frozen splits. All six files (`eval_2000`, `eval_500`, `dev_500`, `real_ref_b_2000`, object labels, stress prompts) were rebuilt and matched the group's files by fingerprint.
- **Sampling:** Stable Diffusion 1.5, PNDM, 30 steps, guidance 7.5, generated from cached CLIP embeddings.
- **LoRA:** Mukul's rank-8 checkpoints. The checkpoint was chosen on 80 `dev_500` captions, then scored on `eval_500`.
- **Statistics:** paired comparisons (same captions, same seeds), bootstrap 95% intervals that resample captions, and a 3-seed check on 100 captions.

## Pipeline checks

| Check | Result |
|---|---|
| Image from cached embeddings vs raw text | mean pixel difference 0.000 |
| Base model before vs after a LoRA load/unload | 0.000 |
| Attention recorder vs saved image | 0.000 |
| Retrieval R@1, 5,000 photos / 2,000 photos | 35.5% / 46.8%, matching exp1 and exp3 exactly |

## Results

**Choosing the checkpoint on dev (CLIPScore).** Scores: base 26.55, step 150 26.56, step 300 26.47, step 450 26.51, step 600 26.61. Step 600 was selected, but the checkpoints differ by less than 0.15.

**CLIPScore on eval_500** (498 pairs). Base 26.18 → LoRA 26.59, a difference of **+0.41 [+0.23, +0.58]**. LoRA scores higher on 61% of captions. Across 3 seeds the difference is +0.52, +0.58 and +0.09 (mean 0.40, sd 0.27), so the gain is small and depends on the seed.

**Image quality** (N = 498, against 2,000 real reference photos):

| | Real photos (floor) | Base | LoRA |
|---|---|---|---|
| FID | 77.1 | 91.1 | 88.4 |
| KID ×1000 | −0.14 | 5.94 | 5.97 |
| Inception Score | 17.37 | 17.06 | 17.37 |

The paired KID difference between LoRA and base is −0.03 [−0.89, +0.94], so **fine-tuning did not measurably change realism**.

**Object recall** (566 named objects; exp3's labels list the targets):

| | Real photos | Base | LoRA | LoRA − base |
|---|---|---|---|---|
| Either judge sees it | 98.4% | 93.8% | 94.2% | +0.4 [−1.6, +2.3] pts |
| Faster R-CNN v2 alone | 96.5% | 89.8% | 92.2% | **+2.5 [+0.4, +4.7] pts** |
| OWL-ViT alone | 93.5% | 84.8% | 87.3% | +2.5 [−0.2, +5.2] pts |

On real photos the judges miss only 2–7% of objects, so most misses on generated images are real. The judges agree only moderately (κ 0.28–0.52), which is why both are reported. By the stricter single-detector measure LoRA draws slightly more of the named objects; by the lenient two-judge measure it makes no difference.

**Stress prompts** (40 each):

| | Base | LoRA |
|---|---|---|
| Counting: exact number correct | 27.5% | 32.5% |
| Multi-object: all three objects present | 15.0% | 17.5% |

Both models fail most of these, and the differences are within noise at n = 40.

**Word timing** (128 target words, 60 captions):
- Median commit step: base 6.5, LoRA 6.0. Paired shift (LoRA − base): −0.14 [−1.06, +0.80] evaluations, so **fine-tuning does not change when words commit**.
- A late commit predicts that the object goes missing. At r ≥ 0.8 the AUROC is **0.86 [0.76, 0.94]** for base and **0.80 [0.60, 0.94]** for LoRA. This matches the team's dev pilot (0.82), but it rests on only 6 and 5 missing objects, so treat it as a pilot.

**Text encoders without training** (2,000 captions):

| Representation | Paraphrase R@1 [95% CI] |
|---|---|
| CLIP (EOS) | 34.0% [31.8, 36.0] |
| TF-IDF words | 20.0% [18.2, 21.7] |
| T5-base | 15.4% [13.9, 17.0] |
| BERT-base | 13.0% [11.6, 14.6] |
| GPT-2 | 3.4% [2.6, 4.2] |

The original word-order and colour tests failed every encoder because sentence structure confounded them. In the fixed versions (only one word swapped for a synonym), CLIP passes 100% of both. BERT passes 0% on both, and T5 0% (word order) and 2.5% (colour). TF-IDF also scores 0%, because a swapped sentence uses exactly the same words.

## Initial Milestone 1 run: training subset and text → embedding pipeline

`notebooks/Tanmayi_Generative_Project_Milestone1.ipynb` was my individual Milestone 1 run, made before the team combined its work. It builds a cleaned training subset from COCO **train2017**, so it never overlaps the val2017 test photos.

| Step | Result |
|---|---|
| Source | COCO train2017: 118,287 images, 591,753 captions |
| Image rules | short side ≥ 400 px, aspect ratio ≤ 2, corrupt and blank files removed, near-duplicates removed with a 256-bit difference hash |
| Caption cleaning | Unicode and whitespace normalised, repeated words removed, 5–40 words, within CLIP's 77-token limit, duplicates dropped |
| CLIP alignment filter | worst-aligned 1% of captions and worst 5% of images dropped (CLIP ViT-L/14 score) |
| Final subset | **3,000 images, 14,899 captions** (4.97 per image), 512 × 512, split 2,400 / 300 / 300 by image |
| Text → embedding | 77 × 768 CLIP states, matched to Stable Diffusion's own text encoder; **12 of 12 automated checks passed** |
| Retrieval on its 300 test photos | text → image R@1 71.6%, R@5 94.5% · image → text R@1 88.0% |

Those retrieval numbers come from a small pool of photos the same CLIP model had already filtered, so they aren't comparable to the 35.5% / 46.8% above. The subset is available for the Milestone 2–3 fine-tuning runs.

## Takeaways

1. The LoRA gives a **small, seed-dependent CLIPScore gain** (about +0.4) and slightly better object recall by the strict judge, with **no change in image quality** and **no change in word timing**.
2. **When a word commits predicts whether its object is drawn** (AUROC about 0.8–0.86) in both models. This supports Milestone 2's plan to measure commit steps on all 2,000 test captions.
3. CLIP is the only encoder that keeps word order and colour binding, so BERT and GPT-2 need a well-trained connector to compete.

## Limitations

- `eval_500` was used instead of `eval_2000` (free-T4 budget), with one LoRA training run.
- The word-timing AUROCs rest on few missing objects.
- CLIPScore uses the same CLIP that conditions the model; FID/KID and object recall are the independent checks.
- The notebook's CKA bootstrap intervals are biased upward, so only the point estimates (T5 0.60, BERT 0.53, GPT-2 0.13) should be quoted.

## How to run

Open `notebooks/Tanmayi_exp.ipynb` in Colab with a T4 GPU. Put the LoRA checkpoints from exp2 (or Mukul's `project_2.zip`) in `MyDrive/w2w_tanmayi/inputs/`, then choose *Run all*. A full run takes about 3.5 hours, and every image is saved to Drive as it goes, so it resumes after a disconnect.
