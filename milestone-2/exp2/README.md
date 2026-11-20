# milestone-2 / exp2

**Author:** Mukul Sridar Haladhi · IE 7615 Deep Learning for AI · Northeastern University

Part of **From Words to Worlds**. Stable Diffusion 1.5 is usually run with 30 DDIM steps and reads its prompt through CLIP. This experiment tests both defaults. First, can a faster sampler with fewer steps keep image quality, and does the timestep schedule matter? Second, can a different language model, BERT, take CLIP's place through a trained connector? All images use the shared protocol in [`../protocol`](../protocol) and the captions in [`../splits`](../splits).

Four questions:

1. **Can 10 steps match 30?** DPM++ 2M at 10 steps against the 30-step DDIM anchor, on all 500 `dev_500` captions.
2. **Does the timestep schedule matter?** Four schedules at 10 steps: leading, trailing, Karras and Align Your Steps (AYS).
3. **Has the sampler converged by 10 steps?** The same four schedules at 30 steps, on the first 200 captions.
4. **Can BERT stand in for CLIP?** Stage 1 of a connector that turns BERT's token states into the states SD's UNet expects.

| Milestone 2 requirement | Here |
|---|---|
| Integrate a text encoder with a diffusion model | A trained connector feeds frozen `bert-base-uncased` into SD 1.5's UNet in place of CLIP ViT-L/14 (stage 1) |
| Run baseline conditional generation | The anchor: DDIM, leading spacing, 30 steps, guidance 7.5, on all 500 `dev_500` captions (seed 0), plus seeds 1–2 on the first 200 |
| Tune the noise schedule | DPM++ 2M with four timestep schedules × 10 and 30 steps; guidance held at 7.5 so the schedule is the only change |
| Record the first set of generated images | 4,400 images, each logged in `m2/images.csv` |
| Generation log, early samples | `m2/runs.csv`, `m2/images.csv`, `pairs_anchor_vs_leading10.png` |

## Results

### 1. Ten steps against thirty (500 captions, seed 0)

| Setting | Object recall (95% interval) | CLIPScore | KID ×1000 | FID | s / image |
|---|---|---|---|---|---|
| **Anchor:** DDIM leading, 30 steps | 93.9% (92.0–95.8) | 26.78 | 4.41 | 85.1 | 5.44 |
| DPM++ 2M leading, 10 steps | 93.3% (91.0–95.3) | 26.87 | 4.48 | 87.3 | 2.00 |
| DPM++ 2M Karras, 10 steps | 93.1% (90.9–95.2) | 26.69 | 4.73 | 84.5 | 2.00 |
| DPM++ 2M AYS, 10 steps | 93.2% (91.1–95.2) | 26.62 | 5.78 | 86.1 | 2.00 |
| DPM++ 2M trailing, 10 steps | 93.0% (90.7–95.0) | 26.75 | 6.40 | 87.4 | 2.00 |

**DPM++ 2M with leading spacing at 10 steps shows no detectable loss against the anchor, at 37% of the time per image.**

### 2. Which schedule (paired on caption and noise)

| 10 steps minus anchor | Recall (points) | KID ×1000 | CLIPScore |
|---|---|---|---|
| leading | −0.5 (−1.8 to 0.7) | +0.03 (−0.79 to 0.73) | +0.12 (−0.10 to 0.34) |
| Karras | −0.9 (−2.6 to 0.7) | +0.30 (−0.48 to 1.16) | −0.07 (−0.28 to 0.14) |
| AYS | −0.7 (−2.3 to 0.9) | **+1.37 (0.57 to 2.11)** | −0.15 (−0.36 to 0.05) |
| trailing | −0.9 (−2.7 to 0.7) | **+1.98 (1.22 to 2.74)** | −0.01 (−0.20 to 0.18) |

- **Recall:** no schedule loses detectable recall; every interval crosses 0.
- **KID:** leading and Karras match the anchor, while trailing and AYS are measurably worse. This goes against published results and rests on 500 images and one seed, so it is reported for this setup only.
- **CLIPScore against leading-10:** AYS is also lower, at −0.27 (−0.50 to −0.05).
- **Appearance:** leading's images are visibly hazier at 10 steps, because its first step does not start from pure noise, but they keep the same objects.

### 3. Ten steps or thirty, same sampler (first 200 captions)

| 30 steps minus 10 steps | Recall (points) | KID ×1000 | CLIPScore |
|---|---|---|---|
| leading | −0.5 (−3.3 to 2.4) | +0.56 (−0.62 to 1.92) | −0.26 (−0.61 to 0.08) |
| Karras | −1.8 (−4.6 to 0.9) | +0.66 (−0.62 to 2.01) | −0.08 (−0.37 to 0.22) |
| AYS | +0.5 (−2.6 to 3.4) | −0.22 (−1.44 to 1.29) | +0.08 (−0.24 to 0.40) |
| trailing | +0.5 (−1.9 to 2.7) | −0.95 (−2.22 to 0.56) | −0.08 (−0.37 to 0.18) |

No metric changes significantly, so DPM++ 2M has converged by 10 steps.

**Across seeds** (first 200 captions, object recall):

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Anchor, DDIM 30 steps | 93.7% | 93.6% | 94.1% |
| DPM++ 2M leading, 10 steps | 93.1% | 95.0% | 96.4% |

Leading-10 does not fall behind the anchor on any seed.

### 4. BERT in place of CLIP (connector, stage 1)

**Design.** 77 learned query slots cross-attend to the token states of a frozen `bert-base-uncased`. The output is 77 × 768 states, the shape SD 1.5's UNet reads from CLIP. The connector has 4 decoder layers and 39M parameters, and it is the only part trained; BERT, CLIP and the UNet stay frozen. It learns from captions alone to reproduce CLIP's states, using mean squared error on targets standardised per position and dimension. On that scale, a loss of 1.0 equals always predicting the average CLIP state.

**Data.** We trained on 100,000 COCO train2017 captions, excluding every image in any of our splits, and held out 2,000 captions from unseen images. Training ran 8,000 steps at batch 128 (AdamW, learning rate 3 × 10⁻⁴, warm-up and cosine decay) in 59 minutes on a T4.

| Step | 0 | 1,000 | 3,000 | 5,000 | 8,000 |
|---|---|---|---|---|---|
| Held-out loss | 1.29 | 0.40 | 0.29 | 0.25 | **0.24** |
| Per-token cosine to CLIP | 0.40 | 0.84 | 0.88 | 0.90 | **0.90** |

Training loss ended at 0.22, and held-out loss never rose, so the connector generalises to new captions rather than memorising them.

For four dev captions with the same starting noise, SD draws the same subject, layout and colours from the connector's states as from CLIP's, with no image training (`connector/clip_vs_bert_samples.png`). Secondary details drop out, such as a magnifying glass and a train's windows.

**Conclusion.** The default 30 DDIM steps are more than SD 1.5 needs: DPM++ 2M with leading spacing reaches the same recall, KID and CLIPScore in 10 steps, at about a third of the cost. The schedule matters for KID but not for whether objects appear. BERT can drive SD's UNet after caption-only training, with close but imperfect agreement with CLIP. Stage 2 of the connector is the next step to close that gap.

## Settings

| Setting | Value |
|---|---|
| Model | `stable-diffusion-v1-5/stable-diffusion-v1-5`, float16, guidance 7.5 |
| Anchor | DDIM, leading spacing, 30 steps (identical to the shared protocol anchor) |
| Sweep | DPM++ 2M (DPMSolverMultistep) × leading / trailing / Karras / AYS × 10 steps (500 captions) and 30 steps (first 200) |
| Seeds | Protocol rule: seed = 100000 × seed_id + caption key, float32 noise drawn on the CPU. Seed 0 for every setting; seed 1 on the first 100 captions for the four 10-step schedules (for diversity); seeds 1–2 on the first 200 captions for the anchor and leading-10 |
| Captions | `dev_500`; reference photos for KID and FID from `ref_2000_v2` |
| Hardware | Kaggle, Tesla T4 |
| Software | Protocol m2-v1, PyTorch 2.11, diffusers 0.42.1 |

## How it is scored

| Metric | What it measures |
|---|---|
| Object recall | Share of the objects named in the caption that the shared object judges detect in the image (higher is better) |
| CLIPScore | How well the image matches its caption, by CLIP ViT-L/14 (higher is better) |
| KID, FID | How close the set of images is to real COCO photos (`ref_2000_v2`, lower is better) |
| LPIPS, CLIP distance | Diversity: how different the seed-0 and seed-1 images of the same caption are (10-step settings, first 100 captions) |

- **Shared scoring code:** all four metrics come from the group's scoring code, so the numbers are comparable with the other exp folders.
- **Intervals:** 95% intervals are bootstrapped over captions, and differences are paired on the same caption and starting noise.
- **KID is the headline image metric** because FID depends on sample size. The same settings give FID about 85 on 500 images and about 137 on 200.
- **Safety checker:** it blacked out 23 of 4,400 images, which are left out of scoring.

## Folder contents

```
milestone-2/exp2/
├── README.md
├── Noise schedule and BERT connector.pdf   ← two-page write-up
├── m2_sweep_kaggle.ipynb                   ← generation, scoring and connector training, with outputs
├── mukul_per_seed_200.csv                  ← recall and CLIPScore per seed, anchor and leading-10
├── pairs_anchor_vs_leading10.png           ← same caption and noise: anchor vs leading-10
├── m2/
│   ├── runs.csv, images.csv                ← protocol logs: every run and every image
│   ├── image_counts.csv                    ← images, flagged and seconds per image, per setting and seed
│   └── scores/
│       ├── objects/                        ← per-image detections and recall, one file per setting, plus the real photos
│       ├── images/                         ← per-image CLIPScore
│       ├── diversity/                      ← LPIPS and CLIP distance between seeds 0 and 1, 10-step settings
│       ├── mukul_schedules_500.csv         ← summary, all settings on 500 captions
│       ├── mukul_s10_vs_s30_200.csv        ← summary, 10 vs 30 steps on 200 captions
│       └── mukul_paired.csv                ← paired differences with intervals
└── connector/
    ├── config.json, target_stats.pt        ← training settings and target standardisation
    ├── training_log.csv, training_curve.png
    └── clip_vs_bert_samples.png            ← same caption and noise: CLIP states vs connector states
```

## How to run

On Kaggle with a T4 GPU, attach [`../protocol`](../protocol) and [`../splits`](../splits) as input data, then run `m2_sweep_kaggle.ipynb` from cell 4. Cells 0–3 are an earlier attempt and are superseded.

| Part | Cells | Images | Time on a T4 |
|---|---|---|---|
| Install packages, load the protocol and splits | 4 | — | not timed |
| Anchor, 8 DPM++ settings, seed 1 for diversity | 5 | 3,700 | ~3 h 20 min |
| Scoring and summary tables | 6–10 | — | not timed |
| Seeds 1–2 for the anchor and leading-10 | 11 | 700 | ~45 min |
| Connector training and samples | 12–16 | — | ~1 h |
| Zip results (without images) | 17 | — | — |

## Limitations

- **One seed for the schedule comparison.** The KID gaps for trailing and AYS rest on seed 0 only. Seeds 1–2 cover just the anchor and leading-10.
- **Only large effects detectable.** The 30-step check uses 200 captions, so step-count effects smaller than about 3 recall points would go undetected.
- **Guidance held at 7.5.** Schedule and guidance may interact, and that was not tested here.
- **Connector image check is illustrative.** It uses four captions, so it is a sanity check, not a measurement. Stage 1 is trained to match CLIP's states, not image quality directly.
- **Diversity on 100 captions.** The LPIPS and CLIP-distance diversity scores use one pair of seeds on the first 100 captions, so they are not reported as differences between schedules.
