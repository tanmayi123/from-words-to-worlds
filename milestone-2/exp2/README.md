# Experiment 2: Noise Schedules & BERT Connector

For Milestone 2, I ran two main experiments: first, testing if a faster sampler with a tweaked timestep schedule can match our baseline image quality, and second, training Stage 1 of a BERT-to-SD connector.

## What I Ran

To keep everything apples-to-apples, I generated all images under our shared protocol (m2-v1) on a single Kaggle T4 (PyTorch 2.11, Diffusers 0.42.1). I used SD 1.5 at float16 with a guidance scale of 7.5, locking in the exact same captions and starting noise across every run:

- **The Baseline (Anchor):** DDIM with leading spacing at 30 steps across all 500 `dev_500` captions.
- **10-Step Sampler Sweep:** DPM++ 2M at 10 steps using four timestep schedules (leading, trailing, Karras, and Align Your Steps) across all 500 captions.
- **Convergence Check:** DPM++ 2M at 30 steps across those same four schedules on the first 200 captions to verify whether 10 steps is actually enough.
- **Seed Robustness:** Extra runs on seeds 1 and 2 for both the anchor and leading-10 on the first 200 captions.
- **BERT Connector (Stage 1):** I trained a 39M-parameter connector to map frozen `bert-base-uncased` token states into the 77 × 768 shape that SD 1.5's UNet expects from CLIP. I trained it for 8,000 steps on 100k COCO train2017 captions (making sure there was zero overlap with our evaluation splits).

All metrics are paired on identical captions and noise seeds with 95% bootstrap confidence intervals, and KID scores are scaled by ×1000. (Note: The safety checker flagged and blacked out 23 out of ~4,400 images, which I excluded from scoring.)

## Key Takeaways & Results

My main finding is that DPM++ 2M with leading spacing at 10 steps matches the 30-step DDIM anchor with no statistically significant drop in quality, while cutting generation time down to 37% (2.0s vs. 5.44s per image).

Against the anchor's baseline of 93.9% recall [92.0, 95.8] and 4.41 KID, here is how the 10-step schedules performed across all 500 captions:

| DPM++ 2M (10 steps vs. Anchor) | Recall Diff (pts) | KID Diff (×1000) | CLIPScore Diff |
|---|---|---|---|
| Leading | −0.5 [−1.8, 0.7] | +0.03 [−0.79, 0.73] | +0.12 [−0.10, 0.34] |
| Karras | −0.9 [−2.6, 0.7] | +0.30 [−0.48, 1.16] | −0.07 [−0.28, 0.14] |
| AYS | −0.7 [−2.3, 0.9] | +1.37 [0.57, 2.11] | −0.15 [−0.36, 0.05] |
| Trailing | −0.9 [−2.7, 0.7] | +1.98 [1.22, 2.74] | −0.01 [−0.20, 0.18] |

- **Schedule Behavior:** None of the schedules lost meaningful object recall (all intervals cross zero). Leading and Karras both matched the anchor on KID, whereas trailing and AYS degraded noticeably. Since that contradicts some published findings, I'm treating it as specific to this setup rather than a general rule.
- **Step Count:** Pushing DPM++ 2M from 10 to 30 steps on the 200-caption subset didn't move the needle on any metric, confirming that the sampler effectively converges by step 10.
- **Seed Consistency:** On the first 200 captions, object recall across seeds 0, 1, and 2 came out to 93.7 / 93.6 / 94.1 for the anchor, compared to 93.1 / 95.0 / 96.4 for leading-10. Leading-10 held its own across every seed.
- **Connector Progress:** Held-out loss dropped from 1.29 down to 0.24 (tracking right alongside training loss at 0.22 with zero overfitting), and per-token cosine similarity to CLIP's states climbed from 0.40 to 0.90. Visually checking four test captions, SD captures the right subject, layout, and color palette from the BERT connector, though it still drops a few secondary details. Since that's only on four prompts, consider it an early sanity check rather than a formal benchmark.
