# Research log

What we did, what broke, how we found it, and what the numbers were. Newest entries are at the bottom of each milestone. Every number here is traceable to a file under `Results/`.

## Milestone 1 (November 2, 2026)

### Environment and smoke test

- Hardware: NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, Blackwell, compute capability 12.0), AMD Ryzen 9 270, 31 GB RAM, Windows 11.
- Blackwell GPUs need PyTorch built for CUDA 12.8 or newer, so the project uses a dedicated conda environment (`w2w`, Python 3.11) with torch 2.11.0+cu128, diffusers 0.42.1 and transformers 5.22.1. Exact versions: `Results/environment/pip_freeze.txt`.
- The original `runwayml/stable-diffusion-v1-5` repository no longer exists on Hugging Face; we load `stable-diffusion-v1-5/stable-diffusion-v1-5`.
- Smoke test (20 steps, guidance 7.5, fp16, 512 x 512): 2.33 s per image, 3.42 GB peak GPU memory, and the same seed reproduces an identical image. Of three test prompts, "A cat sitting on a wooden bench in a park." (seed 2) produced an empty bench with no cat. That failure became the first word-timing example. Files: `Results/m1_smoke_test/`.

### Pilot: how much FID depends on sample size

Before choosing metrics we measured FID between two disjoint sets of real COCO val2017 photos (500 reference images, resized to 512 x 512), varying the number of photos scored. Real photos score as follows: 339.6 at N = 10 (sd 17.2 over 30 draws), 276.7 at N = 25, 233.5 at N = 50, 185.1 at N = 100, 129.1 at N = 250 and 98.3 at N = 500. FID from a few dozen images therefore says almost nothing about realism. Consequences: every FID we report is shown beside the real-against-real score at the same N; parameter sweeps use KID, which is unbiased at small N; full FID is computed only at a few anchor settings on all eligible images.

### Data cleaning

- MS-COCO 2017 validation: 5,000 images, 25,014 raw captions.
- Captions: whitespace and spacing before punctuation repaired in 3,832 captions; 10 exact duplicates within the same image dropped (case-insensitive); wording, spelling and case never edited. 25,004 captions remain, 4 to 7 per image (median 5). None exceeds CLIP's 77-token window; the median caption is 13 tokens and the longest 58.
- Images: 10 grayscale images flagged for conversion; 42 images have a short side under 256 pixels.
- Files: `Results/m1_pipeline/caption_stats.png`, `Results/m1_data/data_card.json`.

### Text-to-embedding cache

- Encoder: CLIP ViT-L/14 text transformer, the same weights Stable Diffusion 1.5 uses. Each caption becomes a 77 x 768 matrix of token states (what the UNet reads) plus CLIP's pooled sentence vector (used for retrieval and scoring). The empty-prompt states for classifier-free guidance are cached as well.
- Stable Diffusion's pipeline encodes prompts without an attention mask, so padding positions carry context the UNet reads. Our encoder matches that.
- First attempt: captions encoded in batches of 256. Token states agreed with the pipeline's to within 0.03, but an image generated from cached states differed from the pipeline's own image by a mean of 3.48 (0 to 255 scale). Diagnosis: at the same batch size, our encoder and the pipeline's are bit-identical; half-precision matrix products on the GPU give slightly different results at batch 256 than at batch 1, and 30 denoising steps amplify the difference to about 4 percent of a seed change. Evidence: `Results/m1_pipeline/parity_batch256_vs_batch1.png`.
- Fix: encode one caption at a time, as the pipeline does. Result: token states identical (maximum difference 0) and images identical (mean absolute pixel difference 0). Encoding all 25,004 captions takes about 7 to 8 minutes.

### Retrieval sanity check

For every caption, CLIP ranks all 5,000 images by similarity: recall@1 35.5 percent, recall@5 59.9 percent, recall@10 70.3 percent, median rank 3 (chance recall@1 is 0.02 percent). File: `Results/m1_pipeline/summary.json`.

### Sample tests

Five captions drawn with a fixed seed from images showing at least two object categories, three noise seeds each, 30 steps, guidance 7.5, generated from the cached states. No image was flagged by the safety checker. Objects usually appear; actions and relations mostly do not (nobody is "flying through the air", nobody sits, one man never "serves" another, and "bike" becomes a bicycle where the real photo shows a motocross jump). Speed: 3.03 s per image on AC power (mean of 10 images); on battery the same run takes about 8.2 s, so all timings are taken plugged in. Files: `Results/m1_pipeline/samples/`, `samples_grid.png`, `timing.json`.

### Attention recorder

- Goal: record, for every word, how much cross-attention it receives at every denoising step and where in the image it lands. Maps are averaged over heads, upsampled from each layer's resolution to the 64 x 64 latent grid and averaged over the 16 cross-attention layers (as in DAAM), but kept separate per step. Only the conditional branch of classifier-free guidance is recorded.
- Version 1 computed the attention output itself with a float32 softmax. On CPU it matched the stock UNet to 5e-6, but on the GPU in float16 it changed the generated image by a mean of 5.2 pixels.
- Version 2 passes the computation to the stock attention processor and recomputes the probabilities on the side. The recorded image is now identical to the stock one (difference 0), and the smoke-test image is reproduced exactly.
- Stable Diffusion's default PNDM scheduler evaluates the UNet once more than the number of inference steps, so a 30-step run records 31 entries. Later milestones switch to a scheduler with one evaluation per step.

### Commit step

- Definition: a word commits at the first step after which its heatmap keeps a Pearson correlation of at least 0.8 with its final layout. The final layout is the mean map of the five evaluations before the last one.
- Why the last evaluation is excluded: at the lowest noise level nearly every word's map decorrelates at once (from about 0.98 to about 0.75) while the image barely changes. Left in, that single step turned real commits into "never".
- The size of that final-step drop varies by word, from 0.004 to 0.325 (mean 0.093 over 44 content words). It is small for words drawn as a clear object (cake 0.004, bench 0.013, dog 0.017) and large for words with no object of their own in the image (bunch 0.325, cube 0.307, red 0.199). This may be a cheap signal that a word was not rendered; Milestone 2 tests it at scale.
- A repeated word (two "man" in one caption) is tracked per position, as "man (1)" and "man (2)".
- The start token takes 89 to 91 percent of all cross-attention in every prompt, so word shares are computed over the prompt's own words.

### First word timelines

- "A cat sitting on a wooden bench in a park." (seed 2, 20 steps): no cat appears. "bench" commits at step 1; "cat" overtakes it in attention share by step 5 but only commits at step 14, on a small patch at the bench's edge.
- "A dog chasing a cat across a grassy yard." (seed 0, 30 steps): both animals appear; "dog" and "cat" both commit at step 3.
- "A red cube on top of a blue sphere.": no blue appears and the sphere is red; "sphere" commits at step 3, "cube" at 7, "red" at 13, "blue" at 18.
- Late commits are not always fatal: "man" commits at step 20 of 30 for the cyclist and still appears.
- The commit order within each prompt holds at thresholds of 0.7 and 0.9; absolute steps shift by 3 to 10.
- Files: `Results/m1_word_timelines/`.

### Image preprocessing and evaluation splits

- Every image is converted to RGB, center-cropped to a square and resized to 512 x 512 with Lanczos filtering, saved as PNG so that real and generated images go through the same lossless format.
- The crop keeps 71.6 percent of an average frame (minimum 24.8 percent). For very wide or tall photos it can cut away what the caption describes; 12 of the 2,000 evaluation pairs lose more than half the frame and are flagged for any per-image comparison with the real photo.
- The 42 images with a short side under 256 pixels are excluded from the image reference set (4,958 remain), because enlarging them more than twice blurs them.
- Frozen splits, seed 2026: `eval_2000` (2,000 reference-eligible images, one random caption each) and `eval_500` (its first 500 rows). Rerunning reproduces the same split. Files: `Results/m1_data/`.

### Review before submission

Three independent reviews (requirements audit, a skeptical-grader read of the report, and a code and reproducibility review) led to these changes:
- the report now defines the commit step in one place, cites Prompt-to-Prompt and Attend-and-Excite, and states the counterexample above;
- the preprocessing script runs without the notebook;
- the recorder refuses inputs it would mishandle;
- file encodings are explicit;
- requirements are pinned;
- the notebooks are cleaned of local paths and progress-bar output.

## Milestone 2

### Steering pilot (November 3, 2026)

- Design: an additive bias of +log(s) inside a target box and -log(s) outside it is put on one word's cross-attention scores from evaluation k onward, with the box on the opposite side from where the object lands without steering; a Faster R-CNN detector (score at least 0.5) judges whether the object's centre ends up in the box. The strength s is the smallest of 2.5, 5, 10 and 25 that moves the object at k = 0 in at least 4 of the first 6 prompts, with the end-of-text tokens steered too as a fallback; if none does, the pilot stops before the main run.
- Parity: with steering off, the steering attention path reproduces the stock image exactly (mean difference 0.0000).
- Calibration at k = 0, seed 0, objects moved out of 6: strength 2.5: 2; 5: 1; 10: 2; 25: 3; 25 with end tokens: 3. No strength reached 4, so the verdict is `NO_STEER` and the timing question (k from 0 to 21) was not run.
- What the images show:
  - At strengths up to 10, the layout stays close to the unsteered image. The horse and the bicycle do not move at any strength.
  - Several "moved" calls are partial. The cat shifts from left of centre to the centre and straddles the box edge. The dog grows rather than relocates, so its centre falls inside the box at strength 2.5 but outside it at stronger settings.
  - At strength 25 the scene recomposes (a smaller clock, extra fence posts, a doorway placed in the box), and the umbrella appears twice, both copies inside the box.
  - Steering the end tokens as well flattens the images into smooth, painted-looking textures, and the safety checker blacked out 2 of the 6 (dog and horse).
  - One detector miss: with the end tokens steered, a faint bicycle drawn inside the box received no detection.
- Files: `Results/m2_pilot_steering/summary.json`.

### Editing pilot (November 3, 2026)

- Design: three ways to change a picture from step k onward, on DDIM with 30 steps, guidance 7.5, seeds 0 and 1 and the same starting noise for every condition. They are: swapping the object word in the prompt (20 prompts, judged by the Faster R-CNN detector); adding a detail to the prompt (24 edits, judged by a CLIP score normalised between the unedited and the fully edited run, taken at 0.5 or more); and dragging the object into a box with gradient guidance on the latents for 8 steps (12 prompts, judged by the detector). For each arm the pre-registered question was whether the step after which the change stops taking is predicted by the word's commit step in the unchanged run better than by one fixed step for every word.
- Parity: our attention path reproduces the stock image exactly (mean difference 0.0000). Run time 76.8 minutes: swap 25.0, detail 29.9, drag 21.9.
- Swap: `WEAK`, 30 usable units of 40.
  - The new object appears at k = 0 in all usable units. Success falls with k: 0.93 at 3, 0.67 at 9, 0.40 at 18 and 0.07 at 27, with a median window close of 16.5.
  - Commit steps spread enough (interquartile range 5).
  - Timing does not predict the close. The per-word rule's leave-one-prompt-out accuracy is 0.683 against 0.733 for a fixed step, and Spearman rho between commit and close is 0.03 (95% interval -0.38 to 0.47).
- Detail: `NO_STEER` under the rule that an edit must still take when started at k = 9 in half of the usable units.
  - The rate at k = 9 is 0.27 (37 usable units of 48). By k it is 0.73 at 3, 0.46 at 6, 0.19 at 12 and 0 from 24, with a median window close of 6.
  - Edits the model could not draw at all were excluded by the 0.01 CLIP rule: the smiling cat, the roman-numeral clock (the original already had roman numerals), the baby giraffe, and some seeds of other edits.
  - Per-word accuracy 0.838 against 0.860 for a fixed step; rho 0.11 (-0.24 to 0.42).
- Drag: `WEAK`, step size 0.05, which moved 4 of 6 calibration objects.
  - It moves the object into the box at k = 0 in 0.91 of the 23 usable units, and still in 0.48 at k = 21; 11 of 23 windows never close.
  - Per-word accuracy 0.739 against 0.696 for a fixed step, a gain of 4.3 points against the required 5; rho -0.25 (-0.66 to 0.25).
- What the images show:
  - Swap:
    - Starting at k = 0 redraws the whole scene; from about k = 6 the original scene is kept and only the object changes.
    - Mid-run swaps (k = 12 to 18 for cat to dog) give hybrids, such as a dog with cat markings, which the detector counts as the new object.
    - "kite" is drawn as the bird of prey, which the detector correctly rejects, and "orange" is drawn as apples.
  - Detail:
    - Accessories and colours are clearly visible when the edit starts at k = 0 or 3 (red hat, sunglasses, snow, blue bird), fade by k = 6 to 12, and are gone by k = 18.
    - Starting at k = 0 also changes the scene; starting at k = 3 to 6 mostly keeps it.
    - CLIP calls the blue bird edit taken up to k = 21 (score 0.51) where the image is nearly unchanged from the original.
  - Drag:
    - The object does end up in the box, but the images are heavily degraded: flat colour fields, posterised textures, geometric shards and collage-like panels.
    - Objects are often doubled (two cats, two giraffes) or fragmented, for example scattered clock numerals, or a bicycle drawn as line art.
    - The detector accepts these degraded objects.
- Files: `Results/m2_pilot_editing/`.

### Rewind viewer (November 3, 2026)

- Our own DDIM sampling loop keeps the latent state and the predicted finished picture at every step, so a painting can be resumed or branched from any step with a changed prompt. It reproduces the stock pipeline exactly (mean difference 0.0000), and branching with an unchanged prompt reproduces the original painting exactly (unit test).
- Five examples were each branched from steps 0, 3, 6, 9, 12 and 15. By eye, the rewind step that keeps the scene and still shows the change is:
  - 6 for bicycle to motorcycle and for the red hat (from step 9 on the edit is gone);
  - 9 to 12 for sunglasses;
  - 12 to 15 for snow;
  - 9 for cat to dog (12 and 15 give cat-dog hybrids).
- The CLIP edit estimate stays high for the sunglasses branch at step 15 (0.94) and the cat-to-dog branches at 12 and 15 (0.94 to 0.95), where the pictures show only a trace of the edit or a hybrid. This matches the editing pilot's finding that the estimate is generous.
- Files: `Results/m2_rewind_demo/`.

### Rewind live (November 3, 2026)

- A local server (`python -m w2w.live`) streams every step of a painting to the browser as it is painted. Changing the sentence repaints it from an earlier step. The automatic rewind point is 3 for a colour change, 6 for a swapped word and 9 for an added detail.
- On the RTX 5070 Laptop a full live painting takes 8.7 to 9.6 seconds, about 3 times a plain painting, because every step is decoded, safety-checked and sent. A branch takes 6.7 to 8.8 seconds, depending on how far back it starts.
- Six versions were tried:
  - the swap (cat to dog, after step 6) and the added detail (sunglasses, after step 9) kept the scene and showed the edit;
  - both colour edits from step 3 (black cat, yellow bus) showed the colour but redrew much of the scene.
- The safety checker blacks out the first two frames of "a cat sitting on a wooden bench", seed 1.
- Files: `Results/m2_rewind_live/`.

### Moments pilot (November 3, 2026)

- Loss arm, design: COCO captions from `eval_2000` were painted (seed 0) until 30 failures were found, where a failure is a named object that the photo contains and the detector cannot find (score 0.3). This took 281 captions. Each failure was rewound to after step 0, 3, 6, 9, 12, 15 and 18 and repainted 8 times with fresh randomness. The moment of loss is the earliest step from which at most 1 of 8 repaints shows the object.
- Loss arm, results: verdict `MOMENTS`.
  - 23 of 30 failures are recoverable from the start (at least 3 of 8 repaints at step 0); 6 have at most 1 of 8 even at step 0.
  - Among the 23, the median moment of loss is step 15 (interquartile range 10.5 to 19.5). 13% are lost by step 6, 26% by step 9, and 6 are not lost by step 18.
  - The median share of repaints with the object falls from 0.5 at step 0 to 0.125 at step 12 and 0 at step 18.
  - The early-warning test (the object word's attention share in the first three steps against the moment of loss) gave rho 0.37, 95% interval -0.08 to 0.74, so it does not pass.
- Loss arm, pictures: in 6 of the 30 failures the object is clearly in the base painting and the detector missed it:
  - 06, a tennis racket;
  - 12, a tennis ball;
  - 13, a vintage car;
  - 24, a black bear;
  - 25, a computer mouse;
  - 29, an airplane silhouette.

  Two more are borderline: 17, a small leather case for "suitcase"; 23, a white van for "car". Three of the 6 "not lost by step 18" failures are among the detector misses. They are kept in the results.
- Window arm, design: 4 swaps and 8 added details, seeds 0 and 1, each started after step 0 to 18, plain and with Prompt-to-Prompt local blending. An edit counts when its CLIP edit score is at least 0.6 (and, for swaps, the detector finds the new object) and at least 70% of the scene survives.
- Window arm, results: verdict `NO_GAIN`.
  - Over the 16 comparable units, the median latest usable start is 13.5 plain and 12 masked, a median gain of 0.
  - The mask raises how much scene survives at early starts (median 0.45 against 0 at step 0, 0.67 against 0.41 at step 3), but edits stop taking at the same steps in both conditions.
- Window arm, pictures:
  - With the mask, the wall, bench and couch stay still at starts 0 and 3 while the object is redrawn, and no seams are visible at the mask edge.
  - From step 6 the red hat no longer appears in either condition, and the sunglasses edits are nearly identical plain and masked.
- Files: `Results/m2_pilot_moments/`.

### Development and reference splits (November 5, 2026)

- Two more sets were drawn from the eligible images left after the test set, with the same seeded generator: `dev_500` (500 image-caption pairs, for choosing settings) and `ref_2000` (2,000 photos, the real reference for FID and KID). The three sets are disjoint by image, every image has a short side of at least 256 px, and 458 eligible images remain unused. `eval_2000` and `eval_500` are byte-identical to before (same SHA-256).

### Recall pilot (November 5, 2026)

- Design: every dev_500 caption that names an object its photo contains (334 captions, 395 instances) was painted once (seed 0, DDIM 30 steps, guidance 7.5) with attention recorded.
  - Faster R-CNN v2 and RetinaNet v2 judged each object: present if either scores at least 0.5, missing if neither reaches 0.3, ambiguous otherwise.
  - The test is whether the object word's commit step (r at least 0.8) separates missing from present objects, measured by AUROC with a caption-level bootstrap.
- Result: verdict `PREDICTS`.
  - 342 present, 36 missing (missing rate 0.095), 17 ambiguous; the detectors disagreed on 39 instances.
  - Median commit step: 5 for present objects, 13 for missing ones. No object word failed to settle; the latest missing commit is step 20.
  - AUROC 0.817 (95% interval 0.735 to 0.885). Within the same object class, 0.820 (0.683 to 0.927).
  - The early attention share (steps 0 to 2) also passes: AUROC 0.761 (0.657 to 0.847), within-class 0.756 (0.598 to 0.891).
- Pictures:
  - Of the 36 missing objects, 3 are clearly present: a giraffe (2809), a refrigerator (10592) and a flat-screen TV (12887). Five are borderline: a bun-like cake (9428), a sausage (18160), a laptop keyboard (1278), a small glass (2772) and a device that may be a remote (9659).
  - Removing the 3 clear misses gives AUROC 0.832 (0.742 to 0.905) and within-class 0.852 (0.705 to 0.949); the official result is unchanged.
  - Six objects called present were checked and are present.
- Files: `Results/m2_pilot_recall/`.

### Rescue pilot (November 5, 2026)

- Design: for the 34 captions with a missing object, the seed-0 painting was repainted exactly (largest mean difference from the recall pilot's image: 0.000). Each failure then got 4 tries under each condition:
  - a new seed (regeneration);
  - a new seed with the noisy sampler (eta 1);
  - going back to after step 3, 6, 9 or 12 and finishing with fresh randomness (rescue).

  A try is fixed when either detector finds the object at 0.5 or more. It keeps the scene when its LPIPS to the original is at most half the median LPIPS between unrelated paintings of the caption (0.794, so a threshold of 0.397).
- Result: verdict `TRADE_OFF`, best rescue step 12 (tied with 9 on kept@4; the later step wins ties).

  | Condition | fix@4 | kept@4 | median LPIPS of the best fix | Cost per try |
  |---|---|---|---|---|
  | Regeneration | 0.85 | 0 | 0.76 | 1.0 |
  | Rescue from 3 | 0.68 | 0.03 | 0.54 | 0.9 |
  | Rescue from 6 | 0.71 | 0.29 | 0.45 | 0.8 |
  | Rescue from 9 | 0.47 | 0.35 | 0.36 | 0.7 |
  | Rescue from 12 | 0.41 | 0.35 | 0.26 | 0.6 |

  - Rescue from 12 keeps the scene with the fix more often than regeneration: a difference of 0.35, interval 0.18 to 0.50.
  - Regeneration fixes 0.44 more often (interval 0.26 to 0.62).
  - Three failures were not fixed by any condition.
  - Sampler check `SAME`: eta 1 sharpness is 0.85 of eta 0, and the CLIPScore difference is 0.20 (interval -0.26 to 0.66).
  - Step 12 was chosen on these same failures, so its intervals are optimistic until the test-split run.
- Pictures:
  - Rescue from step 6 keeps the original layout (the same kitchen, the same cat on the fridge, the same man with the hot dog). Rescue from step 3 changes poses and shapes more.
  - The eta 1 regenerations do not look softer or noisier than the eta 0 ones.
  - Most fixes blend in. A cup added from step 3 sits on the pavement like a pasted sticker.
  - The bed in "A bed and chair sitting next to a window" never appears in any rescue.
- Files: `Results/m2_pilot_rescue/`.

### Word timing on the test split, seed_id 0 (November 18, 2026)

- Design: the confirmatory test of pilot 4, fixed in `docs/m2_preregistration.md` before the run.
  - All 2,000 captions of `eval_2000` were painted once under the group anchor (protocol m2-v1: DDIM, 30 steps, guidance 7.5, leading spacing, seed = caption_idx, CPU noise, safety checker on) with cross-attention recorded.
  - Every object in Praniti's labels was judged by Faster R-CNN v2 (score at least 0.5) or OWL-ViT (at least 0.1): present if either finds it, missing if both miss it.
  - The primary test uses rows 432 to 1999 only, because pilot 3 had already painted rows 0 to 431 in Milestone 1. All 2,000 captions are a sensitivity check.
- GPU check (`scripts/m2_check.py`): noise sum 36.4438 and the anchor's 30 timesteps both match the group's values, and Painter paints the protocol's 10 development images pixel for pixel (largest difference 0).
- Primary reading, commit step: `PREDICTS`.
  - Population: 1,801 objects in 1,319 captions; 9 objects in 8 images blacked out by the safety checker are excluded.
  - 1,710 present and 91 missing, a missing rate of 0.0505.
  - AUROC 0.678 (95% interval 0.631 to 0.724), within-class 0.700 (0.597 to 0.796).
  - Median commit step 15 for missing objects and 7 for present ones. No object word failed to settle.
- Secondary reading, early attention share: `WEAK`. AUROC 0.612 (0.557 to 0.668), within-class 0.686 (0.560 to 0.796).
- Sensitivity checks (all on the primary population except the first):

  | Check | AUROC (95% interval) | Within-class (95% interval) | Missing | Reading |
  |---|---|---|---|---|
  | All 2,000 captions | 0.677 (0.639 to 0.715) | 0.670 (0.587 to 0.750) | 120 | PREDICTS |
  | Pilot 4's judges | 0.635 (0.593 to 0.679) | 0.736 (0.639 to 0.821) | 106 | WEAK |
  | Faster R-CNN alone | 0.613 (0.576 to 0.651) | 0.688 (0.587 to 0.782) | 150 | WEAK |
  | Both judges agree | 0.688 (0.641 to 0.733) | 0.708 (0.606 to 0.806) | 91 | PREDICTS |
  | Commit at r 0.7 | 0.683 (0.634 to 0.731) | 0.729 (0.636 to 0.815) | 91 | PREDICTS |
  | Commit at r 0.9 | 0.675 (0.633 to 0.719) | 0.710 (0.584 to 0.819) | 91 | PREDICTS |
  | Without person | 0.817 (0.765 to 0.863) | 0.781 (0.702 to 0.852) | 77 | PREDICTS |
  | First object of each caption | 0.705 (0.640 to 0.765) | 0.687 (0.566 to 0.803) | 41 | PREDICTS |
  | Within object-size thirds | 0.678 (0.631 to 0.724) | 0.685 (0.636 to 0.734) | 91 | PREDICTS |

- Pictures:
  - Of 15 objects judged missing, 13 are missing.
    - 1956 (chair): a red chair is clearly present.
    - 1067 (airplane): borderline; a possible speck in the sky.
    - 8522 (banana): a safety-checker black image, which the analysis excludes.
  - Five objects judged present are present.
  - Five random images look like ordinary Stable Diffusion 1.5 pictures.
- Development grid: the first 200 dev captions under seven settings (leading, linspace and trailing spacing; guidance 3 and 12; seed_ids 1 and 2) were painted, recorded, judged and summarised. Each reads TOO_FEW on its own (13 to 16 missing objects), as expected; the paired comparison is the next step.
- Files: `Results/m2_timing/`, `Results/m2_check/`, `Results/m2_runs/`.

### The same test captions under three seeds (November 18, 2026)

- Design: the second pre-registered test (`docs/m2_preregistration.md`, "Second test: the same object under three seeds", written before any seed_id 1 or 2 image existed).
  - All 2,000 test captions were painted again with seed_ids 1 and 2, under exactly the anchor setting of the seed_id 0 run, then judged and analysed the same way.
  - Within the primary population (rows 432 to 1999, images not blacked out), an object is discordant when it is missing under at least one seed and present under another.
  - Each (missing seed, present seed) pair of a discordant object is won when the missing painting's commit step is later. The caption, the object's class and its size are the same on both sides, so none of them can produce the result.
- Per-seed primary results: all three read `PREDICTS`.

  | Seed_id | Present | Missing | Missing rate | AUROC (95% interval) | Within-class (95% interval) | Early share |
  |---|---|---|---|---|---|---|
  | 0 | 1,710 | 91 | 0.051 | 0.678 (0.631 to 0.724) | 0.700 (0.597 to 0.796) | WEAK |
  | 1 | 1,684 | 115 | 0.064 | 0.667 (0.624 to 0.709) | 0.692 (0.584 to 0.791) | WEAK |
  | 2 | 1,698 | 100 | 0.056 | 0.653 (0.609 to 0.695) | 0.632 (0.504 to 0.759) | CLASS_ONLY |

- Within-object test, commit step: `SEEN_COMING`.
  - 173 discordant objects in 164 captions.
  - Within-object AUROC 0.704 (95% interval 0.647 to 0.761).
  - The missing painting commits on average 3.3 steps later than a present painting of the same object (0.704 is the share of pairs where it is later, ties counted half; 68% go that way outright).
- Within-object test, early share: also `SEEN_COMING`. AUROC 0.627 (0.571 to 0.683); the share is lower in the missing painting in 67% of pairs (mean difference -0.010).
- Of 1,785 objects scored under every seed, 1,589 are present under all three, 115 are missing under one seed, 55 under two and 26 under all three.
- Pictures (6 discordant objects, all three seeds side by side):
  - Real differences: 8167 (chair: sinks only under seed 2), 2697 (horse: cows only under seed 0), 6662 (chair: missing on two sailing ships, plausibly on a ferry deck under seed 2).
  - Borderline judge calls: 17312 (person: a tiny skier still visible under seed 2), 12416 (toilet: a stylised bowl in an abstract pattern under seed 1), 24255 (laptop: a partial object under seed 0, Faster R-CNN 0.41). Seed 2's "present" horse in 2697 looks like a cow.
  - Commit steps went the predicted way in 4 of the 6. 12416 and 2697 went against it, and 6662 tied at 15 under every seed.
- Files: `Results/m2_timing/eval_2000__ddim-leading-s30-g7.5__s1/`, `__s2/` and `__seeds012/`.

### Development grid, paired comparison (November 19, 2026)

- Exploratory, not pre-registered. `scripts/m2_grid.py` compares each of the seven recorded runs on the first 200 dev captions with the anchor (DDIM, leading spacing, guidance 7.5, seed_id 0), object by object, with 2,000 bootstrap resamples of captions.
- Anchor: 221 objects in 169 captions, missing rate 0.068, median commit step 9 (timestep 661).
- No setting changes the missing rate: every paired interval contains 0.
- Spacing (linspace, trailing) leaves the commit step where it is (changes +0.18 and +0.06 steps, both intervals across 0) but moves the commit timestep up by about 21 and 36. So the word commits at about the same step whatever the spacing, not at the same noise level.
- Guidance moves the commit step: guidance 3 commits 0.95 steps later (0.51 to 1.34) and guidance 12 0.78 steps earlier (0.46 to 1.11). Rank agreement with the anchor: Spearman 0.89 and 0.95.
- Seed_ids 1 and 2 agree less with the anchor (Spearman 0.77 and 0.80) than any other setting, with no shift in the median.
- Within-object test across seed_ids 0 to 2 on these captions: commit step 0.861 (0.758 to 0.950) over 31 discordant objects, the missing painting committing 5.4 steps later on average; early share 0.607 (0.483 to 0.733), not seen.
- Files: `Results/m2_grid/`.

### Early-warning recordings and development results (November 19 and 20, 2026)

- Design: `docs/m2_detector_preregistration.md`, written before any of these recordings existed. Can a missing object be seen coming early in the painting, from what, and does the painting add anything to what the prompt already says?
- Recorded, for every painting, with `scripts/m2_detector.py`:
  - every token's cross-attention at all 30 steps, pooled to 16 × 16;
  - after each of the 11 decision steps, the latent and DDIM's guess of the finished latent;
  - previews: each guess decoded;
  - drafts: 5 DPM-Solver++ steps to the end from decision steps 0, 2, 4 and 8.

  All of them are scored by Faster R-CNN v2, RetinaNet v2, OWL-ViT and CLIP ViT-L/14.
- Paintings:
  - Development: 841 captions (eval_2000 rows 0 to 999 with labelled objects) × seed_ids 0, 1, 2 = 2,523, repainted from the word-timing runs.
  - Prospective: 851 captions (rows 1000 to 1999) × seed_ids 3, 4, 5 = 2,553 new pictures. These were judged, and their labels are unread until the detectors are locked.
- Integrity: 0 of 841 development repaints differ from the stored picture or measures, under each of seed_ids 0, 1 and 2.
- Previews: 838, 840 and 838 development paintings, and 845, 850 and 846 prospective. The rest were blacked out by the safety checker and are left out by design.
- Run time:

  | Phase | Minutes |
  |---|---|
  | record development | 141 |
  | record prospective | 176 |
  | judge prospective | 21 |
  | preview development | 282 |
  | preview prospective | about 278 |
  | features development | 1 |
  | develop | 6 |

  - The previews ran with `--batch 2` (about 6.6 s per painting). The default batch spilled video memory into system RAM and ran at about 80 s per painting.
  - `preview prospective` was cut off once overnight and resumed with the same command.
- Size: `data/m2_detector/` is 5.02 GB; the new pictures are 1.08 GB.
- **Development results** (out-of-fold, 5 folds by caption; development, not the test): 3,412 objects in 841 captions, 204 missing (6.0%). The ceiling, RetinaNet on the finished picture, is AUROC 0.991.

  AUROC by decision step k (k + 1 of 30 steps done):

  | k | prompt | prompt + attention | prompt + preview | full | full, boosted trees | prompt + draft | everything |
  |---|---|---|---|---|---|---|---|
  | 0 | 0.706 | 0.781 | 0.710 | 0.780 | 0.741 | 0.935 | 0.935 |
  | 1 | 0.706 | 0.789 | 0.725 | 0.794 | 0.773 | | |
  | 2 | 0.706 | 0.802 | 0.751 | 0.814 | 0.798 | 0.964 | 0.965 |
  | 3 | 0.706 | 0.813 | 0.792 | 0.838 | 0.813 | | |
  | 4 | 0.706 | 0.823 | 0.825 | 0.858 | 0.858 | 0.980 | 0.976 |
  | 6 | 0.706 | 0.849 | 0.878 | 0.902 | 0.901 | | |
  | 8 | 0.706 | 0.860 | 0.924 | 0.937 | 0.936 | 0.993 | 0.991 |
  | 10 | 0.706 | 0.862 | 0.953 | 0.958 | 0.955 | | |
  | 12 | 0.706 | 0.866 | 0.971 | 0.972 | 0.969 | | |
  | 15 | 0.706 | 0.869 | 0.984 | 0.982 | 0.983 | | |
  | 20 | 0.706 | 0.870 | 0.994 | 0.993 | 0.994 | | |

  - H1 (k = 2): the full detector beats the prompt by 0.107 (0.077 to 0.141), ADDS. Within the same object across seeds it reads SEEN_COMING, but only just: 0.583 (0.508 to 0.657) over 120 discordant objects.
  - H2: prompt + attention minus prompt + preview is +0.072 (0.044 to 0.101) at k = 0 and -0.091 (-0.121 to -0.065) at k = 10, ATTENTION_FIRST. The preview first passes attention at k = 4.
  - H3: the full detector reaches 0.90 at k = 6, and the everything detector at k = 0. Catch rates with 10% of present objects flagged, full against prompt: 0.49 against 0.26 at k = 3, 0.90 at k = 10 and 0.99 at k = 15. With drafts, prompt + draft catches 0.82 at k = 0, for 5 extra network evaluations.
  - H4: prompt + draft beats prompt + preview by 0.225 (0.184 to 0.271) at k = 0 and 0.213 (0.175 to 0.255) at k = 2, DRAFT_AHEAD. Attention and the preview add nothing to the draft at k = 2: +0.002 (-0.002 to 0.005), NOT_SHOWN.
  - The draft is scored by the same judges that make the labels, so part of its strength may be those judges agreeing with themselves on hard objects. The within-object test, where the prompt and the class are fixed, is the fairer check.
- These are development values. The pre-registered readings come from the one-time prospective test after the detectors are locked.
- Files: `Results/m2_detector/development/` (`summary.json`, `curves.csv`, `scores.csv`), `Results/m2_detector/features_development.csv`, `Results/m2_detector/runs/`, recordings in `data/m2_detector/`.

### Early warning: the locked one-time test (November 20, 2026)

- Design: `docs/m2_detector_preregistration.md`. Every detector was fit on all 3,412 development rows and locked (`Results/m2_detector/lock.json`: 89 detectors, models sha256 c6770b37fbdd, 15:12:35 UTC). It was then applied once to the prospective paintings: eval_2000 rows 1000 to 1999, seed_ids 3, 4 and 5, painted new and never used to fit or choose anything. `prospective/runs.log` holds one run.
- Prospective test: 3,433 objects in 851 captions, 190 missing (5.5%). The ceiling, RetinaNet v2 on the finished picture, is AUROC 0.986 (0.981 to 0.990); on development it was 0.991.
- AUROC by steps done, development (out-of-fold) / prospective test:

  | Steps done | prompt | prompt + attention | prompt + preview | prompt + draft | full | everything |
  |---|---|---|---|---|---|---|
  | 1 | 0.706 / 0.788 | 0.781 / 0.807 | 0.710 / 0.803 | 0.935 / 0.947 | 0.780 / 0.814 | 0.935 / 0.945 |
  | 3 | 0.706 / 0.788 | 0.802 / 0.825 | 0.751 / 0.827 | 0.964 / 0.962 | 0.814 / 0.847 | 0.965 / 0.964 |
  | 5 | 0.706 / 0.788 | 0.823 / 0.841 | 0.825 / 0.876 | 0.980 / 0.984 | 0.858 / 0.891 | 0.976 / 0.984 |
  | 9 | 0.706 / 0.788 | 0.860 / 0.871 | 0.924 / 0.957 | 0.993 / 0.991 | 0.937 / 0.961 | 0.991 / 0.991 |
  | 11 | 0.706 / 0.788 | 0.862 / 0.874 | 0.953 / 0.975 | | 0.958 / 0.976 | |
  | 16 | 0.706 / 0.788 | 0.869 / 0.879 | 0.984 / 0.991 | | 0.982 / 0.990 | |
  | 21 | 0.706 / 0.788 | 0.870 / 0.877 | 0.994 / 0.995 | | 0.993 / 0.995 | |

- The readings (prospective test, with development in brackets):
  - **H1, the painting adds to the prompt: ADDS.** At k = 2 (3 steps done), full minus prompt is +0.059 (0.027 to 0.090) [development +0.107]. The full detector's within-object AUROC reads `SEEN_COMING`: 0.693 (0.623 to 0.762) over 120 discordant objects in 115 captions [development 0.583].
  - **H2, attention before the picture: NOT_SHOWN.**
    - At k = 0, prompt + attention minus prompt + preview is +0.004 (-0.034 to 0.040), no clear difference [development +0.072, attention ahead].
    - At k = 10 it is -0.101 (-0.132 to -0.074): the preview is ahead [development -0.091].
    - The preview first passes attention at k = 1 [development k = 4].
    - Development's ATTENTION_FIRST did not hold on the test.
  - **H3, how early (descriptive):**
    - The full detector reaches AUROC 0.90 at k = 6 (7 steps done), the same as development. The everything detector does so already at k = 0.
    - Catch rate with 10% of present objects flagged, full against prompt: 0.58 against 0.35 at k = 3, 0.96 at k = 10 and 0.99 at k = 15.
    - With drafts, prompt + draft catches 0.86 at k = 0 (5 extra network evaluations) and 0.93 at k = 2.
  - **H4, the draft:**
    - **DRAFT_AHEAD.** Prompt + draft minus prompt + preview is +0.144 (0.112 to 0.180) at k = 0 and +0.135 (0.104 to 0.169) at k = 2 [development +0.225 and +0.213].
    - **STILL_ADDS: NOT_SHOWN.** Everything minus prompt + draft at k = 2 is +0.003 (-0.005 to 0.010) [development +0.002].
- Boosted-tree check (secondary, reported only): the full detector as gradient-boosted trees tracks the logistic regression within about 0.01 at every step (prospective 0.813 at 1 step, 0.836 at 3, 0.995 at 21).
- Reading it whole:
  - The prompt alone predicts a missing object well (0.79 on the test).
  - What the model is heading towards adds to it from the first steps.
  - A quick 5-step draft finished from the very first step reaches 0.95, close to the 0.986 ceiling, and attention and the one-step preview add nothing on top of it.
  - The claim that attention knows before the preview shows it held on development but not on the test.
  - The draft is scored by the same judges that make the labels. The within-object test, where the prompt is fixed, also puts the draft far ahead (about 0.83 at 1 step against 0.62 for the full detector).
- Figures: `how_early.png` (AUROC by step, both parts, with the ceiling), `within_object.png` (within-object AUROC by step) and `filmstrip.png`.
- The filmstrip is development data, chosen by the script: caption 11053, "A dog in the grass catching a frisbee.", object frisbee.
  - Under seed_id 1 a white frisbee is under the dog's paw in every draft and in the finished picture, with risk 0.00 throughout.
  - Under seed_id 0 a running dog with no frisbee appears from the first draft on, with risk 0.81, 0.94, 1.00 and 1.00, and the frisbee is really missing in the finished picture. A small magenta smear near the dog in that first draft fades by the second.
  - No frame is black.
- Files: `Results/m2_detector/` (`lock.json`, `models.joblib`, `features_prospective.csv`, `prospective/`, the figures).
