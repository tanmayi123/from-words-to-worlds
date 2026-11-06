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
