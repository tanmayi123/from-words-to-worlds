# Design decisions

Each decision, the reason for it, and the date it was made.

| Date | Decision | Reason |
|---|---|---|
| Nov 2, 2026 | Dataset: MS-COCO 2017 validation (5,000 images, about 25,000 captions) | Natural captions, a real photo per caption, object labels for 80 categories, and the standard benchmark for zero-shot text-to-image FID |
| Nov 2 | Model: Stable Diffusion 1.5, frozen, from `stable-diffusion-v1-5/stable-diffusion-v1-5` | Its text encoder is CLIP ViT-L/14, so "CLIP plus diffusion" is one model; its UNet uses standard noise prediction, so guidance and scheduler experiments map cleanly |
| Nov 2 | Encode prompts without an attention mask, one caption at a time | Matches the pipeline exactly; batching changed results through half-precision arithmetic |
| Nov 2 | Cache 77 x 768 token states, pooled vectors and the empty-prompt states | The UNet reads token states; pooled vectors serve retrieval and scoring; the empty prompt is what guidance runs against |
| Nov 2 | Center-crop then resize to 512 x 512, never stretch; save as PNG | COCO photos are mostly 4:3 or wider; stretching distorts shapes and would bias image metrics; lossless storage keeps real and generated images comparable |
| Nov 2 | Exclude images with a short side under 256 px from the image reference set | Enlarging them more than twice blurs them, which would penalise every generated image |
| Nov 2 | Freeze evaluation splits (2,000 and 500 pairs, seed 2026) | Every setting must be compared on exactly the same prompts |
| Nov 2 | Report every FID beside the real-against-real score at the same sample size; use KID for sweeps | FID shrinks strongly with sample size (339.6 for 10 real photos, 98.3 for 500) |
| Nov 2 | Recorder passes the computation to the stock attention processor | Recomputing the output changed images on the GPU; delegation keeps them identical |
| Nov 2 | Record only the conditional branch of classifier-free guidance | It is the branch that reads the prompt; the unconditional branch reads the empty prompt |
| Nov 2 | Commit step: first step after which the heatmap keeps r of at least 0.8 with the mean map of the five evaluations before the last | A stable, threshold-swept definition; the last evaluation is excluded because nearly every map decorrelates there at once |
| Nov 2 | Word shares renormalised over the prompt's own words | The start token takes about 90 percent of attention and would hide every word |
| Nov 2 | Milestone 1 keeps the default PNDM scheduler; later milestones use one evaluation per step | Keeps the Milestone 1 images identical to the smoke test; a cleaner time axis for the full study |
| Nov 2 | All timings are measured on AC power | The laptop runs about 2.5 times slower on battery |
| Nov 5 | Separate development and reference sets (`dev_500`, `ref_2000`), disjoint by image from the test set | Settings must never be chosen on the test set, and FID and KID need a real reference that shares no photo with the test set, at the same sample size |
