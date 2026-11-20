# Milestone 2: Conditional Generation and Measurement

Part of [From Words to Worlds](../README.md). Report: [`report/IE7615_P2_M2_Group7_report.pdf`](report/IE7615_P2_M2_Group7_report.pdf) (8 pages, with a one-page summary on page 1).

Stable Diffusion 1.5, conditioned by its CLIP ViT-L/14 text encoder, run by all four of us through one shared protocol. Same captions, same starting noise for every caption, same two object judges, same log format. Together we generated and logged **23,673 images in 47 runs**, and trained the first stage of a connector that lets BERT drive the same model.

## Layout

| Folder | Contents | Lead |
|---|---|---|
| [`protocol/`](protocol) | `m2_protocol.py` (version `m2-v1`, sha256 starts `8b9a2185`): the anchor setting, the seed rule and the run log every member used | Santhoshkumar |
| [`splits/`](splits) | the caption sets (`eval_2000`, `eval_500`, `dev_500`), their object labels, the stress prompts and the FID/KID reference set `ref_2000_v2` | Santhoshkumar, Praniti |
| [`logs/`](logs) | the merged generation log (every image, every run) and the connector's training log | Santhoshkumar |
| [`report/`](report) | the Milestone 2 report | all |
| [`exp1/`](exp1) | when words are painted; seeing a missing object coming (pre-registered, one-time test); the human check of the judges | Santhoshkumar |
| [`exp2/`](exp2) | noise schedules on DPM-Solver++ 2M; BERT connector, stage 1 | Mukul |
| [`exp3/`](exp3) | stress-prompt benchmark; sampler and guidance grid | Praniti |
| [`exp4/`](exp4) | guidance and step-count sweeps; the shared scoring code (FID, KID, IS, CLIPScore, both judges) | Tanmayi |

Each experiment folder has its own README with what it tests, how to run it and its results.

## The shared protocol

| Setting | Value |
|---|---|
| Model | Stable Diffusion 1.5, float16, safety checker on |
| Anchor | DDIM, 30 steps, leading timestep spacing, guidance 7.5, eta 0 |
| Starting noise | drawn on the CPU from seed `100000 × seed_id + caption_idx`, passed in as latents, so every machine paints a caption from the same noise |
| Judges | an object counts as present if Faster R-CNN v2 scores it at least 0.5 or OWL-ViT (`a photo of a {class}`) at least 0.1; on the real photos the judges find 98.9% of objects |
| Settings | chosen on `dev_500` only; reported on `eval_500` / `eval_2000` |
| Log | every image in `runs.csv` and `images.csv`; runs made before the protocol keep their own seed rule and are marked as such |

## Milestone 2 results

| Question | Answer |
|---|---|
| Guidance and step count | Only guidance 3 lowers object recall detectably (−2.0 points); 10 to 50 steps leave it unchanged. On the 500 test captions, DDIM at 10 steps keeps recall (+0.0 points [−1.0, +1.0]) at 1.7 s per image instead of 4.7 s. |
| Noise schedule | DPM-Solver++ 2M with leading spacing at 10 steps matches the 30-step DDIM anchor on recall and KID at 37% of the time. |
| Stress prompts | Colour binding passes 3%, multi-object 13%, spatial 21%, counting 29%. The usual failure is an object never drawn. |
| Text encoder | The BERT connector reaches held-out loss 0.24 (1.0 = predicting the mean CLIP state) and cosine 0.90 to CLIP's states, with no overfitting. |
| When words are painted | A word's commit step predicts whether its object goes missing (AUROC 0.68 under each of 3 seeds); only 26 of 196 lost objects are lost under every seed. |
| Seeing a missing object coming | Pre-registered, sealed, tested once on 2,553 new paintings: a 5-step draft made after the first step scores AUROC 0.947 against a ceiling of 0.986. |
| Are the judges right? | 240 judge calls labelled blind by two of us: people agree 95% of the time (kappa 0.90); the rule's "missing" is right 97% of the time. |

All intervals are 95% bootstrap intervals over captions; comparisons between settings are paired on the same captions and noise.

## Data and licences

COCO data is not stored in this repo: annotations are CC BY 4.0, and images keep their Flickr licences and are used for research only. Generated images, model checkpoints and feature files are on Drive and linked from each experiment's README.
