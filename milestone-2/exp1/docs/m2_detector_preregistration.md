# Pre-registration: how early can a lost object be seen coming? (Milestone 2, early warning)

Written on Thursday, November 19, 2026, before any of the recordings it analyses exist. First version 8 AM; revised at 9:45 AM the same day, still before any recording, to add the fast-forward drafts, a fourth scorer (CLIP), a third prospective seed, H4 and the human check. The analysis code is `scripts/m2_detector.py`; every summary it writes records the script's sha256, so any later change is visible.

## Why this analysis

Tasks 17 and 18 showed that single attention measures predict a missing object: the commit step reads PREDICTS on the test split (AUROC 0.678), and within the same object across seeds both the commit step (0.704) and the early attention share (0.627) read SEEN_COMING. Two numbers from those runs, computed on November 19 before this document, set the bar for anything called a detector:

- **The prompt alone predicts a lot.** A logistic regression that sees only the prompt (the object's class, how many objects the caption names, where the word sits, the caption's length and the word's token count) reaches AUROC 0.754 on the three seed runs of the test split (5-fold cross-validation, folds by caption). On the same rows the commit step scores 0.669 and the early share 0.618. A detector has to beat the prompt, not chance.
- **The labels set a ceiling.** RetinaNet v2, which plays no part in the labels, scores the finished pictures at AUROC 0.982 against them. No detector judged by these labels can be expected to do much better, however late it looks.

Closest published work: HEaD+ (Betti et al., IJCV 2026) predicts, per object, whether it will be missing, from cross-attention maps, a decoded guess of the final image and the object's text embedding, and restarts flagged paintings with a new seed. It reports catch rates at a fixed recall, with no prompt-only baseline and no comparison across seeds of the same prompt. Early quality prediction from attention maps or activations (Diffusion Probe and Probe-Select, both CVPR 2026) predicts image-level scores, also without a prompt-only baseline.

So this analysis asks four questions:

1. **What does the painting add to the prompt**, step by step?
2. **Which part of the painting knows first**: the attention, or the picture the model is heading towards?
3. **How early is the detector good enough**, measured against the ceiling?
4. **Is a cheap draft enough?** Finishing the painting roughly, in a few large steps, is the obvious strong baseline. Does it beat the model's one-step guess, and do attention and the guess still add anything to it?

## Data

| Part | Captions | Seeds | Paintings |
|---|---|---|---|
| Development | `eval_2000` rows 0 to 999 with labelled objects (841) | seed_ids 0, 1, 2 | 2,523, repainted from tasks 17 and 18 with the fuller recorder below; each must reproduce its stored picture exactly |
| Prospective test | `eval_2000` rows 1000 to 1999 with labelled objects (851) | seed_ids 3, 4, 5 | 2,553, new; nobody has made them |

- Objects: Praniti's labels (`groundtruth_eval_2000.csv`, rows with `target` true).
- Labels: the group judges on the finished picture, as in task 17. Faster R-CNN v2 at 0.5 or OWL-ViT at 0.1; an object is missing when both miss it.
- Objects in pictures the safety checker blacks out are left out.
- Setting: the group anchor (protocol `m2-v1`; DDIM, 30 steps, leading spacing, guidance 7.5, CPU noise, safety checker on), painted with `Painter` as in task 17.

No detector is fit on the prospective captions, and their seed_id 3 to 5 paintings are new. Their seed_id 0 to 2 paintings were part of tasks 17 and 18, which tested single measures only.

**Integrity.**

- A development painting that does not reproduce its stored picture pixel for pixel is left out and listed. If more than 1% differ, the analysis stops.
- Nobody looks at the prospective labels or detector scores before the detectors are locked. The `test` phase refuses to run until `Results/m2_detector/lock.json` exists, checks the locked models against their recorded sha256, and appends every run, with its time, to `Results/m2_detector/prospective/runs.log`.

## What is recorded for each painting

- **Attention.** Cross-attention at all 30 steps, aggregated as before (heads averaged, every layer upsampled to 64 × 64, layers averaged), then average-pooled to 16 × 16, for the start token, every caption token and the end token.
- **Decision steps** k in {0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20}. Steps count from 0, so after step k, k + 1 of the 30 steps are done. After each one, the latent itself and DDIM's predicted x0 (the model's one-step guess of the finished latent) are kept.
- **Previews.** Each one-step guess is decoded with SD 1.5's VAE.
- **Drafts** (fast-forward). At decision steps 0, 2, 4 and 8, the kept latent is finished quickly: five DPM-Solver++ (2M) steps, with timesteps spaced evenly from the next one down to 0, guidance 7.5, the same prompt, decoded with the VAE. A draft costs 5 extra network evaluations, a sixth of a full painting.
- **Scorers.** Every preview and draft is scored by the same three judges as finished pictures (Faster R-CNN v2, RetinaNet v2, OWL-ViT base-patch32 with its 80 class queries: the best score for the object's class), and by CLIP ViT-L/14: 100 times the cosine similarity between the picture and "a photo of a {class}". The labels stay the group judges' verdict on the finished picture.

## Features available at decision step k

Only what exists by the end of step k.

**Prompt** (the same at every step): the object's COCO class (one-hot; classes with fewer than 10 development rows share one column), the number of labelled objects in the caption, the word's position (word index over the caption's length in words), the caption's length in words, the number of tokens in the object's words, and whether the class is person.

**Attention.** The object's measured token o is the one the commit step uses. A_t(s) is token s's 16 × 16 map at step t. Caption tokens leave out the start and end tokens; content tokens are those of content words, as for the early share.

| Feature | Definition at step t |
|---|---|
| share | the object's share of the attention given to content tokens (task 17's early share, per step) |
| peak | at each position, A_t(o) over the sum of A_t over caption tokens; smoothed with a 3 × 3 Gaussian (sigma 0.5, reflected edges); the maximum. Attend-and-Excite's neglect signal (Chefer et al., 2023) without its softmax sharpening |
| focus | 1 minus the entropy of A_t(o), scaled to sum to 1, divided by log 256 |
| overlap | the largest, over the caption's other labelled objects q, of the sum over positions of min(p_o, p_q), where p is a map scaled to sum to 1; 0 when the caption has one labelled object |
| lead | share(o) / (share(o) + the largest share of another labelled object); 1 when there is none |
| steady | Pearson r between A_t(o) and A_{t-1}(o), from step 1 |

Each enters twice: its value at step k and its mean over steps 0 to k (for steady, 1 to k, and none at k = 0). That gives 12 attention features, 10 at k = 0.

**Preview:** the four scorers on the preview after step k, and the best each gave on the previews up to step k (8 features).

**Draft** (at draft steps only): the four scorers on the draft from step k, and the best each gave on the drafts up to step k (8 features).

The three judges' scores enter as logits, clipped to [0.0001, 0.9999]; the CLIP score enters as it is. A missing value (an object without content tokens, for example) is replaced by the development median.

## Detectors

For each decision step, L2-regularized logistic regressions (C = 1, standardized features, equal class weights, scikit-learn):

| Family | Features | Steps |
|---|---|---|
| prompt | prompt | all |
| attention | attention | all |
| preview | preview | all |
| draft | draft | draft steps |
| prompt + attention | both | all |
| prompt + preview | both | all |
| prompt + draft | both | draft steps |
| full | prompt, attention and preview | all |
| everything | prompt, attention, preview and draft | draft steps |

The prompt family is the baseline. Secondary: the full detector as gradient-boosted trees (HistGradientBoostingClassifier, 200 iterations, learning rate 0.05, 15 leaves, random_state 0), reported only; it cannot change a reading.

## Evaluation

- **Development:** 5-fold cross-validation with folds by caption: the captions, in a random order drawn with seed 0, are dealt into the 5 folds in turn, so every seed of a caption sits in the same fold. The same folds serve every family and step. Out-of-fold scores.
- **Prospective:** every detector refit on all development rows, locked, then applied once.

Measures, on objects in pictures the safety checker kept:

- **AUROC** (ties count half).
- **Average precision.** With about 6% of objects missing, a coin scores about 0.06.
- **Catch rate:** the share of missing objects flagged when the threshold is the 90th percentile of the present objects' scores, so 10% of present objects are flagged. This is the "90% recall of present objects" point HEaD+ reports.
- **Within-object AUROC:** task 18's statistic on the detector's score, over pairs of paintings of the same object that differ only in seed, one missing and one present. The prompt is identical inside each pair, so this isolates what the painting adds.
- **Cost:** extra network evaluations per decision (0 for attention and previews, 5 per draft).

95% intervals come from 2,000 bootstrap resamples of captions (seed 0). Differences between families are paired (the same resamples). Development intervals ignore fold-to-fold variation.

## Hypotheses and readings

Read on the prospective test, with the development values reported beside them.

**H1, the painting adds to the prompt.** At k = 2, a tenth of the way through:

| Measure | Reading |
|---|---|
| AUROC of the full detector minus the prompt family's | ADDS if the paired interval is above 0, NOT_SHOWN otherwise |
| the full detector's within-object AUROC | task 18's rule: TOO_FEW under 30 discordant objects, SEEN_COMING if the interval is above 0.5, NOT_SEEN otherwise |

**H2, the attention knows before the picture shows it.** The AUROC difference, prompt + attention minus prompt + preview, at k = 0 and at k = 10. ATTENTION_FIRST if it is above 0 at k = 0 and below 0 at k = 10, both intervals excluding 0. Otherwise each step's result is reported as found. Also reported: the first decision step at which prompt + preview's AUROC passes prompt + attention's.

**H3, how early.** Descriptive: the first decision step at which the full detector's AUROC reaches 0.90, the same for the everything detector over the draft steps, and catch rates at k = 3, 10 and 15 (about 10%, 33% and 50% of the steps, near HEaD+'s 5, 16 and 25 of 50) for the full and prompt families.

**H4, the draft.** At k = 0 and k = 2:

| Measure | Reading |
|---|---|
| AUROC of prompt + draft minus prompt + preview | DRAFT_AHEAD if the paired interval is above 0 at both steps, NOT_SHOWN otherwise |
| AUROC of everything minus prompt + draft | STILL_ADDS if the paired interval is above 0 at k = 2, NOT_SHOWN otherwise |

**Ceiling**, shown on every figure: RetinaNet's score on the finished picture, as AUROC against the labels on the same objects.

## Human check of the labels (exploratory, fixed now)

240 finished development pictures, drawn with seed 0 in three equal groups: objects both judges call missing, objects exactly one judge finds, objects both judges find. Each is labelled by two team members, independently and without seeing the judges' calls, as present, absent or unclear. Reported: how often the two people agree, how often each judge agrees with them, and the full detector's AUROC against the human labels on these pictures, with each group weighted by its share of all development objects. It cannot change any reading above.

## Fixed now, exploratory later

Fixed by this document: the data, the recording, the decision and draft steps, the scorers, the features, the families and their settings, the folds, the measures, the readings and the human check's design. Anything else is exploratory and labelled that way, including other classifiers or features and the example paintings chosen for figures (taken from development data only).

Outputs: `Results/m2_detector/` (`development/`, `lock.json`, `prospective/` and the figures). Recordings: `data/m2_detector/`. The prospective pictures join the anchor's other images in `data/m2_images/eval_2000/ddim-leading-s30-g7.5/` and the group's generation log.
