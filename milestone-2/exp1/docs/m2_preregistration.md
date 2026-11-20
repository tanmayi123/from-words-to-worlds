# Pre-registration: does a late commit step predict a missing object? (Milestone 2, test split)

Written on Wednesday, November 18, 2026, before the run it describes. The analysis code (`scripts/m2_timing.py`) and every threshold below are fixed now; `summary.json` records the sha256 of the script and of the shared protocol, so any later change is visible.

## Question and hypothesis

When Stable Diffusion paints a caption, each word's cross-attention map settles into its final layout at some denoising step: its commit step. The hypothesis is that objects whose word commits late are the ones that go missing from the finished picture.

Evidence so far:

- Pilot 4, exp1's development split, 334 captions, 36 missing objects: AUROC 0.817 [0.735, 0.885], within-class 0.82 [0.683, 0.927], read as PREDICTS.
- exp4, 60 captions of the test split, PNDM sampler: AUROC 0.86 [0.76, 0.94] on only 6 missing objects, too few to read.

This run is the confirmatory test, read once by the rules below.

## Which test captions count

Two earlier experiments already painted part of the test split. Pilot 3 (the moment-of-loss arm, Milestone 1) went through `eval_2000` in order, painting every caption that names an annotated object, until its 30th failure at row 431. exp4's 60 word-timing captions are rows 2 to 94, inside the same block. Pilot 3 noted the commit step of its 30 failures but compared nothing with present objects, so only exp4's small look tested this hypothesis there. Still, a confirmatory test should rest on captions nobody has studied, so:

- **Primary population:** rows 432 to 1999 of `eval_2000` (1,568 captions; 1,809 labelled objects in 1,326 of them).
- **Sensitivity:** all 2,000 captions.

All 2,000 are painted either way, since the images are also the anchor's test-split images for the image-quality scores. (Separately, Milestone 1's five sample captions describe three photos of the test split, rows 89, 474 and 1349, but they are not the split's own captions for those photos.)

## Design

| | |
|---|---|
| Captions | `eval_2000`, one image per caption, seed_id 0; the primary test uses rows 432 to 1999 |
| Objects | Praniti's `groundtruth_eval_2000.csv`, rows with `target` true: 2,291 objects in 1,692 captions over the whole split, 1,809 in the primary population |
| Generation | the group anchor (protocol `m2-v1`): SD 1.5 in float16, DDIM, 30 steps, eta 0, guidance 7.5, leading spacing, seed = caption_idx, starting noise drawn on the CPU, safety checker on. Painted with exp1's `Painter`, which records cross-attention and paints the same picture as the shared protocol (checked by test) |
| Excluded | objects in images the safety checker blacks out |

**Commit step.** The object's token is the last token, among the tokens of its words, that contains a letter or digit ("people," is measured on "people"). Its attention map at each of the 30 steps is compared, by Pearson r, with its reference layout: the mean map of the 5 evaluations before the last, with the last left out. The commit step is the first step from which r stays at least 0.8 at every later step except the last. A word that never settles gets 30.

**Early share.** The object's share of the attention given to the caption's content words, averaged over steps 0 to 2.

**Judges** (exp4's, calibrated on real photos). Faster R-CNN v2 with a class score of at least 0.5, and OWL-ViT base-patch32 with the query "a photo of a {class}" and a score of at least 0.1. An object is present when either judge finds it and missing when both miss it.

## Primary test and how it will be read

AUROC: the probability that a missing object has a later commit step than a present one (ties count half). Its 95% interval comes from 2,000 bootstrap resamples of captions (seed 0). The within-class AUROC counts only pairs of the same COCO class, so it can't be driven by some classes being both late and often missed.

| Reading | Rule |
|---|---|
| TOO_FEW | fewer than 30 missing objects |
| PREDICTS | AUROC at least 0.65, its interval above 0.5, and the within-class interval above 0.5 |
| CLASS_ONLY | the AUROC interval is above 0.5 but the within-class interval is not |
| WEAK | both intervals above 0.5, AUROC below 0.65 |
| NONE | the AUROC interval includes 0.5 |

NONE or CLASS_ONLY would mean the pilot's result did not hold on new captions. Since the labels, judges and safety checker differ from the pilot's, an AUROC somewhat below 0.82 would not be surprising; the reading above is what counts.

**Secondary.** The early share, with lower share predicting missing, read by the same rules.

## Sensitivity checks

Reported in full next to the primary result. None of them can change the primary reading. All but the first use the primary population.

- all 2,000 captions, including the 432 painted before;
- pilot 4's judges (Faster R-CNN v2 and RetinaNet v2: present at 0.5 or more, missing below 0.3, in between left out);
- Faster R-CNN alone;
- only objects both judges agree on;
- commit thresholds r of 0.7 and 0.9;
- without the person class (a third of all objects);
- only the first object of each caption;
- within thirds of object size (pairs compared only inside the same third).

## Changes from pilot 4

The split (test instead of development), the labels (Praniti's instead of caption word matching), the judges (exp4's two instead of two detectors), the seed rule and CPU noise of the shared protocol (pilot 4 used seed 0 on the GPU for every caption), and the safety checker (now on, as in the stock pipeline).

## After the run

One run of the primary test. No re-running with other thresholds or judges to change the reading, and the extra seeds below do not replace seed_id 0's result. Anything not listed here is exploratory and labelled that way, including the development-split grid that varies spacing, guidance and seed.

## Second test: the same object under three seeds

Added on the same day, before any image with seed_id 1 or 2 exists.

The primary test compares different objects. Objects differ in class, size and how hard their caption is, and the within-class AUROC removes only the first of these. The proposal asks something sharper: whether a word that is about to be dropped can be seen coming. So the whole test split is painted twice more, with seed_ids 1 and 2, under the same setting. A caption and its objects are then fixed, and only the starting noise changes.

- **Population:** the primary population (rows 432 to 1999), objects in images not blacked out by the safety checker.
- **Discordant objects:** judged missing under at least one seed and present under another, by the same group judges.
- **Statistic:** for each discordant object, every (missing seed, present seed) pair is one comparison, won when the missing painting's commit step is later (ties count half). The within-object AUROC is all wins over all comparisons. Nothing about the caption, the object's class or its size can produce it. 95% interval from 2,000 bootstrap resamples of captions (seed 0).
- **Reading:** TOO_FEW with fewer than 30 discordant objects; SEEN_COMING if the interval lies above 0.5; NOT_SEEN otherwise.
- **Same rules, second measure:** the early attention share (steps 0 to 2), with a lower share in the missing painting counting as a win. SEEN_COMING here would mean the loss shows by the third step of 30.
- **Also reported:** the primary test repeated under seed_ids 1 and 2, and how many objects go missing under 0, 1, 2 or 3 of the seeds (a failure under every seed points at the caption; a failure under some seeds points at the noise).

Outputs: `Results/m2_timing/eval_2000__ddim-leading-s30-g7.5__s1/` and `__s2/`, and the comparison in `Results/m2_timing/eval_2000__ddim-leading-s30-g7.5__seeds012/summary.json`.

Outputs: `Results/m2_timing/eval_2000__ddim-leading-s30-g7.5__s0/` (`objects.csv`, `curves.csv`, `captions.csv`, `judges.csv`, `objects_judged.csv`, `summary.json`, `commit_vs_recall.png`). `curves.csv` keeps every object's correlation and attention share at each of the 30 steps, so Milestone 3 can ask when a missing object is lost relative to its commit step without repainting. Images: `data/m2_images/eval_2000/ddim-leading-s30-g7.5/`.
