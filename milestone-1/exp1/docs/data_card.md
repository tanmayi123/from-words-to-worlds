# Data card

Machine-readable version: `Results/m1_data/data_card.json`.

**Source.** MS-COCO 2017 validation split (images + captions_val2017.json + instances_val2017.json).

| Item | Value |
|---|---|
| Images | 5,000 |
| Captions after cleaning | 25,004 |
| Captions per image | 4 to 7 (median 5) |
| Captions with whitespace repaired | 3,832 |
| Duplicate captions dropped | 10 |
| Captions longer than 77 tokens | 0 |
| Images converted to RGB | 10 |
| Image size after preprocessing | 512 x 512, center crop, Lanczos, PNG |
| Fraction of the frame kept by the crop | mean 71.6%, minimum 24.8% |
| Images in the reference set (short side of at least 256 px) | 4,958 |
| Images excluded from the reference set | 42 |
| Images with at least one labelled object | 4,952 |
| Evaluation split | 2,000 pairs, one caption per image, seed 2026 (`Results/m1_data/eval_2000.csv`) |
| Variance split | first 500 rows of the evaluation split (`Results/m1_data/eval_500.csv`) |
| Development split | 500 pairs, one caption per image, drawn after the evaluation split with the same generator; used to choose settings, never the test set (`Results/m1_data/dev_500.csv`) |
| Reference split | 2,000 photos, drawn after the development split; used only as the real reference for FID and KID (`Results/m1_data/ref_2000.csv`) |
| Unused eligible images | 458 (the three splits are disjoint by image) |

**Caption policy.** Whitespace and spacing-before-punctuation repaired; exact duplicates within an image (case-insensitive) dropped; wording, spelling and case never changed.

**Known limitation.** For very wide or tall photos the center crop can cut away what the caption describes. 12 of the 2,000 evaluation pairs keep less than half of the original frame; they are flagged for any per-image comparison with the real photo. Image-level metrics are unaffected as long as real and generated images get the same processing.

**Licences.** COCO annotations are CC BY 4.0. COCO images keep their original Flickr licences and are not redistributed in this repository; the code downloads them from the official COCO site.
