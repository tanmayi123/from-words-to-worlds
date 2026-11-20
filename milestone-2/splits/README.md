# Milestone 2 splits

One set of splits for every Milestone 2 run. Six files are byte-for-byte copies of Praniti's Milestone 1 files in `milestone-1/exp3/results/`; `ref_2000_v2.csv` is new. Check a file with `sha256sum <file>` (or `hashlib.sha256(open(f, "rb").read()).hexdigest()` in Python); the first 16 characters must match.

| File | Rows | Use | sha256 (first 16) |
|---|---|---|---|
| `eval_2000.csv` | 2,000 captions | test set, never used for choices | `352b3f6183ead6d7` |
| `eval_500.csv` | 500 captions | the first 500 rows of eval_2000 | `d550fdee6a500dde` |
| `dev_500.csv` | 500 captions | every tuning choice (seed 2027) | `ce3ea05b947c39aa` |
| `ref_2000_v2.csv` | 2,000 photos | real-photo reference for KID and FID | `50b3b51c3d53aecc` |
| `groundtruth_eval_2000.csv` | 2,768 objects | object labels for eval_2000 (2,291 scorable targets in 1,692 captions) | `363904c53d082449` |
| `groundtruth_dev_500.csv` | 674 objects | object labels for dev_500 (559 scorable targets in 424 captions) | `a79db08f6885a803` |
| `stress_prompts.csv` | 200 prompts | five families of 40: colour binding, counting, multi-object, spatial, word order | `33606d070aef5408` |

A label row is a scorable target when `target` is true: the word names a COCO class without ambiguity, the class is in the photo, and its box survives the square crop.

## The new reference set

Milestone 1 had two reference sets, and both overlapped photos used elsewhere: Praniti's `real_ref_b_2000` shares 353 photos with her `dev_500`. A reference set that contains the photos behind our prompts makes KID and FID look better than they are, so `ref_2000_v2` is drawn away from everything we generate from.

- Pool: the 4,958 COCO val2017 photos marked `reference_eligible` in Praniti's `preprocess_log.csv` (short side at least 256 pixels), minus the 2,000 photos of eval_2000, the 500 of Praniti's dev_500 and the 500 of exp1's Milestone 1 dev set (the two dev sets share 88). 2,046 photos remain.
- Draw: 2,000 of them with Praniti's `draw_outside` code, `random.Random(2031).sample(sorted(pool), 2000)`.
- Checks: no overlap with eval_2000 or either dev set; no duplicates. It shares 1,332 photos with `real_ref_b_2000`, which is expected, since both are large draws from the same pool.
- Fingerprint of the id list: `5b93d8e075932b60` (sha256 of the image ids joined by newlines, first 16 characters).

`real_ref_b_2000.csv` and exp1's `ref_2000.csv` are retired for Milestone 2. The Milestone 1 folders are left as they were.
