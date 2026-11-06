# Milestone 1 · Exp 3 — Evaluation groundwork, text-encoder comparison and re-captioning study

**Author:** Praniti Sunil Kale · IE 7615 Deep Learning for AI · Northeastern University

This experiment builds the data and evaluation groundwork the rest of the project is scored with, and adds two studies that need no training:

1. **Milestone 1 pipeline + evaluation sets.** COCO 2017 val prepared with light cleaning; frozen test, dev and real-reference splits; object labels for every test caption with a crop check; 200 stress prompts; a text → embedding cache verified against Stable Diffusion's own output; first samples.
2. **Text encoders compared without training.** CLIP ViT-L/14 (Stable Diffusion's own encoder), BERT-base, T5-base and GPT-2 on the same 2,000 test captions, plus a TF-IDF word-count baseline.
3. **Re-captioning study.** Is Stable Diffusion's missing content the fault of the model, the captions, or the 77-token limit? 100 test photos, four captions each (human, short, detailed, detailed fitted to 77 tokens), same seeds, judged against the real photo.

---

## Folder contents

```
milestone1/exp3/
├── README.md
├── notebooks/
│   ├── 01_M1_plus_TextEncoder_Comparison.ipynb   ← Part 1: Milestone 1 pipeline; Part 2: text-encoder comparison (run, with outputs)
│   └── 02_Recaptioning_FittedCaptions.ipynb      ← re-captioning study, runs from scratch (run, with outputs)
├── reports/
│   ├── M1_plus_TextEncoder_Report.pdf            ← every choice, result and figure for notebook 01
│   └── Recaptioning_FittedCaptions_Report.pdf    ← every result and figure for notebook 02
├── results/
│   ├── milestone1_summary.json                   ← every Milestone 1 and text-encoder number in one file
│   ├── config.json                               ← every setting and seed
│   ├── manifest.jsonl, captions.csv              ← one row per image / per caption (25,004 captions)
│   ├── preprocess_log.csv                        ← size, mode, crop share, eligibility per image
│   ├── data_checks.json                          ← EXIF tags, near-duplicates, caption quirks (counted, never changed)
│   ├── splits/                                   ← eval_2000, eval_500, dev_500, real_ref_b_2000
│   ├── groundtruth_eval_2000.csv, groundtruth_dev_500.csv, groundtruth_summary.json  ← object labels + crop check
│   ├── stress_prompts.csv                        ← 200 prompts with expected answers
│   ├── split_words.json, retrieval.json          ← tokenizer analysis, caption → photo retrieval
│   ├── text_encoders/                            ← CKA, paraphrase retrieval, scene clustering, word-order/colour tests
│   ├── recaptioning/named_read_missing.csv       ← objects named, read by SD, and still missing (for rescue experiments)
│   └── figures/                                  ← every figure from notebook 01
└── samples/                                      ← the 15 Milestone 1 sample images + samples.csv (caption, seed, settings)
```

---

## How to run

Both notebooks run in **Google Colab with a T4 GPU** (Runtime → Change runtime type → T4 GPU), then **Run all**. On Colab they mount Google Drive and save everything to `MyDrive/w2w_praniti/`; COCO is downloaded to the Colab disk each session.

| Notebook | What it does | Time on a T4 |
|---|---|---|
| `01_M1_plus_TextEncoder_Comparison.ipynb` | Downloads COCO val2017, cleans and preprocesses it, freezes the splits, builds object labels and stress prompts, caches the 77 × 768 CLIP states, checks them, generates the samples; then embeds the test captions with four encoders and compares them | ~35 min |
| `02_Recaptioning_FittedCaptions.ipynb` | Rebuilds the test split, downloads the 100 photos, captions them with Florence-2, fits the detailed captions to 77 tokens, generates 400 images, scores them against the real photos | ~45 min from scratch, ~15 min when the images are already on Drive |

Every random draw uses a fixed seed. Rerunning notebook 01 reproduced every figure and sample image pixel for pixel; rerunning notebook 02 reproduced every number.

---

## Key choices and why

| Choice | Why |
|---|---|
| COCO 2017 val, all 5,000 photos | Every caption has a real photo and labelled boxes, so we can check whether a generated image shows what the caption names |
| Captions cleaned for spacing only, wording never changed | Changing even a full stop changes the tokens CLIP reads, and so the image |
| Center crop to 512 × 512, PNG | Matches Stable Diffusion's output size and format; nothing is stretched |
| Frozen test split (seed 2026) | Every later setting is scored on the same captions |
| Separate dev split | Settings are chosen on dev and reported on test, so the test set is never tuned on |
| Second set of 2,000 real photos | FID is only meaningful beside the score real photos get against other real photos at the same N |
| Object labels with unclear words excluded and a crop check | "An orange cat" contains no fruit, and an object the crop removed cannot fairly count against the model |
| Captions encoded one at a time, no attention mask | Exactly how the pipeline encodes a prompt; batching changes the images |

---

## Results

### Milestone 1

| Result | Value |
|---|---|
| Images, captions after cleaning | 5,000 · 25,004 (3,832 spacing fixes, 10 duplicates dropped) |
| Frozen splits | 2,000 test · 500 dev · 2,000 real reference |
| Test captions with a scorable object | 1,692 of 2,000 (2,291 targets) |
| Image from cached embeddings vs raw text | mean pixel difference **0.000** |
| Caption → photo retrieval among 2,000 photos | R@1 46.8%, R@10 84.0% (chance 0.05%) |
| Samples | 5 captions × 3 seeds beside the real photo: objects usually appear; actions and details often do not |

### Text encoders without training (2,000 test captions)

| Representation | CKA with SD's CLIP | Paraphrase R@1 | Scene 10-NN accuracy |
|---|---|---|---|
| CLIP ViT-L/14 (EOS vector) | 0.79 | **34.0%** | **74.5%** |
| CLIP ViT-L/14 (token mean) | 1.00 | 21.4% | 70.0% |
| TF-IDF words (baseline) | 0.51 | 20.0% | 64.4% |
| T5-base | 0.60 | 15.5% | 68.5% |
| BERT-base | 0.53 | 13.0% | 65.8% |
| GPT-2 | 0.13 | 3.4% | 56.0% |

Only CLIP beats plain word counting at matching paraphrases; T5 is closer to CLIP than BERT is; GPT-2's mean-pooled vectors are nearly identical for every caption. For connectors: a CLIP connector as the control, BERT, and T5 as a stronger third candidate than GPT-2. The word-order / colour triplet test is **not** used for any decision: as built, it measures sentence structure rather than meaning (see the report, Section 18).

### Re-captioning (100 test photos, one seed; 4 blacked-out images left out)

| | Human | Short | Detailed (cut at 77) | Detailed, fitted |
|---|---|---|---|---|
| Photo objects named in the part SD reads (no person) | 54.7% | 62.1% | 71.5% | 74.7% |
| Similarity of image to real photo (CLIP-I) | 0.704 | 0.709 | 0.716 | 0.718 |
| Object recall (no person; detector on real photos: 91.5%) | 67.7% | 67.7% | 62.6% | 66.0% |
| Named-object recall: named, read, drawn (no person) | 88.5% | 88.0% | 76.3% | 72.0% |

**With this sample, longer captions gave no detectable gain**, and fitting the detailed captions to 77 tokens changed no measure detectably. Objects that long captions ask for are drawn less reliably (detailed vs human −12.4 points, 95% range −23.9 to −1.3). CLIPScore drops about 4 points just from scoring against a different caption, so models must always be compared against the same reference captions.

`results/recaptioning/named_read_missing.csv`: 79 objects in 73 images that a caption named, Stable Diffusion read, and the detector did not find (74 where the detector finds the object in the real photo). Columns: `image_id, caption_idx, source, seed, caption, missing_object, detector_found_in_real_photo, image_path`. `image_path` points to the author's Drive; the images can be regenerated from the caption and seed.

---

## Notes and known issues

- **dev_500 and real_ref_b_2000 currently share 353 photos** (neither shares any photo with the test split). The proposal describes three disjoint sets; the reference set will be redrawn to exclude dev photos so the files match. The test and dev splits do not change.
- The re-captioning study is a pilot (100 photos, one seed); 300 photos and two seeds would narrow the 95% ranges.
- Object labels come from a hand-built word table and have not been hand-checked.
- Large generated-image folders are not committed; they are reproducible from the notebooks.

---

## Environment

Google Colab, Tesla T4, Python 3.13, PyTorch 2.11 (CUDA 13), Diffusers 0.42.1, Transformers 5.22.1. Models: `stable-diffusion-v1-5/stable-diffusion-v1-5`, `openai/clip-vit-large-patch14`, `google-bert/bert-base-uncased`, `google-t5/t5-base`, `openai-community/gpt2`, Florence-2-large, torchvision Faster R-CNN ResNet-50 FPN v2.
