# milestone-2 / exp3: The stress-prompt benchmark (which skills does Stable Diffusion 1.5 get right?)

**Author:** Praniti Sunil Kale · IE 7615 Deep Learning for AI · Northeastern University

Part of [From Words to Worlds](../../README.md). Continues [`milestone-1/exp3`](../../milestone-1/exp3), where the 200 stress prompts, the evaluation splits and the training-free encoder comparison were built.

Milestone 1 built 200 **stress prompts**: short prompts that each test one skill text-to-image models are known to get wrong, stored with their expected answer. This experiment **runs that exam** on Stable Diffusion 1.5, scores every image automatically, and repeats it across guidance scales and samplers. It also finishes two Milestone 1 loose ends: a fully disjoint real reference set, and the encoder comparison with 95% intervals and a fixed word-order test.

| Milestone 2 requirement | Here |
|---|---|
| Integrate a text encoder with a diffusion model | Stable Diffusion 1.5 conditioned by its CLIP ViT-L/14 text encoder |
| Run baseline conditional generation | 200 stress prompts × 3 seeds at the default settings |
| Tune classifier-free guidance + noise schedule (≥ 3 values each) | Guidance {3, 7.5, 12} × sampler {DDIM, PNDM, DPM++} |
| Record the first set of generated images | 2,200 images, each in `results/generation_log.csv` |
| Generation log, early samples | `results/generation_log.csv`, `samples/` (10 images with prompts) |

---

## Folder contents

```
milestone-2/exp3/
├── README.md
├── notebooks/
│   └── Praniti_M2_StressPrompt_Benchmark.ipynb    ← the full run, with outputs
├── results/
│   ├── generation_log.csv                         ← every image: prompt, family, seed, guidance, scheduler, steps, seconds, safety flag
│   ├── scores.csv                                 ← every image's verdict from the judges
│   ├── baseline_by_family.csv                     ← pass rate per skill at the baseline, with 95% intervals
│   ├── sweep_by_family.csv, guidance_effect.csv   ← the guidance × sampler grid
│   ├── seconds_per_image.csv                      ← time per image per sampler
│   ├── encoder_comparison_with_intervals.csv      ← Milestone 1 encoder comparison, with intervals and the fixed word-order test
│   ├── stress_prompt_object_labels.csv            ← one row per object named in each stress prompt (prompt_id, category, word, word_start)
│   ├── splits/real_ref_2000_disjoint.csv          ← kept for the record; the group's reference set is milestone-2/splits/ref_2000_v2.csv
│   ├── observations.txt, m2_exp3_summary.json, config.json
│   └── figures/
│       ├── fig1_object_even_there.png             ← correct / drawn but wrong / object not drawn, per skill
│       ├── fig2_failure_examples.png              ← the two typical failures
│       └── fig3_sweep_heatmaps.png                ← pass rate by guidance and sampler
└── samples/                                       ← one passing and one failing image per skill + samples.csv
```

The 2,200 generated images are not in the repo (see *Not in the repo* below).

---

## How to run

Google Colab, T4 GPU, **Run all**. Results are saved to `MyDrive/w2w_praniti/results/m2_exp3/` and images to `MyDrive/w2w_praniti/runs/m2_stress/`; a rerun skips images already on Drive.

| Part | Images | Time on a T4 |
|---|---|---|
| Baseline | 600 | ~50 min |
| Sweep (8 new settings) | 1,600 | ~2 h 15 min |
| Scoring + encoder follow-up | — | ~15 min |

## Settings

| Setting | Value |
|---|---|
| Model | `stable-diffusion-v1-5/stable-diffusion-v1-5`, fp16, 30 steps |
| Baseline | PNDM, guidance 7.5, seeds 0, 1, 2 |
| Grid | DDIM / PNDM / DPM++ × guidance 3 / 7.5 / 12, seed 0 (PNDM 7.5 reuses the baseline's seed-0 images) |
| Schedulers | Stable Diffusion's own scheduler config, switched with `from_config`. DDIM and PNDM use Stable Diffusion's leading spacing (first step at t = 958); **DPM++ is DPMSolverMultistep (2M) with diffusers' default linspace spacing (first step at t = 999)**, logged as `dpmpp2m, linspace spacing` |
| Seeds | `torch.Generator("cuda").manual_seed(seed)`, i.e. the noise is drawn on the GPU |
| Hardware | Google Colab, Tesla T4 |
| Software | PyTorch 2.14.1 (CUDA 13); diffusers not pinned and its version not recorded by the notebook (latest release on the day of the run) |

These runs were made before the group's shared protocol in `milestone-2/protocol` (DDIM, 30 steps, guidance 7.5, seed = 100000 × seed_id + caption_idx drawn on the CPU, diffusers 0.42.1), so they use their own seed rule. All settings within this experiment share prompts and seeds, so they are comparable with each other.

## The judges

| Skill (40 prompts each) | Example | Pass when |
|---|---|---|
| Multi-object | "a horse, a bus and a bottle" | all three objects are detected |
| Counting | "four boats" | the detected count equals the number asked for |
| Colour binding | "a yellow book and a white truck" | both objects are detected and CLIP reads the right colour on each crop |
| Spatial | "a bowl to the left of a zebra" | both objects are detected and their boxes are in the right relation |
| Word order | "a bear following a sheep" / "a sheep following a bear" | the image matches its own sentence better than its swapped twin (CLIP) |

Objects are found by torchvision's Faster R-CNN (ResNet-50 FPN v2), confidence ≥ 0.5; colours and word order are judged with CLIP ViT-L/14. Unlike the rest of the project's evaluation, which uses two object judges, the stress scoring uses Faster R-CNN alone, and the CLIP colour reading has not been checked against people. The scoring rules are checked on hand-made detections before use. Images blacked out by the safety checker (6 of 2,200) are left out.

---

## Results

### Baseline (PNDM, guidance 7.5, 3 seeds; 95% intervals over prompts)

| Skill | Pass rate | Detail |
|---|---|---|
| Counting | **29%** (21–38%) | within one of the right number 64%; the object is drawn at all 98% |
| Spatial | **21%** (13–29%) | both objects drawn in only 44% of images |
| Multi-object | **13%** (7–20%) | 59% of the named objects drawn on average |
| Colour binding | **3%** (1–7%) | both objects drawn in only 30 of 120 images; of those, colours right in 4, swapped in 6, other colours in 20 |
| Word order | — | **can't be measured with this judge**: the CLIP judge sits at 52% (39–65%), a coin flip, because CLIP is itself largely blind to word order. This is a limit of the judge, not a model result. Both animals are drawn in only 25% of images |

Colour binding fails mostly because one of the two objects is not drawn at all; even when both are, the colours are swapped (6 images) slightly more often than right (4), a comparison that rests on only 30 images. Typical failures: "a yellow book and a white truck" gives no book and a **yellow** truck (the colour moves to the wrong object); "a bowl to the left of a zebra" gives no zebra but a **zebra-patterned** bowl (the animal becomes a texture).

### Guidance × sampler grid (seed 0)

**No sampler or guidance setting detectably changes any skill.** Raising guidance from 3 to 12 (averaged over the three samplers) changes every skill by less than its 95% interval:

| Skill | Guidance 3 | Guidance 12 | Change (95% interval) |
|---|---|---|---|
| Multi-object | 7.5% | 4.2% | −3% (−10% to +2.5%) |
| Counting | 20.8% | 25.0% | +4% (−8% to +17%) |
| Colour binding | 4.2% | 4.2% | 0% (−3% to +3%) |
| Spatial | 20.8% | 16.7% | −4% (−12% to +3%) |
| Word order | 55.8% | 56.7% | +1% (−8% to +10%) |

Each grid cell uses one seed and 40 prompts, so differences between individual cells are within noise and are not read as "best settings".

| Sampler | Seconds per image (30 steps, T4) |
|---|---|
| DPM++ | 5.04 |
| DDIM | 5.09 |
| PNDM | 5.23 |

Guidance does not change the work per step, so it does not change the time.

**Conclusion.** Stable Diffusion 1.5 fails most compositional skills, mostly because an object is never drawn, and none of the tested inference settings detectably changes that. With 40 prompts per skill, changes smaller than about 10 points would go undetected. Sampler and guidance do not fix it. Milestone 3 tests three other routes: rewinding the painting, LoRA, and the encoder connector.

### Milestone 1 follow-up: the encoder comparison with 95% intervals (2,000 test captions)

| Representation | Paraphrase R@1 | Scene 10-NN | Word order (fixed) | Colour (fixed) |
|---|---|---|---|---|
| CLIP ViT-L/14 sentence vector (EOS) | **34.0%** (31.9–36.0) | **74.5%** (72.6–76.3) | 40% (20–60) | **72.5%** (57–85) |
| CLIP ViT-L/14 token mean | 21.4% (19.6–23.3) | 70.0% (67.9–72.0) | 100%* | 100%* |
| T5-base | 15.4% (13.9–17.1) | 68.4% (66.4–70.5) | 20% (5–40) | 2.5% (0–7) |
| BERT-base | 13.0% (11.5–14.5) | 65.8% (63.7–67.8) | 5% (0–15) | 0% |
| GPT-2 | 3.4% (2.6–4.2) | 56.0% (53.8–58.2) | 30% (10–50) | 25% (12–40) |
| TF-IDF words (baseline) | 19.8% (18.1–21.5) | 64.3% (62.2–66.4) | 0% | 0% |

\* An artifact, not a result: CLIP reads left to right, and the synonym version shares the start of the sentence with the original while the swapped version differs from the second word, so mean-pooled CLIP states favour it automatically. The sentence vector (EOS) is the fair CLIP row.

Figures for the report: `results/figures/fig1_object_even_there.png` (correct / drawn but wrong / not drawn, per skill), `fig2_failure_examples.png` (the two failures above) and `fig3_sweep_heatmaps.png`.

In the fixed test, an encoder passes if a same-meaning rewording with one synonym ("following" → "trailing", "and" → "along with") is closer to the original than the meaning-changing swap that uses exactly the same words. Word counting scores 0% by construction.

**What changes from Milestone 1:** only CLIP's **sentence vector** clearly beats word counting at matching paraphrases (CLIP's token mean overlaps it); T5 is only **slightly** ahead of BERT (the intervals nearly touch); and only CLIP's sentence vector shows real sensitivity to colour binding. The connector plan: a CLIP connector as the control, BERT, and **T5 instead of GPT-2** (a change from the proposal). Reason: without training, T5 is closer to CLIP than GPT-2 (linear CKA 0.60 vs 0.13) and better at matching paraphrases (15.4% vs 3.4% R@1), GPT-2's mean-pooled vectors are nearly identical for every caption, and Imagen, cited in the proposal, found a T5 text encoder works well for text-to-image generation.

### Real reference set

`results/splits/real_ref_2000_disjoint.csv`: 2,000 eligible COCO val2017 photos drawn (seed 2028) from outside both `eval_2000` and `dev_500` (the Milestone 1 reference set shared 353 photos with `dev_500`). It still shares 339 photos with the 500-caption dev set used in the `milestone-1/exp1` pilots, so **the project uses `ref_2000_v2`** (the same drawing code, seed 2031, disjoint from `eval_2000` and both dev sets); 1,624 of its photos are also in this set. This file is kept for the record only.

### Object labels for the stress prompts

`results/stress_prompt_object_labels.csv`: one row per object named in each of the 200 stress prompts (400 rows): `prompt_id, family, prompt, category, word, word_start`, where `category` is the COCO class and `word_start` is the word's index in `prompt.split()`. For the Milestone 3 early-warning runs on the multi-object, counting and spatial prompts.

---

## Not in the repo

The 2,200 generated images (about 1 GB), one folder per setting: **[Google Drive](https://drive.google.com/drive/folders/1erEln6NRRof3ZmoD8YRmzz1tVs1VXohK)** (anyone with the link can view). The notebook regenerates them from the prompts and seeds in `results/generation_log.csv`.

---

## Limitations

- The judges (detector and CLIP) have not yet been checked against human labels; detector misses lower every pass rate.
- Word order cannot be judged reliably with CLIP; a vision-language yes/no judge or VQAScore would be needed.
- The grid uses one seed per setting; only large effects would be detectable.
- The stress prompts are templated, and some combinations are unusual scenes.
- Word order is reported as not measurable with the CLIP judge, not as a model failure.
- These runs predate the shared protocol (seed rule, default sampler, pinned diffusers version), so their images are not bit-identical to protocol runs.
