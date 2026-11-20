# Milestone 2 generation protocol

Every Milestone 2 image is made through `m2_protocol.py`, so that four people on four machines produce numbers that sit in one table. The file fixes one anchor setting, one seed rule and one log format. Each sweep changes exactly one thing away from the anchor.

## Setup on Colab

From the `M2.zip` shared in the group chat: upload it with Colab's Files panel, then

```python
!pip install -q diffusers==0.42.1
!unzip -q -o M2.zip
import sys; sys.path.insert(0, "M2/protocol")
import m2_protocol as m2
```

From GitHub instead:

```python
!pip install -q diffusers==0.42.1
!wget -q https://raw.githubusercontent.com/tanmayi123/from-words-to-worlds/main/milestone-2/protocol/m2_protocol.py
import m2_protocol as m2
```

Then, either way:

```python
pipe = m2.load_pipeline()      # SD 1.5, half precision, safety checker on
m2.smoke_check(pipe)           # the Wednesday 6 PM check
```

`smoke_check` must print these two lines on every machine:

```
first dev caption_idx 3669, noise sum 36.4438: OK
anchor timesteps [958, 925, 892] ... [34, 1] (30): OK
```

It then paints the anchor for the first 10 dev captions into `m2_smoke/`. The pictures should look the same on every machine; tiny pixel differences between GPU types are normal.

`m2.load_split("dev_500")` reads the `splits` folder that sits beside the `protocol` folder (as in the zip), or a local `milestone-2/splits/`, and otherwise downloads it from `milestone-2/splits/` on GitHub. Please pin diffusers to 0.42.1: scheduler code changes between versions.

## The anchor and the seed rule

| | |
|---|---|
| Model | Stable Diffusion 1.5 (`stable-diffusion-v1-5/stable-diffusion-v1-5`), float16, 512 x 512 |
| Text | the caption, encoded by the pipeline's own CLIP, one prompt at a time |
| Sampler | DDIM, eta 0 |
| Steps | 30 |
| Guidance | 7.5 |
| Spacing | leading (SD's default): timesteps 958, 925, ..., 34, 1 |
| Safety checker | on; a flagged image is logged and dropped from comparisons in pairs |
| Seed | `100000 * seed_id + caption_idx` (stress prompts: their row number 0 to 199 in place of caption_idx) |
| Starting noise | drawn on the CPU in float32 from that seed, passed to the pipeline as `latents` |

Drawing the noise on the CPU is what makes a laptop and a Colab T4 start from the same noise; noise drawn on a GPU is not guaranteed to match across GPU models. Sweeps use seed_id 0. Diversity pairs use seed_id 1 on the first 100 dev captions.

Why DDIM and not PNDM: the word-timing and rewind code needs one network call per step and an exact restart from any step. PNDM spends 31 calls on 30 steps (it evaluates timestep 925 twice). PNDM is still in the sampler sweep, which links back to exp4's numbers.

## Your runs

`run_config` makes one image per prompt and seed_id, saves it, and logs it. It resumes: images already on disk are skipped, so a dropped Colab session just picks up where it stopped. Keep the log folder on Drive.

**Tanmayi** (guidance and step count, dev_500):

```python
log = m2.RunLog("/content/drive/MyDrive/m2", owner="Tanmayi", axis="guidance")
for g in (3, 5, 7.5, 10, 12):
    m2.run_config(pipe, m2.GenConfig(guidance=g), "dev_500", log)
    m2.run_config(pipe, m2.GenConfig(guidance=g), "dev_500", log, seed_ids=(1,), first=100)   # diversity pairs
steps = m2.RunLog("/content/drive/MyDrive/m2", owner="Tanmayi", axis="steps")
for n in (10, 20, 50):
    m2.run_config(pipe, m2.GenConfig(steps=n), "dev_500", steps, first=200)
```

**Mukul** (noise schedule on DPM-Solver++ 2M, dev_500):

```python
log = m2.RunLog("/content/drive/MyDrive/m2", owner="Mukul", axis="noise schedule")
for s in ("leading", "trailing", "karras", "ays"):
    m2.run_config(pipe, m2.GenConfig("dpmpp2m", s, 10), "dev_500", log)
    m2.run_config(pipe, m2.GenConfig("dpmpp2m", s, 10), "dev_500", log, seed_ids=(1,), first=100)
    m2.run_config(pipe, m2.GenConfig("dpmpp2m", s, 30), "dev_500", log, first=200)
```

**Praniti** (sampler x guidance, 200 stress prompts):

```python
log = m2.RunLog("/content/drive/MyDrive/m2", owner="Praniti", axis="sampler x guidance")
for sampler in ("ddim", "pndm", "dpmpp2m"):
    for g in (3, 7.5, 12):
        m2.run_config(pipe, m2.GenConfig(sampler, "leading", 30, g), "stress_prompts", log)
m2.run_config(pipe, m2.GenConfig("pndm"), "stress_prompts", log, seed_ids=(1, 2))           # the 3-seed benchmark
```

**Santhosh** runs the anchor on eval_2000 and the recorded DDIM runs on the laptop with exp1's `Painter`, which records cross-attention. A test in exp1 checks that `Painter` paints exactly the same picture as this file for the same setting and seed.

## What gets written

```
<log folder>/runs.csv      one row per setting and split
<log folder>/images.csv    one row per image
<log folder>/images/<split>/<config_id>/<key>_s<seed_id>.png
```

`config_id` is `<sampler>-<schedule>-s<steps>-g<guidance>`; the anchor is `ddim-leading-s30-g7.5`. `runs.csv` records the sampler, schedule, steps, network calls per image, guidance, seed_ids, seed rule, dtype, GPU, diffusers and torch versions, the protocol version and the file's sha256, seconds per image, image and flagged counts, and start and finish times. `images.csv` holds config_id, split, key (caption_idx or prompt_id), seed_id, seed, file, flagged and seconds. Scores are joined onto `images.csv` by config_id, split, key and seed_id.

At the end, the two CSV files go into your `milestone-2/expN/` folder. The images stay on Drive; only sample grids go on GitHub.

## Runs made before this protocol

They count. Log each one with its real settings and its own seed rule; it just isn't paired image for image with anyone else's.

```python
log.add_run(m2.GenConfig("pndm", "leading", 30, 7.5), "stress_prompts", n_prompts=200, seed_ids=(0, 1, 2),
            seed_rule="own: seeds 0, 1, 2 for every prompt, CUDA generator",
            n_images=600, n_flagged=0, sec_per_image=5.3)
```

If a DPM-Solver++ run didn't set the spacing, it ran diffusers' default, which is linspace: log it as `GenConfig("dpmpp2m", "linspace", 30, g)`.

## What each setting actually visits

The noise schedule is the list of noise levels the sampler visits; the sampler is the update rule between them. `python m2_protocol.py` prints the full timestep list for every planned setting. In short:

| Setting | Network calls | Timesteps visited |
|---|---|---|
| ddim-leading-s30 (anchor) | 30 | 958, 925, ..., 34, 1 |
| ddim-linspace-s30 | 30 | 999, 965, ..., 34, 0 |
| ddim-trailing-s30 | 30 | 999, 966, ..., 66, 32 |
| pndm-leading-s30 | 31 | 958, 925, 925, 892, ..., 34, 1 |
| dpmpp2m-leading-s30 | 30 | 961, 929, ..., 65, 33 |
| dpmpp2m-leading-s10 | 10 | 901, 811, ..., 181, 91 |
| dpmpp2m-trailing-s10 | 10 | 999, 899, ..., 199, 99 |
| dpmpp2m-karras-s10 | 10 | 999, 916, 815, 687, 523, 327, 146, 41, 7, 0 |
| dpmpp2m-ays-s10 | 10 | 999, 850, 736, 645, 545, 455, 343, 233, 124, 24 |
| dpmpp2m-ays-s30 | 30 | 999, 953, 903, ..., 24, 8, 2 |

Three things worth knowing:

- DPM-Solver++ finishes every schedule with one more update to zero noise, and diffusers spaces its "leading" list over one extra point, so it starts at 961 where DDIM starts at 958. The sampler sweep therefore changes the update rule and, slightly, the levels visited; the table above says exactly how much.
- Leading spacing never starts from pure noise (901 at 10 steps), the flaw Lin et al. (WACV 2024) point out; trailing starts at 999. This is why the 10-step arm should separate the schedules most.
- Align Your Steps (Sabour et al., 2024) at 10 steps is NVIDIA's published list for SD 1.5. At 30 steps the 11 published noise levels are interpolated log-linearly, NVIDIA's own recipe, and each level is mapped to the nearest training timestep.

Two schedules are left out on purpose: zero-terminal-SNR rescaling needs a v-prediction model (SD 1.5 predicts noise), and swapping the training beta schedule at inference no longer matches the trained model.
