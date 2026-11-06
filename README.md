# Time Travel Inside a Diffusion Model
### Watching, rewinding and changing Stable Diffusion while it paints

**IE 7615 Deep Learning for AI · Northeastern University · Fall 2026 · Group 7**

Tanmayi Shurpali · Santhoshkumar Chandrasekar · Praniti Sunil Kale · Mukul Sridar Haladhi

## The question

A text-to-image model receives a whole sentence at once but paints it over many denoising steps, and it often leaves out an object the sentence names. We ask **when** each word gets painted, whether a word about to be dropped can be **seen coming**, and whether **going back** to an earlier step can save it without losing the rest of the picture.

📄 **Proposal:** [milestone-1/proposal/IE7615_P2_M2_Group7_proposal.pdf](milestone-1/proposal/IE7615_P2_M2_Group7_proposal.pdf)

## Repository layout

| Folder | Contents |
|---|---|
| `milestone-1/` | Setup & dataset: proposal, exp1, exp2, exp3 |
| `milestone-2/` | Measurement |
| `milestone-3/` | Intervention |
| `milestone-4/` | Final report |

## Milestone 1: Setup & Dataset

| Part | What we did |
|---|---|
| Dataset | MS-COCO 2017 validation: 5,000 photos, 25,004 captions after light cleaning (wording never edited) |
| Preprocessing | RGB, center crop, resize to 512 × 512; photos with a short side under 256 px left out |
| Splits | 2,000 test pairs · 500 development pairs · 2,000 real photos as the FID/KID reference · 200 stress prompts |
| Environment | PyTorch + Diffusers + CLIP, Stable Diffusion 1.5 |
| Text → embedding | CLIP ViT-L/14 turns each caption into a 77 × 768 matrix; images from the cached embeddings are pixel-identical to Stable Diffusion's own output |
| Sanity check | CLIP ranks the correct photo first for 35.5% of captions among 5,000 (chance 0.02%) |

## Team

| Member | Leads |
|---|---|
| Tanmayi Shurpali | image-quality evaluation (FID, KID, IS), guidance and step-count sweeps, calibrating the object judges |
| Santhoshkumar Chandrasekar | data and embedding pipeline, attention recorder and commit measure, rewind loop and Rewind viewer |
| Praniti Sunil Kale | evaluation sets, object labels and stress prompts, training-free encoder comparison |
| Mukul Sridar Haladhi | LoRA fine-tuning, encoder connector, painting objects from our first course project |

## Running the code

Notebooks run on a free Google Colab T4 GPU (*Runtime → Change runtime type → T4 GPU → Run all*) and download COCO and the model weights themselves. COCO data is not stored in this repo.
