# Time Travel Inside a Diffusion Model
### Watching, Rewinding and Changing Stable Diffusion While It Paints

**IE 7615 Deep Learning for AI · Northeastern University, Boston, MA · Group 7**

Tanmayi Shurpali · Santhoshkumar Chandrasekar · Praniti Sunil Kale · Mukul Sridar Haladhi

📄 **Proposal:** [milestone-1/proposal/IE7615_P2_M1_Group7_proposal.pdf](milestone-1/proposal/IE7615_P2_M1_Group7_proposal.pdf)

---

## I. Question

A text-to-image model receives a whole sentence at once but paints it over many denoising steps, and it often leaves out an object the sentence names. We ask:

1. **When** does each word get painted?
2. Can a word that is about to be dropped be **seen coming**?
3. Can **going back** to an earlier step save it without losing the rest of the picture?

Earlier work shows that cross-attention maps fix an image's layout in the first steps, aggregates them into per-word heatmaps, and repairs neglected objects by boosting their attention or by resampling. To our knowledge, none of it measures, word by word and across a dataset, the step at which a word's position is settled or an object is lost, or what a repair costs the rest of the picture. We measure both on 2,000 COCO captions, test whether they predict which objects appear, and test repairs by rewinding the generation.

## II. Dataset

| | |
|---|---|
| Source | MS-COCO 2017 validation: 5,000 photos, 25,004 captions after cleaning; labels for 80 object categories |
| Cleaning | whitespace repaired in 3,832 captions, 10 duplicates dropped, wording never edited; every caption fits CLIP's 77-token window |
| Images | converted to RGB (10 were grayscale), center-cropped and resized to 512 × 512; the crop keeps 72% of an average frame; 42 photos with a short side under 256 px left out |
| Splits (fixed seeds, disjoint) | 2,000 test pairs · 500 development pairs for choosing settings · 2,000 real photos as the FID/KID reference |
| Stress prompts | 200 (word order, counting, colour binding, spatial, multi-object) |
| Training data | a cleaned 3,000-image subset of the 118,287-image train2017 split for LoRA fine-tuning; train2017 captions for the connector's first stage |

## III. Model

- **Backbone:** pretrained **Stable Diffusion 1.5**, frozen except for LoRA adapters and the connector.
- **Text encoder:** **CLIP ViT-L/14** turns each prompt into a 77 × 768 matrix. The UNet denoises a 64 × 64 × 4 latent over 20–50 steps, reading that matrix through cross-attention, and a VAE decodes the result.
- **Our own sampling loop** controls classifier-free guidance, the DDPM and DDIM samplers, and attention recording.
- **Word timing:** each word's cross-attention heatmap is recorded at every step. A word **commits** at the first step after which its heatmap stays correlated (Pearson r ≥ 0.8) with its final layout.
- **Encoder comparison:** BERT and GPT-2 reach the UNet through a connector with 77 learned query slots. It is trained first on captions to reproduce CLIP's states, then on image–caption pairs, with an identical CLIP connector as the control.

## IV. Evaluation

- **Image quality:** FID, KID and Inception Score, each next to the score real photos get at the same sample size. Guidance and step-count sweeps trace the fidelity–diversity trade-off.
- **Text alignment:** CLIPScore, object recall from two detectors of different design, and VQAScore for actions and relations where compute allows.
- **Rigour:** settings are chosen on the development pairs only. Comparisons share captions and seeds, use three seeds for every headline comparison, and report bootstrap confidence intervals.

## Repository layout

| Folder | Contents | Lead |
|---|---|---|
| `milestone-1/proposal/` | project proposal | all |
| `milestone-1/exp1/` | data and text → embedding pipeline, attention recorder, commit measure, rewind loop and pilots | Santhoshkumar |
| `milestone-1/exp2/` | LoRA fine-tuning; CLIP vs BERT through a connector | Mukul |
| `milestone-1/exp3/` | evaluation sets, object labels, stress prompts, training-free encoder comparison, re-captioning study | Praniti |
| `milestone-1/exp4/` | combined evaluation: FID/KID/IS against the real-photo floor, object judges calibrated on real photos, word timing under LoRA; initial training-subset run | Tanmayi |
| `milestone-2/` | Measurement | |
| `milestone-3/` | Intervention | |
| `milestone-4/` | Final report | |

Each experiment folder has its own README with what it tests, how to run it and its results.

## Milestone 1 results

| Check | Result |
|---|---|
| Text → embedding | all 25,004 captions encoded and cached; images from the cache are pixel-identical to Stable Diffusion's own output |
| Retrieval sanity check | CLIP ranks the correct photo first among all 5,000 for 35.5% of captions, top five for 59.9% (chance 0.02%) |
| Generation | 30 steps, guidance 7.5, 3.0 s per image on a laptop RTX 5070; named objects usually appear, actions and relations mostly don't |
| Development pilot | of 378 named objects, the 36 both detectors miss commit later than the 342 they find (median step 13 vs 5, AUROC 0.82) |
| Rewind pilot | for 34 failures, a new seed brings the object back 85% of the time but never keeps the picture; going back to step 12 brings it back 41% of the time and keeps the picture for 35% |
| Image quality (exp4) | FID vs real reference: real photos 77.1, base 91.1, LoRA 88.4; LoRA does not measurably change KID |
| Object judges (exp4) | calibrated on real photos: the two detectors find 98.4% of named objects there, 93.8% in base SD images |
| Other directions | LoRA gave a small CLIPScore gain; BERT trails CLIP without training; longer recaptions gave no detectable gain in objects drawn |

## V. Plan

| Milestone | Focus |
|---|---|
| **2: Measurement** | commit steps on all 2,000 test captions; image-quality baseline; sweeps of guidance, step count and sampler; train the connector's first stage so BERT and GPT-2 can drive the model |
| **3: Intervention** | rewind a painting to find when a missing object is lost for good, and whether going back saves it better than repainting from a new seed; the Rewind viewer; text encoders through the connector; LoRA to paint objects from our first course project, judged by that project's classifier |
| **4: Final report** | full evaluation; ethics (biases inherited from COCO and the model's training data, and misuse of tools that paint specific real objects) |

## VI. Team

Each member leads one area, and all four review every result and write the reports.

| Member | Leads |
|---|---|
| **Tanmayi Shurpali** | image-quality evaluation (FID, KID, Inception Score against the real-photo floor), the guidance and step-count sweeps, and calibrating the object judges on real photos |
| **Santhoshkumar Chandrasekar** | the data and text-to-embedding pipeline, the attention recorder and commit measure, and the rewind loop with its experiments and the Rewind viewer |
| **Praniti Sunil Kale** | the evaluation sets, object labels and stress prompts, and the training-free encoder comparison |
| **Mukul Sridar Haladhi** | LoRA fine-tuning, the encoder connector, and painting objects from our first course project |

## Running the code

Notebooks run on a free Google Colab T4 GPU (*Runtime → Change runtime type → T4 GPU → Run all*) and download COCO and the model weights themselves. COCO data is not stored in this repo: annotations are CC BY 4.0, and images keep their Flickr licences and are used for research only.
