# Mukul Sridar Haladhi — Fine-tuning and connector experiments (Milestone 1)

Report: `Mukul_FineTuning_and_Connector_Report.pdf`

## 1. LoRA fine-tuning (`1_lora_finetuning/`)
SD v1.5 UNet fine-tuned with LoRA (rank 8, all attention projections, 1.59 M of 861 M params) on ~3,000 COCO
train2017 pairs; evaluated on 40 held-out val2017 captions with paired seeds.

| Checkpoint | CLIPScore | Δ vs base [95% CI] | Win rate | Val loss |
|---|---|---|---|---|
| base | 25.52 | — | — | 0.15120 |
| step 150 | 25.96 | +0.44 [−0.11, 1.01] | 50% | 0.15107 |
| step 300 | 26.28 | +0.75 [0.14, 1.38] | 62% | 0.15106 |
| step 450 | 26.29 | +0.77 [0.16, 1.40] | 65% | 0.15103 |
| step 600 | 26.37 | +0.84 [0.16, 1.53] | 72% | 0.15102 |

13.8 min on a T4, peak 2.59 GB. `results/lora_r8_pilot/` has the logs, figures, samples and final LoRA weights.

## 2. BERT vs CLIP through a trained connector (`2_connector_bert_vs_clip/`)
CLIP and BERT each feed the UNet through an identical connector (LayerNorm → MLP → LayerNorm, 1.58 M params,
random init), with and without UNet LoRA; same data, steps, prompts and seeds as run 1.

| Run | CLIPScore | Δ vs base [95% CI] |
|---|---|---|
| base (CLIP direct) | 25.50 | — |
| clip_conn | 14.35 | −11.20 [−13.11, −9.38] |
| bert_conn | 12.43 | −13.07 [−14.71, −11.48] |
| clip_conn_lora | 13.69 | −11.81 [−13.65, −10.01] |
| bert_conn_lora | 11.83 | −13.73 [−15.16, −12.28] |

BERT vs CLIP through the same connector: −1.95 [−3.47, −0.50] (connector only), −2.16 [−3.96, −0.41] (with LoRA).
A from-scratch connector at 600 steps loses most of the prompt, which motivates the two-stage connector training.

## Reproduce
Note for rerunning the LoRA notebook: newer `transformers` changed `get_image_features`, so in the evaluation cell replace the two `clip.get_image_features` / `clip.get_text_features` lines with `clip.visual_projection(clip.vision_model(pixel_values=x).pooler_output)` and `clip.text_projection(clip.text_model(input_ids=tk["input_ids"], attention_mask=tk["attention_mask"]).pooler_output)` (then normalize as before).

Open either notebook in Colab or Kaggle with a T4 GPU and run all (data downloads automatically).
