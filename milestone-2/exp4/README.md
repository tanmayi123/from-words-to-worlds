# Discriminative Project Deep Learning: IE7615 Milestone 1 (Group 7)

**Task:** closed-set image classification of 73 student-registered objects (OBJ001 to OBJ073) from photos.
We compare CNNs trained from scratch with ImageNet-pretrained CNNs on the TA's final merged dataset.

**Group members:** Mukul Sridar Haladhi, Praniti Sunil Kale, Santhoshkumar Chandrasekar, Tanmayi Shurpali

**Selected model:** ConvNeXt-Tiny, fully fine-tuned. Test accuracy 99.23% (95% CI 98.7 to 99.7), macro F1 99.23%, 8 errors on 1,042 unseen test photos. It was chosen using validation data only. The full analysis is in `report/`.

---

## Repository structure

```
Discriminative-Project-Deep-Learning/
├── README.md
├── requirements.txt
├── final_pipeline/                                   # the run that produces every number in the report
│   ├── Group7_IE7615_Milestone1_final_notebook.ipynb
│   └── outputs/
│       ├── model_comparison.csv                      # main results table
│       ├── robustness_test.csv, robustness_val.csv   # accuracy under 4 perturbations
│       ├── significance_mcnemar.csv, mcnemar_all_pairs.csv
│       ├── selection_validation.csv                  # model selection on validation data
│       ├── per_class_selected.csv                    # precision/recall/F1 per Object ID
│       ├── misclassified_all_models.csv, hardest_test_images.csv
│       ├── background_shortcut.csv, label_audit_suspects.csv
│       ├── class_counts.csv, removed_duplicates.csv, train/val/test_split.csv, classes.json
│       ├── *_history.csv                             # per-epoch training history of each model
│       ├── *_probs.npz                               # saved validation/test predictions of each model
│       ├── results_summary.md / .pdf, run_log.txt
│       └── figures/                                  # curves, robustness, confusion matrix, Grad-CAM, misclassified grids
├── experiments/                                      # exploratory experiments used to shortlist models
│   ├── experiment_1/
│   │   ├── experiment_1_models.ipynb                 # custom CNN, ConvNeXt-Tiny linear probe, ResNet-50,
│   │   │                                             # EfficientNet-B0, ConvNeXt-Tiny, ConvNeXt-Tiny basic-aug ablation
│   │   └── outputs/                                  # per-model run_config, train.log, history, metrics, test_probs;
│   │                                                 # session logs, data audit CSVs, figures
│   ├── experiment_2/
│   │   ├── experiment_2_models.ipynb                 # custom CNN, MobileNetV3-Large, DenseNet-121, EfficientNetV2-S
│   │   └── outputs/                                  # per-model history and test predictions, comparison table,
│   │                                                 # results.json, splits, validation curves
│   └── experiment_3/
│       └── experiment_3_models.py                    # AlexNet, ResNet-101, GoogLeNet, MNASNet-1.0,
│                                                     # ShuffleNetV2 x1.5, SqueezeNet 1.1
└── report/
    └── Final_Group7_IE7615_Milestone1_Report.pdf
```

## Models tested

| Where | Models |
|---|---|
| `final_pipeline/` (results in the report) | **A2** custom CNN (scratch), **C** ResNet-50, **D** EfficientNet-B0, **E** ConvNeXt-Tiny (selected), **G** DenseNet-121 |
| `experiments/experiment_1/` | custom CNN, ConvNeXt-Tiny linear probe, ResNet-50, EfficientNet-B0, ConvNeXt-Tiny, ConvNeXt-Tiny with basic augmentation |
| `experiments/experiment_2/` | custom CNN, MobileNetV3-Large, DenseNet-121, EfficientNetV2-S |
| `experiments/experiment_3/` | AlexNet, ResNet-101, GoogLeNet, MNASNet-1.0, ShuffleNetV2 x1.5, SqueezeNet 1.1 |

The exploratory experiments were used to shortlist architectures. Only the final pipeline trains every model on one shared cleaned dataset and split, so only its numbers can be compared with each other (report, Section 5.1 and Appendix A).

## Model weights

The trained weights (`.pt`) are larger than GitHub's 100 MB file limit, so they are attached to this repository's **[Releases](../../releases)** page instead of being stored in the code.

## Final pipeline in brief

1. **Data audit:** skips junk files, verifies every image, and checks each filename against its Object ID. The label always comes from the folder name.
2. **Cleaning:** removes 385 background-only photos (`_bg_` / `BG###` tags), then 16 exact duplicates by MD5. Images are 224x224 RGB. 6,942 images remain.
3. **Split:** stratified 70/15/15 with seed 42 (4,859 / 1,041 / 1,042 images).
4. **Quality checks:**
   - out-of-fold label audit (97.9%);
   - near-duplicate check, which defines a hard test subset of 943 images.
5. **Training:** one recipe for every model:
   - robust augmentation;
   - AdamW with separate head and backbone learning rates;
   - warm-up then cosine schedule;
   - label smoothing and class weights;
   - mixed precision;
   - early stopping on validation macro F1;
   - resume-safe checkpoints.
6. **Evaluation:**
   - bootstrap confidence intervals;
   - hard-subset accuracy, top-5 accuracy and TTA;
   - robustness tests;
   - McNemar tests;
   - confusion matrix and misclassified images for every model;
   - Grad-CAM and a background-shortcut test.
7. **Selection, using validation data only:**
   1. keep models within 0.5 points of the best macro F1;
   2. pick the highest perturbed-validation accuracy;
   3. break ties with the fewest parameters.

## How to run

**Final pipeline (Google Colab, T4 GPU, about 1.5 to 2 hours):**
1. Put the TA dataset zip in Google Drive as `MyDrive/IE7615DeepLearning.zip`.
2. Open `final_pipeline/Group7_IE7615_Milestone1_final_notebook.ipynb` in Colab.
3. Choose Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**, and allow Drive access.
4. Outputs are written to `MyDrive/IE7615_project_v3/`. If Colab disconnects, run all again: finished models are skipped and an interrupted model resumes from its last epoch.

**Experiments:**
- `experiment_1`: runs on a local machine with a GPU. Set `PROJECT` / `LOCAL_DATA` in the config cell to the dataset folder.
- `experiment_2`: runs in Google Colab, with the dataset zip in the same Drive location as the final pipeline.
- `experiment_3`: runs locally (CUDA, Apple MPS or CPU). Set `PROJECT` at the top of the file, then run `python experiment_3_models.py`.

Install the dependencies with `pip install -r requirements.txt`. Colab already has them installed.

## Not included

- **The dataset.** It is the course's student-collected data, provided by the TAs, so we do not redistribute it.
- **Privacy.** One dataset image (in OBJ054) is a phone screenshot that shows personal contact details. It is blurred in every figure and output in this repository, and the figures that showed it unblurred were left out.
