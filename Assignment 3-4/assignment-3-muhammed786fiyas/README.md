# DA5402 — Assignment 3 & 4: Cattle Muzzle Biometric Identification
**Student:** Muhammed Fiyas  
**Roll No:** DA25M018  
**Course:** DA5402 — MLOps 

---

## Project Overview

This project implements a reproducible MLOps pipeline for **cattle muzzle biometric identification** using DVC. Three classification models are trained and evaluated on a dataset of 20 cattle breeds.

The pipeline covers data preparation, feature extraction, model training (SVM, MLP, CNN), and evaluation — all tracked and reproducible via DVC.

---

## Dataset

| Property | Value |
|----------|-------|
| Total breeds | 20 (out of 28 raw folders) |
| Train images | 961 |
| Val images | 113 |
| Test images | 113 |
| Image size | 224 × 224 |
| Augmented train (v2) | 1922 |

---

## Pipeline

```
data/raw
   ↓
prepare      → resize to 224×224, split 80/10/10
   ↓
transform    → HOG feature extraction + v2 augmented dataset
   ↙    ↘
train_svm  train_mlp    train_cnn
   ↘    ↙    ↙
      evaluate
```

Run the full pipeline:
```bash
dvc repro
```

---

## Models

### SVM (Support Vector Machine)
- Features: HOG (26,244 dimensions)
- Preprocessing: StandardScaler
- Params: C=1.0, kernel=rbf

### MLP (Multi-Layer Perceptron)
- Features: HOG + PCA (200 components)
- Architecture: 256 → 128 → 20
- Params: max_iter=200, lr=0.001

### CNN (EfficientNetB0 Transfer Learning)
- Base: EfficientNetB0 (ImageNet pretrained)
- Training: 2-phase (frozen base → gentle fine-tuning)
- Phase 1: 20 epochs, lr=0.0005
- Phase 2: 50 epochs, unfreeze top 20 layers, lr=0.00001
- Class weights capped at 5.0

---

## Results

| Model | Test Accuracy | Test Macro F1 |
|-------|--------------|---------------|
| SVM   | 0.2920       | 0.0670        |
| MLP   | 0.2478       | 0.0895        |
| **CNN**   | **0.5929**   | **0.4320**    |

View metrics:
```bash
dvc metrics show
```

View plots:
```bash
dvc plots show
```

---

## Experiment History

| Tag | Model | Val F1 | Notes |
|-----|-------|--------|-------|
| exp-svm | SVM | 0.1103 | HOG + StandardScaler |
| exp-mlp | MLP | 0.1346 | HOG + PCA + MLP |
| exp-cnn-v3 | MobileNetV2 | 0.3355 | class weights cap=3 |
| exp-cnn-v7 | EfficientNetB0 | 0.3256 | phase2, unfreeze=20 |
| exp-final | EfficientNetB0 | **0.4320** | final test F1 |

---

## Project Structure

```
assignment-3-muhammed786fiyas/
├── src/
│   ├── prepare.py              # resize + split data
│   ├── transform.py            # HOG features + augmentation
│   ├── train_svm.py            # SVM training
│   ├── train_mlp.py            # MLP training
│   ├── train_cnn.py            # CNN training (EfficientNetB0)
│   ├── evaluate.py             # evaluation on test set
│   └── utils/
│       ├── config.py           # load config.yaml
│       └── logger.py           # timestamped logging
├── data/
│   └── raw.dvc                 # pointer to raw data
├── metrics/
│   ├── scores.json             # final F1 scores
│   ├── cnn_training_history.csv
│   ├── confusion_matrix.csv
│   └── f1_comparison.csv
├── dvc_plots/
│   └── index.html              # all DVC plots
├── reports/
│   └── report.pdf              # assignment report
├── screencast/
│   └── demo.mp4                # project walkthrough video
├── config.yaml                 # all file paths
├── params.yaml                 # all hyperparameters
├── dvc.yaml                    # pipeline definition
├── dvc.lock                    # pipeline state (reproducibility)
└── requirements.txt            # dependencies
```

---

## Deliverables

### Report
The full assignment report is available at:
```
reports/report.pdf
```

### Screencast
A video walkthrough of the project explaining:
- Pipeline structure and DVC usage
- Experiment tracking and results
- Model performance analysis

Available at:
```
screencast/demo.mp4
```

---

## Setup & Reproduction

### 1. Clone and setup environment
```bash
git clone https://github.com/DA5402-MLOps-JAN26/assignment-3-muhammed786fiyas
cd assignment-3-muhammed786fiyas

conda create -n mlops_a3 python=3.10
conda activate mlops_a3
pip install -r requirements.txt
```

### 2. Pull data
```bash
dvc pull
```

### 3. Run full pipeline
```bash
dvc repro
```

### 4. View results
```bash
dvc metrics show
dvc plots show
```

---

## Key Parameters

All hyperparameters are in `params.yaml`:

```yaml
train_cnn:
  base_model: efficientnetb0
  epochs: 50
  batch_size: 16
  learning_rate: 0.0005
  phase1_epochs: 20
  unfreeze_layers: 20
  finetune_lr: 0.00001
  max_class_weight: 5.0
```

To reproduce a specific experiment:
```bash
git checkout exp-cnn-v7
dvc checkout
dvc repro
```

---

## Dependencies

```
tensorflow >= 2.12
scikit-learn
scikit-image
numpy
pyyaml
dvc
```

Install all:
```bash
pip install -r requirements.txt
```