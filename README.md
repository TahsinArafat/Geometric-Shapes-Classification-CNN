# Geometric Shapes Classification (CNN, PyTorch)

A simple Convolutional Neural Network (CNN) built with PyTorch to classify three geometric shapes:
**Circle**, **Square**, **Triangle**.

**Best way to run:** open the notebook in Google Colab and execute the cells top-to-bottom.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TahsinArafat/Geometric-Shapes-Classification-CNN/blob/main/210116.ipynb)

---

## Project Overview
This repository contains:
- A full training + evaluation notebook: `210116.ipynb`
- A standard dataset split under `training/` (train/validation/test)
- A small custom “phone” dataset under `dataset/` for real-world testing
- Saved trained weights under `model/`
- Pre-generated plots and figures under `results/`

The notebook trains a CNN and then evaluates:
1. Standard test split performance (including a confusion matrix)
2. Predictions on the custom smartphone photos

---

## Model Architecture
The model is implemented with `torch.nn.Module` and is designed for **$64 \times 64$ RGB** images.

**Architecture (from the notebook):**
- Conv(3→32, k=3, p=1) → ReLU → MaxPool(2×2)
- Conv(32→64, k=3, p=1) → ReLU → MaxPool(2×2)
- Flatten (64×16×16)
- FC(64×16×16 → 128) → ReLU
- FC(128 → 3)

**Training hyperparameters (from the notebook):**
- `IMG_SIZE = 64`
- `BATCH_SIZE = 64`
- `LEARNING_RATE = 0.001`
- `EPOCHS = 10`

**Preprocessing (from the notebook):**
- Resize to `(64, 64)`
- Normalize mean `(0.5, 0.5, 0.5)` and std `(0.5, 0.5, 0.5)`

---

## Repository Structure
```text
.
├── 210116.ipynb                # Main notebook (train/eval/plots)
├── README.md
├── model/
│   └── 210116.pth              # Saved model weights
├── results/
│   ├── AvE.png                 # Accuracy vs epochs
│   ├── LvE.png                 # Loss vs epochs
│   ├── confusion_matrix.png
│   ├── custom_predictions.png
│   └── error_analysis.png
├── dataset/                    # Custom smartphone photos (10 images)
└── training/                   # Standard dataset split
    ├── train/
    ├── validation/
    └── test/
```

---

## Datasets
### 1) Standard split (`training/`)
The notebook uses `torchvision.datasets.ImageFolder` with this structure:
`training/{train,validation,test}/{circles,squares,triangles}/...`

### 2) Custom phone photos (`dataset/`)
`dataset/` contains **10** real-world images (e.g. `circle1.jpg`, `square1.jpg`, `triangle1.jpg`) used for quick sanity-check predictions.

---

## Results & Visuals

### 1) Training History
| Accuracy vs Epochs | Loss vs Epochs |
| :---: | :---: |
| ![Accuracy Plot](results/AvE.png) | ![Loss Plot](results/LvE.png) |

### 2) Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### 3) Custom "Phone" Predictions
![Custom Predictions](results/custom_predictions.png)

### 4) Error Analysis
![Error Analysis](results/error_analysis.png)

---

## How to Run

### Option A: Google Colab (recommended)
1. Open `210116.ipynb` in Colab (badge at the top of this README).
2. Run all cells.

The notebook will:
- Clone this repository
- Load the datasets from `training/` and `dataset/`
- Train the CNN
- Evaluate on the standard test split
- Generate plots (accuracy/loss), confusion matrix, and error analysis
- Run predictions on the custom phone photos

### Option B: Run locally
If you prefer running locally, install the notebook dependencies (based on the notebook imports):
```bash
pip install torch torchvision matplotlib numpy pillow scikit-learn seaborn
```
Then open and run the notebook:
```bash
jupyter notebook 210116.ipynb
```

---

## Notes
- `model/210116.pth` is a saved state dictionary produced by training.
- Class names are inferred from folder names via `ImageFolder` (e.g. `circles`, `squares`, `triangles`).