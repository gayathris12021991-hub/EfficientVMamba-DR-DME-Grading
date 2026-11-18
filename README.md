# EfficientVMamba-Based Joint Diabetic Retinopathy and DME Risk Grading

## Overview

This repository presents a multi-task deep learning framework for the simultaneous grading of Diabetic Retinopathy (DR) severity and Diabetic Macular Edema (DME) risk from color fundus images.

The proposed framework integrates an EfficientVMamba-S backbone with task-specific feature decomposition and bidirectional DR–DME cross-task attention. An Adaptive Hare Escape Optimization (AHEO) strategy is employed to optimize the model's training configuration.

The framework is evaluated using the Indian Diabetic Retinopathy Image Dataset (IDRiD) Disease Grading subset.

## Proposed Framework

The proposed architecture consists of four main components:

- **EfficientVMamba-S Backbone:** Extracts shared retinal representations using visual state-space processing, efficient 2D scanning, convolutional pathways, and Squeeze-and-Excitation refinement.
- **Task-Specific Feature Decomposition:** Projects shared visual tokens into separate DR and DME feature representations.
- **Bidirectional DR–DME Cross-Task Attention:** Enables reciprocal information exchange between the two task-specific representations through token-level scaled dot-product attention.
- **Task-Specific Classification Heads:**
  - DR grading: CORAL-style ordinal classification with five severity grades.
  - DME risk grading: Three-class softmax classification.

The training configuration is optimized using Adaptive Hare Escape Optimization (AHEO).

## Key Contributions

- EfficientVMamba-S-based multi-task learning for joint DR and DME risk grading.
- Decoupled task-specific representations for the two grading objectives.
- Bidirectional token-level cross-task attention for reciprocal feature exchange.
- CORAL-style ordinal learning for five-class DR severity grading.
- AHEO-based hyperparameter optimization.
- Comparative evaluation against alternative attention configurations, optimization methods, and contemporary deep learning architectures.
- Validation-controlled model selection with an independently held-out test set.

## Dataset

**Dataset:** Indian Diabetic Retinopathy Image Dataset (IDRiD) – Disease Grading Subset.

The dataset contains 516 color fundus images with two annotations:

- **DR severity:** Five classes (Grade 0–4).
- **DME risk:** Three classes (Grade 0–2).

### Dataset Partitioning

| Subset | Number of images |
|---|---|
| Training | 330 |
| Validation | 83 |
| Held-out testing | 103 |
| **Total** | **516** |

The 413 development images were divided into training and validation subsets using an image-level random split (scikit-learn `train_test_split`, `test_size = 0.2`, `random_state = 42`, without stratification).

The official 103-image test partition was kept separate from training and hyperparameter optimization.

### Dataset Access

The IDRiD dataset is available through its official distribution and associated research resources.

Dataset reference:

Kazeminezhad, *Indian Diabetic Retinopathy Image Dataset (IDRiD)—Segmentation, Grading & Localization ZIP Files*, Zenodo (2025).
DOI: https://doi.org/10.5281/zenodo.17219542

Please follow the dataset's applicable terms of use.

## Data Preprocessing

The preprocessing pipeline includes:

- Removal of non-retinal margins.
- Cropping of the visible retinal field.
- Resizing to 224 × 224 pixels.
- RGB image representation.
- Normalization of 8-bit intensity values to [0, 1].
- Online data augmentation during training.

The augmentation pipeline includes horizontal flipping, rotation, affine transformations, brightness variation, and contrast variation.

Validation and test images are not augmented.

## Hyperparameter Optimization

The proposed Adaptive Hare Escape Optimization (AHEO) strategy is used to identify an effective training configuration.

Five hyperparameters are optimized:

| Hyperparameter | Search range | Encoding |
|---|---|---|
| Learning rate | 1 × 10⁻⁵ – 1 × 10⁻³ | log₁₀-uniform |
| Batch size | 8–32 | integer |
| Weight decay | 1 × 10⁻⁶ – 1 × 10⁻³ | log₁₀-uniform |
| Dropout | 0.1–0.5 | uniform (continuous) |
| Training epochs | 50–150 | integer |

All four optimizers (random search, Optuna-TPE, HEO, and AHEO) use the same encoded search space. Learning rate and weight decay are searched in log₁₀ space: random search samples them log-uniformly, Optuna-TPE uses logarithmic float parameters, and HEO/AHEO perform position updates, normalization, and diversity computation in log₁₀-encoded coordinates.

**AHEO settings:** population size M = 10, iterations T = 10, Lévy-flight β = 1.5, k_D = 0.50, k_I = 1.00, γ ∈ [0.50, 1.50]. Each candidate is trained for a proxy length of round(0.2E) epochs, and the selected configuration is retrained at full length.

**Fitness function:** J = 0.5 (1 − macro-F1_DR) + 0.5 (1 − macro-F1_DME), computed on the validation subset.

### Selected Configuration

| Parameter | Selected value |
|---|---|
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 32 |
| Weight decay | 1 × 10⁻⁴ |
| Dropout | 0.30 |
| Training epochs | 100 |
| Optimizer | AdamW |
| Learning-rate scheduler | Linear warm-up + cosine annealing |
| Minimum learning rate | 1 × 10⁻⁶ |
| Gradient clipping | Global norm 1.0 |
| Random seed | 42 |

The main optimizer comparison used 110 fitness evaluations per optimizer. An additional repeated-seed comparison used five seeds (42, 101, 202, 303, 404) with a matched budget of 50 evaluations per run. Within each seed, all optimizers start from the same ten initial candidates, so the results are paired by seed and compared with the exact two-sided Wilcoxon signed-rank test with Holm correction.

## Experimental Environment

The experiments were conducted using:

- **Programming language:** Python
- **Deep learning framework:** PyTorch
- **GPU environment:** Google Colaboratory
- **GPU:** NVIDIA Tesla T4 (16 GB VRAM)
- **Image processing:** OpenCV
- **Numerical computation:** NumPy
- **Data augmentation:** Albumentations


## Experimental Results

All validation metrics (macro-F1, per-grade recall, and the objective J) are computed from the same validation confusion matrices (83 images). The final configuration reached a validation macro-F1 of 97.13% (DR) and 97.64% (DME), giving J = 0.0262.

The proposed framework achieved the following performance on the official 103-image IDRiD test set.

### Diabetic Retinopathy Grading

| Metric | Result |
|---|---|
| Accuracy | 98.06% |
| Matthews Correlation Coefficient (MCC) | 97.38% |
| Cohen's Kappa | 97.38% |
| Macro Precision | 98.32% |
| Macro Recall | 98.32% |
| Macro F1-score | 98.32% |

### Diabetic Macular Edema Risk Grading

| Metric | Result |
|---|---|
| Accuracy | 99.03% |
| Matthews Correlation Coefficient (MCC) | 98.34% |
| Cohen's Kappa | 98.32% |
| Macro Precision | 99.32% |
| Macro Recall | 96.67% |
| Macro F1-score | 97.90% |
