# Shell AI Hackathon - Fuel Blend Property Prediction

## Overview

This project predicts 10 properties of fuel blends (`BlendProperty1` – `BlendProperty10`) from the fractions and properties of their 5 components. The solution combines LightGBM, XGBoost, and CatBoost with feature engineering, 10-fold cross-validation, and a MAPE-optimized weighted ensemble trained separately for each target.

## Approach

### 1. Feature Engineering

Additional features are created from the raw blend composition data (5 components × 10 properties):

- **Weighted blend properties** – for each property, the sum of `ComponentN_fraction × ComponentN_PropertyM` across all components
- **Fraction interactions** – pairwise products of the 5 component fractions
- **Statistical property features** (for each of the 10 properties):
  - Mean across components
  - Standard deviation across components
  - Difference of each component's property from the mean

### 2. Models Used

Three gradient boosting models are trained for each target:

| Model    | Objective               | Key settings                                          |
| -------- | ----------------------- | ----------------------------------------------------- |
| LightGBM | MAPE                    | 2000 estimators, lr 0.01, early stopping (100 rounds) |
| XGBoost  | Squared error           | 2000 estimators, lr 0.01, max depth 6                 |
| CatBoost | MAPE                    | 2000 iterations, lr 0.02, depth 6, early stopping (100 rounds) |

### 3. Cross Validation

- 10-Fold K-Fold Cross Validation (shuffled, `random_state=42`)
- Out-of-fold (OOF) predictions are collected for each model
- Test predictions are averaged across the 10 folds

### 4. Weighted Ensemble

Instead of simple averaging, the model weights are optimized **per target** with SciPy's SLSQP optimizer to minimize MAPE on the out-of-fold predictions:

```
Final Prediction = w₁ × LightGBM + w₂ × XGBoost + w₃ × CatBoost
```

where `w₁ + w₂ + w₃ = 1` and each `wᵢ ∈ [0, 1]`.

### 5. Evaluation Metric

Mean Absolute Percentage Error (MAPE), using a safe implementation that adds a small epsilon to avoid division by zero.

## Project Structure

```
.
├── blend_property.ipynb   # Full pipeline: features, training, ensembling, submission
├── requirements.txt
└── README.md
```

The dataset files (`train.csv`, `test.csv`) are not included in this repository.

## Installation

1. Clone the repository

```bash
git clone https://github.com/AshutoshP29/fuel-efficiency-predictor-shell-ai-hackathon.git
cd fuel-efficiency-predictor-shell-ai-hackathon
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Solution

1. Place `train.csv` and `test.csv` (from the Shell AI Hackathon) in the project directory.
2. Open and run all cells in `blend_property.ipynb`:

```bash
jupyter notebook blend_property.ipynb
```

The notebook will:

1. Generate engineered features
2. Train all three models with 10-fold cross validation for each of the 10 targets
3. Find the optimal ensemble weights per target
4. Generate test predictions
5. Write the submission file

**Output:** `submission_mape_ensemble_compatible.csv`

## Key Features

- Domain-informed feature engineering
- LightGBM + XGBoost + CatBoost ensemble
- Per-target optimized weighted averaging
- 10-fold cross validation
- MAPE-based optimization
- Reproducible (seed = 42)

## Future Improvements

- Hyperparameter optimization using Optuna
- Stacking ensemble with a meta-learner
- Feature selection techniques
- Additional domain-specific fuel chemistry features

## Author

[Ashutosh P](https://github.com/AshutoshP29) — developed for the Shell AI Hackathon.
