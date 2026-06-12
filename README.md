# Shell AI Hackathon - Fuel Blend Property Prediction

## Overview

This project predicts the properties of fuel blends using an ensemble of gradient boosting models. The solution combines LightGBM, XGBoost, and CatBoost with advanced feature engineering, K-Fold cross-validation, and optimized weighted ensembling to achieve high prediction accuracy.

## Approach

### 1. Feature Engineering

The model creates additional features from the raw blend composition data:

- Weighted blend properties using component fractions
- Fraction interaction features between fuel components
- Statistical property features:
  - Mean of component properties
  - Standard deviation of component properties
  - Difference from mean for each component property

These engineered features help capture relationships between blend composition and final blend properties.

### 2. Models Used

Three gradient boosting models are trained independently:

- LightGBM Regressor
- XGBoost Regressor
- CatBoost Regressor

### 3. Cross Validation

- 10-Fold K-Fold Cross Validation
- Shuffle enabled
- Random seed = 42

Out-of-fold predictions are generated for each model.

### 4. Weighted Ensemble

Instead of simple averaging, model weights are optimized using SciPy's constrained optimization (SLSQP) to minimize Mean Absolute Percentage Error (MAPE).

Final prediction:

```
Final Prediction = w₁ × LightGBM + w₂ × XGBoost + w₃ × CatBoost
```

Where: `w₁ + w₂ + w₃ = 1`

### 5. Evaluation Metric

Mean Absolute Percentage Error (MAPE)

A custom safe MAPE implementation is used to avoid division-by-zero issues.

## Project Structure

```
.
├── train.csv
├── test.csv
├── main.py
├── requirements.txt
├── README.md
└── submission_mape_ensemble_compatible.csv
```

## Installation

1. Clone the repository

```bash
git clone <repository-url>
cd shell-ai-hackathon
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Solution

Place `train.csv` and `test.csv` in the project directory.

Run:

```bash
python main.py
```

The script will:

1. Generate engineered features
2. Train all models using 10-fold cross validation
3. Find optimal ensemble weights
4. Generate predictions
5. Create the submission file

Output: `submission_mape_ensemble_compatible.csv`

## Key Features

- Advanced feature engineering
- LightGBM + XGBoost + CatBoost ensemble
- Optimized weighted averaging
- 10-Fold cross validation
- MAPE-based optimization
- Fully reproducible (seed = 42)

## Future Improvements

- Hyperparameter optimization using Optuna
- Stacking ensemble with meta-learner
- Feature selection techniques
- Additional domain-specific fuel chemistry features

## Authors

Developed for the Shell AI Hackathon
