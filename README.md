# Brain Stroke Prediction: Scenario 1

## Project overview

This repository contains the Jupyter notebook for **Scenario 1: Brain Stroke Prediction using Machine Learning Classification**.

The objective is to build, compare and evaluate machine learning classification models that predict the target variable `stroke` from demographic, lifestyle and medical features. The dataset is highly imbalanced, so model selection focuses on **F1-score** rather than accuracy alone.

The final notebook produces a predictions CSV for the unseen dataset with a single required column named `Predictions`.

## Repository contents

```text
.
├── Fenner_Backhouse_scenario_1.ipynb
├── Fenner_Backhouse_scenario_1.csv              # Created after running the notebook
└── Processed_datasets/
    └── Brain Stroke/
        ├── brain_stroke_data.csv
        └── brain_stroke_unseen.csv
```

## Dataset

The project uses the Brain Stroke dataset from Kaggle:

- Source: https://www.kaggle.com/datasets/jillanisofttech/brain-stroke-dataset
- Training file: `brain_stroke_data.csv`
- Unseen prediction file: `brain_stroke_unseen.csv`
- Target variable: `stroke`

The unseen dataset does not contain the target column. It is used only at the end of the notebook to generate the final predictions file.

## Features

The model uses the following feature groups:

| Feature type | Variables |
|---|---|
| Continuous | `age`, `avg_glucose_level`, `bmi` |
| Binary | `hypertension`, `heart_disease` |
| Categorical | `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status` |

The value `Unknown` in `smoking_status` is treated as a valid category rather than as a missing value.

## Methodology

The notebook follows the full data science process up to model evaluation:

1. Data loading and initial inspection
2. Missing value and duplicate checks
3. Exploratory data analysis
4. Feature grouping and preprocessing
5. Feature engineering
6. Stratified train-test split
7. Out-of-fold model comparison
8. Class imbalance handling
9. Hyperparameter tuning
10. Final internal test evaluation
11. Unseen data prediction generation

## Preprocessing

The preprocessing pipeline applies:

- Standard scaling to continuous features
- Direct passthrough for binary features
- One-hot encoding for categorical features

No imputation is applied because both the training and unseen datasets contain no missing values.

## Class imbalance handling

The target variable is highly imbalanced, with substantially fewer stroke cases than non-stroke cases. To address this, the notebook compares:

- Class weighting
- Random oversampling
- Standard baseline models without imbalance adjustment

The final model is selected primarily using F1-score.

## Models compared

The notebook compares a focused set of classification models:

- Dummy baseline
- Logistic Regression
- K-Nearest Neighbours
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine
- Logistic Regression with random oversampling
- Random Forest with random oversampling
- Gradient Boosting with random oversampling
- Engineered Logistic Regression with random oversampling

## Model evaluation

The notebook reports:

- Classification report
- Confusion matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- ROC curve
- Precision-recall curve

F1-score is used as the main model selection metric because accuracy can be misleading for imbalanced classification.

## Out-of-fold evaluation

The notebook uses out-of-fold predictions for model comparison and threshold tuning.

This means each training row is predicted by a model that was not trained on that row. This gives a more reliable estimate of performance than tuning a threshold on the same data used to fit the model.

## Feature engineering

The notebook tests engineered features such as:

- `medical_risk_count`
- `age_hypertension`
- `age_heart_disease`
- `glucose_bmi_ratio`
- `age_group`
- `bmi_group`

These features are only used in the final model if they improve out-of-fold F1-score.

## Output

Running the notebook creates:

```text
Fenner_Backhouse_scenario_1.csv
```

The file contains one column:

```text
Predictions
```

Each row contains the predicted class for the corresponding row in `brain_stroke_unseen.csv`.

## How to run

1. Clone this repository.
2. Place the dataset files in the following folder:

```text
Processed_datasets/Brain Stroke/
```

3. Ensure the files are named:

```text
brain_stroke_data.csv
brain_stroke_unseen.csv
```

4. Install the required Python packages.
5. Open and run the notebook:

```text
Fenner_Backhouse_scenario_1.ipynb
```

6. The predictions file will be saved as:

```text
Fenner_Backhouse_scenario_1.csv
```

## Requirements

The notebook uses the following main Python libraries:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

Install the required packages with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## Notes

This notebook is intended as a clean submission notebook. It does not include private checks against hidden labels or additional exploratory experiments outside the final modelling process.
