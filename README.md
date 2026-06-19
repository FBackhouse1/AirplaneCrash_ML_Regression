# Scenario 2: Airplane Crash Fatalities Prediction

## Project overview

This repository contains the final machine-learning pipeline for Scenario 2 of the Data Science Process project. The task is to build a regression model that predicts the number of fatalities in historical airplane crash records.

The target variable is:

```text
Fatalities
```

The final submission output is a CSV file with one column:

```text
Predictions
```

The notebook follows the full data science process, including data loading, data checks, cleaning, exploratory data analysis, feature engineering, preprocessing, model comparison, hyperparameter tuning, model evaluation and final prediction generation.

## Repository structure

Expected structure:

```text
.
├── Fenner_Backhouse_scenario_2_final_pipeline.ipynb
├── Fenner_Backhouse_scenario_2.csv
├── README.md
└── Processed_datasets/
    └── Airplanes/
        ├── Airplane_Crashes_and_Fatalities_Since_1908_data.csv
        └── Airplane_Crashes_and_Fatalities_Since_1908_unseen.csv
```

## Input data

The notebook expects the datasets to be stored using the following relative paths:

```text
Processed_datasets/Airplanes/Airplane_Crashes_and_Fatalities_Since_1908_data.csv
Processed_datasets/Airplanes/Airplane_Crashes_and_Fatalities_Since_1908_unseen.csv
```

## Output file

The final predictions are saved in the same folder as the notebook:

```text
Fenner_Backhouse_scenario_2.csv
```

The file contains exactly one column:

```text
Predictions
```

## Methodology

The pipeline includes:

1. Data loading and validation
2. Missing-value and duplicate checks
3. Target cleaning
4. Exploratory data analysis
5. Feature engineering
6. Preprocessing
7. Model comparison
8. Focused hyperparameter tuning
9. Final internal test evaluation
10. Final prediction generation

The final model is selected using cross-validated R². RMSE, MSE and MAE are also reported.

## Feature engineering

The notebook creates date, time, numeric exposure, aircraft, operator, route, location and summary-text features.

Location features include broad region, over-water indicators, remote-terrain indicators and simple location wording flags. These are used instead of coordinates because many historical locations are ambiguous and external geocoding would reduce reproducibility.

The `Summary` column is tokenised using TF-IDF because it contains accident descriptions. Numeric tokens are removed before TF-IDF is applied.

## Models compared

The notebook compares:

* Dummy Regressor
* Linear Regression
* Ridge Regression
* Ridge Regression with log-transformed target
* Lasso Regression
* ElasticNet Regression
* K-Nearest Neighbours Regressor
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* Extra Trees Regressor

## How to run

1. Place the notebook in the project root folder.
2. Place the datasets in:

```text
Processed_datasets/Airplanes/
```

3. Run all notebook cells from top to bottom.
4. Confirm that the final prediction file has been created:

```text
Fenner_Backhouse_scenario_2.csv
```

5. Confirm that the output file contains exactly one column named `Predictions`.

## Requirements

Main libraries used:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
```

::: 
