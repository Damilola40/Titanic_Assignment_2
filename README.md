# Titanic Survival Prediction

A machine learning project that predicts whether a passenger survived the Titanic disaster, using the classic Titanic dataset.

## What this project does
- Cleans the raw Titanic train/test data (missing Age, Embarked, Fare, Cabin)
- Explores feature relationships using a correlation heatmap
- Trains and compares three classification models: Logistic Regression, Decision Tree, Random Forest
- Saves the best-performing model and uses it to predict survival on the test set

## Files
- `data_cleaning.py` — cleans raw data, saves `train_cleaned.csv` and `test_cleaned.csv`
- `train_model.py` — trains and evaluates the three models, saves the best one as `titanic_model.pkl`
- `predict_test.py` — loads the saved model and generates `submission.csv`

## Data cleaning steps
- Filled missing `Age` using median age grouped by `Pclass` and `Sex`
- Filled missing `Embarked` with the most common port
- Filled missing `Fare` using median fare grouped by `Pclass`
- Replaced `Cabin` with a `Has_Cabin` flag (1 = cabin recorded, 0 = missing)
- Dropped `Ticket` and `Name` columns

## Features used
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Has_Cabin

## Model results
- Logistic Regression: 82.12%
- Random Forest: 81.56%
- Decision Tree: 78.21%

## How to run
1. `python3 data_cleaning.py`
2. `python3 train_model.py`
3. `python3 predict_test.py`
