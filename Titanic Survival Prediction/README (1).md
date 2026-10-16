# 🚢 Titanic Survival Prediction

## Overview
This project builds a machine learning model to predict whether a Titanic passenger survived, using the classic Titanic dataset (from Kaggle). The model takes passenger information — such as age, sex, ticket class, and family size — and predicts a binary outcome: **survived (1)** or **did not survive (0)**.

## Dataset
The dataset (`train.csv`) contains information on 891 Titanic passengers, including:
- `PassengerId`, `Name`, `Ticket` — identifiers (not used directly in the model)
- `Survived` — the target variable (0 = No, 1 = Yes)
- `Pclass` — ticket class (1st, 2nd, 3rd), a proxy for socio-economic status
- `Sex`, `Age` — demographic info
- `SibSp`, `Parch` — number of siblings/spouses and parents/children aboard
- `Fare` — ticket price
- `Cabin`, `Embarked` — cabin number and port of embarkation

## Workflow

### 1. Data Cleaning
- `Embarked` — filled missing values with the mode (`S`)
- `Age` — filled missing values with the median age grouped by `Title`
- `Cabin` — mostly missing (~77%); converted into a `Deck` feature instead of being dropped entirely

### 2. Feature Engineering
- **`Title`** — extracted from `Name` (e.g. Mr, Mrs, Miss, Master), with rare titles (Dr, Rev, Major, Col, Don, Lady, Sir, Capt, Countess, Jonkheer) grouped into a single `Rare` category
- **`Deck`** — extracted from the first letter of `Cabin`; missing cabins labeled `U` (Unknown)
- **`FamilySize`** — `SibSp + Parch + 1`
- **`IsAlone`** — binary flag, `1` if the passenger had no family aboard

### 3. Encoding
- `Sex` — label encoded (0 = male, 1 = female)
- `Embarked`, `Title`, `Deck` — one-hot encoded (with `drop_first=True` to avoid redundant columns)

### 4. Dropped Columns
`Name`, `Ticket`, `PassengerId`, and the original `Cabin` were dropped after their useful information was extracted into new features.

### 5. Modeling
The cleaned dataset was split into training (80%) and test (20%) sets. Two models were trained and compared:

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Logistic Regression | – | 86.0% |
| Decision Tree (no depth limit) | 98.6% | 81.0% |
| Decision Tree (max_depth=4) | 83.4% | 84.9% |

## Key Takeaways
- Feature engineering (especially `Title`, `FamilySize`, and `IsAlone`) meaningfully improved model performance.
- An unrestricted Decision Tree overfit the training data (a large gap between train and test accuracy).
- Limiting `max_depth` reduced overfitting and improved generalization.
- Logistic Regression achieved the best overall test accuracy among the models tried.

## Possible Next Steps
- Try a Random Forest classifier
- Apply feature scaling to improve Logistic Regression's convergence
- Use cross-validation for a more robust performance estimate
- Tune hyperparameters systematically (e.g. with `GridSearchCV`)

## Requirements
- Python 3
- pandas
- scikit-learn

## How to Run
1. Open `Titanic Survival Prediction.ipynb` in Jupyter Notebook
2. Run the cells in order from top to bottom
3. The final cells show model training, evaluation, and prediction on new data
