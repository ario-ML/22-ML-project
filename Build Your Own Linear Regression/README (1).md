# Build Your Own Linear Regression

A from-scratch implementation of Linear Regression in Python/NumPy — no `sklearn.linear_model` used for the core algorithm. Built to understand the math and mechanics behind the model, not just to call a library.

## What's inside

- **Two training methods**, selectable via a `method` parameter:
  - `gradient_descent` — iterative optimization, stepping against the gradient of the MSE loss
  - `normal_equation` — closed-form analytical solution: `θ = (XᵀX + αI)⁻¹Xᵀy`
- **L2 (Ridge) regularization** on both methods, controlled by `alpha` (the bias/intercept term is never regularized)
- **Feature standardization** built in (`StandardScaler`), so gradient descent converges properly regardless of feature scale
- **A scikit-learn-style API**: `fit(X, y)`, `predict(X)`, `score(X, y)` (returns R²)

## Files

| File | Purpose |
|---|---|
| `linear_regression.py` | Core implementation: `LinearRegression` class + `StandardScaler` helper |
| `Build Your Own Linear Regression.ipynb` | Notebook: data cleaning, EDA, training, evaluation |
| `housing.csv` | California Housing Prices dataset (not included — see below) |

## Dataset

[California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) (Kaggle). ~20,000 rows, predicting `median_house_value` from features like `median_income`, `total_rooms`, `housing_median_age`, and `ocean_proximity`.

Download `housing.csv` from the link above and place it next to the notebook.

## Usage

```python
from linear_regression import LinearRegression

# Normal equation (closed-form, no hyperparameters to tune)
model = LinearRegression(method="normal_equation", alpha=0.0)
model.fit(X_train, y_train)
model.score(X_test, y_test)  # R²

# Gradient descent
model = LinearRegression(
    method="gradient_descent",
    learning_rate=0.1,
    n_iterations=5000,
    alpha=0.0,
)
model.fit(X_train, y_train)
model.predict(X_test)
```

## Preprocessing pipeline (as done in the notebook)

1. Fill/handle missing values in `total_bedrooms`
2. Encode the categorical `ocean_proximity` column to numeric
3. Split features (`X`) and target (`y`)
4. `train_test_split` — **before** scaling, to avoid data leakage
5. Fit `StandardScaler` on `X_train` only; transform both `X_train` and `X_test` with it

## Results

| Model | R² (test set) |
|---|---|
| Normal Equation (ours) | 0.6172 |
| Gradient Descent (ours, tuned) | 0.6172 |
| scikit-learn `LinearRegression` | 0.6172 |

All three converge to the same solution, which is expected: linear regression with MSE loss is a convex optimization problem with a unique global optimum, so any correct method — closed-form or iterative — should land on it. This agreement is effectively a correctness check on the from-scratch implementation.

The moderate R² suggests plain linear regression can't fully capture this dataset's structure (likely due to non-linear effects, e.g. raw latitude/longitude). Natural next steps: polynomial/interaction features, tuning `alpha` for regularization, or moving to a non-linear model.

## Requirements

- numpy
- pandas (notebook only)
- scikit-learn (notebook only, for comparison and preprocessing utilities)
