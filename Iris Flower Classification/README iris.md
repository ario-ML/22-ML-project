# Iris Flower Classification

A machine learning project that classifies iris flowers into three species — *Setosa*, *Versicolor*, and *Virginica* — based on four physical measurements (sepal length, sepal width, petal length, petal width).

## Dataset

The classic **Iris dataset**, loaded directly from `sklearn.datasets.load_iris()`. It contains 150 samples, 50 per species, with no missing values.

## Project Workflow

1. **Exploratory Data Analysis (EDA)** — examined feature distributions and relationships between species using pairplots and correlation analysis. Setosa is linearly separable from the other two species; Versicolor and Virginica show some overlap.
2. **Preprocessing** — split the data into training and test sets (80/20).
3. **Model Training & Comparison** — trained and evaluated three classifiers:
   - Decision Tree Classifier
   - K-Nearest Neighbors (tuned K from 1–9 to find the optimal value)
   - Logistic Regression (tuned regularization strength `C`)
4. **Evaluation** — beyond accuracy, used confusion matrices and classification reports (precision, recall, F1-score) to understand per-class performance and identify where each model makes mistakes.

## Results

| Model | Accuracy |
|---|---|
| Decision Tree Classifier | 0.967 |
| K-Nearest Neighbors (K=3) | 0.967 |
| Logistic Regression | 0.933 |

Decision Tree and KNN slightly outperformed Logistic Regression on this test split. Most misclassifications, where they occurred, were between Versicolor and Virginica — consistent with the overlap observed during EDA.

## Tools & Libraries

- Python, Jupyter Notebook
- scikit-learn (models, metrics)
- pandas, numpy
- seaborn, matplotlib (visualization)

## How to Run

1. Open `Iris Flower Classification.ipynb` in Jupyter.
2. Run all cells in order.

## Possible Next Steps

- Add PCA visualization to show class separability in 2D.
- Try cross-validation instead of a single train/test split for more robust accuracy estimates.
- Add feature importance analysis for the Decision Tree model.
