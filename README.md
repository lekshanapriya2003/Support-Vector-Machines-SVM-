# Breast Cancer Classification using Support Vector Machine (SVM)

A complete machine learning pipeline to classify breast tumors as **Malignant (1)** or **Benign (0)** using Support Vector Machines (SVM), including preprocessing, model comparison (Linear vs RBF), hyperparameter tuning, and decision boundary visualization.

---

## Problem Statement

Early detection of breast cancer significantly improves survival rates.
This project builds a classification model to predict whether a tumor is malignant or benign using numerical diagnostic features extracted from digitized images of breast mass.

---

## Dataset

**Dataset Used:** Breast Cancer Wisconsin (Diagnostic) Dataset

* Total Samples: **569**
* Features: **30 numerical features**
* Target Variable:

  * `M` → Malignant → **1**
  * `B` → Benign → **0**

### Feature Categories

Each tumor is described using three measurements:

* Mean
* Standard Error (SE)
* Worst (largest value)

Examples:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Symmetry
* Fractal dimension

---

## Project Workflow

### 1. Data Preprocessing

* Dropped unnecessary columns (`id`, unnamed columns)
* Encoded target variable (`M=1`, `B=0`)
* Train-test split (80% train / 20% test)
* Feature scaling using `StandardScaler`

Why scaling?

SVM is distance-based. Without scaling, large-value features dominate smaller ones.

---

### 2. Model Training

Two SVM kernels were tested.

#### Linear Kernel SVM

Accuracy: **95.61%**

Confusion Matrix:

```
[[41  2]
 [ 3 68]]
```

Interpretation:

* 2 false positives
* 3 false negatives

---

#### RBF Kernel SVM

Accuracy: **98.24%**

Confusion Matrix:

```
[[41  2]
 [ 0 71]]
```

Interpretation:

* 0 false negatives (critical in medical diagnosis)
* Better performance than linear kernel

---

### 3. Hyperparameter Tuning

Used `GridSearchCV` to optimize:

* `C` → Regularization parameter
* `gamma` → Kernel coefficient

Best Parameters:

```
{'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
```

Cross-validation Score: **97.58%**
Optimized Test Accuracy: **98.24%**

There was no improvement over default RBF parameters, indicating the defaults were already near optimal for this dataset.

---

### 4. Decision Boundary Visualization

To visualize classification behavior:

* Reduced dataset to 2 features
* Trained Linear SVM
* Used `mlxtend` to plot decision regions

This helps understand:

* Margin separation
* Support vectors
* Linear separability in lower dimensions

---

## Model Comparison

| Model      | Accuracy | False Negatives | Best Use Case               |
| ---------- | -------- | --------------- | --------------------------- |
| Linear SVM | 95.61%   | 3               | Linearly separable data     |
| RBF SVM    | 98.24%   | 0               | Non-linear complex patterns |

Conclusion: RBF kernel performs significantly better for this dataset.

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* mlxtend

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib mlxtend
```

### 2. Run the notebook or script

```bash
python svm_breast_cancer.py
```

---

## Key Learnings

* Feature scaling is mandatory for SVM.
* RBF kernel captures non-linear relationships effectively.
* Hyperparameter tuning does not always guarantee better performance.
* In medical classification, minimizing false negatives is more important than maximizing overall accuracy.

---

## Important Observation

Even though accuracy is high, this dataset is relatively clean and well-structured.
Real-world medical datasets are noisier and often more imbalanced.

This model should not be used in production healthcare settings without rigorous validation, regulatory review, and external testing.

---

## Future Improvements

* Add ROC-AUC evaluation
* Perform cross-validation comparison across multiple models
* Compare with:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Apply PCA before SVM
* Perform class imbalance testing

---

## Final Verdict

The RBF SVM model achieves:

* **98.24% Accuracy**
* **Zero False Negatives on Test Set**
* Strong generalization on this dataset

For this dataset, SVM with RBF kernel is an effective classifier.

