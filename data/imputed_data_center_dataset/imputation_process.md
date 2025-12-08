# Imputed Data Center Dataset

This directory contains the *fully imputed* version of the consolidated data center dataset.  
All missing numerical and categorical values have been filled using the imputation procedures defined in the `data_source` module.  
This document provides an overview of the mathematical basis for each imputation method and specifies the **default pipeline** used by `generate_all_data.py`.

---

## 1. Overview of Imputation Methods

The `data_source` class implements several imputation and scaling strategies.  
Each is designed for different types of missingness and downstream analytical needs.

### **1.1 Mean Imputation**
For numeric columns:
$$
x_i = 
\begin{cases}
x_i, & \text{if } x_i \text{ is observed} \\
\bar{x} = \frac{1}{n} \sum_{j=1}^{n} x_j, & \text{if } x_i \text{ is missing}
\end{cases}
$$
Simple and fast, but reduces variance and can bias relationships between variables.

---

### **1.2 Median Imputation**
Similar to mean imputation but uses the median:
$$
x_i =
\begin{cases}
x_i, & \text{if } x_i \text{ is observed} \\
\tilde{x}, & \text{if } x_i \text{ is missing}
\end{cases}
$$
More robust to outliers than mean imputation, often used for highly skewed data.

---

### **1.3 Mode (Most Frequent) Imputation**
Used for categorical features:
$$
x_i =
\begin{cases}
x_i, & \text{if } x_i \text{ is observed} \\
\operatorname{mode}(X), & \text{if } x_i \text{ is missing}
\end{cases}
$$

---

### **1.4 Z-Score Standardization**
When the user specifies `method="zscore"`, the following normalization is applied before modeling or imputation:
$$
x'_i = \frac{x_i - \mu}{\sigma}
$$
where \(\mu\) and \(\sigma\) are the mean and standard deviation of the column.  
This transformation ensures each feature has zero mean and unit variance.

---

### **1.5 Min–Max Scaling**
When the user specifies `method="minmax"`, all numeric features are scaled to the [0, 1] range:
$$
x'_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}
$$
This is useful when model interpretability depends on maintaining non-negative bounded values.

---

### **1.6 Iterative Bayesian Ridge Imputation (Default Method)**
The most sophisticated and default approach in the pipeline uses a **Multivariate Iterative Imputer**:
$$
x_j = f_j(X_{\neg j}) + \epsilon_j
$$
where \(f_j\) is estimated via **Bayesian Ridge Regression** and \(X_{\neg j}\) denotes all other features.

This process resembles the MICE (Multiple Imputation by Chained Equations) algorithm:

1. Initialize all missing values using mean imputation.
2. For each feature \(x_j\) with missing values:
   - Regress \(x_j\) on all other current feature estimates \(X_{\neg j}\).
   - Use the fitted model \(f_j\) to predict missing values for \(x_j\).
3. Iterate over all features multiple times until convergence or a maximum iteration count (default = 10).

The **Bayesian Ridge** model introduces L2-regularization with probabilistic priors:
$$
p(w|\lambda) \propto \exp\left(-\frac{\lambda}{2}\|w\|^2\right)
$$
$$
p(y|X,w,\alpha) \propto \exp\left(-\frac{\alpha}{2}\|y - Xw\|^2\right)
$$
which stabilizes estimates and propagates uncertainty across chained regressions.

---

## 2. Default Imputation Pipeline (`generate_all_data.py`)

By default, `generate_all_data.py` executes the following steps:

1. **Data Cleaning**  
   - Standardizes column names and types.  
   - Drops duplicate rows and trivial metadata columns.

2. **Feature Scaling (Optional)**  
   - If specified, applies `StandardScaler()` for z-score normalization or `MinMaxScaler()` for bounded features.  
   - Default pipeline leaves values unscaled for interpretability.

3. **Iterative Imputation**  
   - Missing numeric values are filled using:
     ```python
     imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
     ```
   - Each feature with missing values is modeled as a function of all others.
   - The imputation process continues until convergence.

4. **Categorical Handling**  
   - Categorical columns are imputed with their most frequent category.
   - Encoded representations (if any) are included as auxiliary predictors.

5. **Output**  
   - The fully imputed datasets are saved as multiple CSV files (one per imputation) using the repository naming pattern:
      ```
      data/imputed_data_center_dataset/DCS_Imputed_<TARGET>_<i>.csv
      ```
      where `<TARGET>` can be `Full`, `PJM`, or `New_York` and `<i>` is the imputation index (1..m).
   - The files retain the same schema as the cleaned source data, ensuring drop-in compatibility for downstream modeling.

---

## 3. Notes and Limitations

- **Assumption:** Missingness is at least MAR (Missing At Random).  
- **Effect:** Iterative BayesianRidge can over-smooth extreme values but preserves correlations.  
- **Recommendation:** For uncertainty analysis, run multiple imputations with different random seeds.



