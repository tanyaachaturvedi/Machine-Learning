# Machine Learning Practice

This repository is my personal collection of machine learning projects and notebooks, created to document my learning journey. It's designed to be beginner-friendly, with explanations for the key concepts used in each project.

* **Supervised_Learning:** Models that learn from data with a known "answer" or "label" (e.g., predicting prices, classifying species).
* **Unsupervised_Learning:** Models that find hidden patterns in data without any labels (e.g., clustering customers).

## Projects

Below is a list of the projects I've completed. Each is a self-contained notebook.

---

### 1. Fundamental Regression & Classification

* **File:** `Supervised_Learning/Linear_and_Logistic_Regression.ipynb`
* **Description:** A notebook implementing and evaluating fundamental supervised learning models for both regression (predicting a value) and classification (predicting a category).

**What I Implemented:**

* [Exploratory Data Analysis (EDA)](#eda)
* [Linear Regression](#linear-regression)
* [Ridge (L2) & Lasso (L1) Regression](#regularization)
* [GridSearchCV](#gridsearchcv)
* [Logistic Regression](#logistic-regression)
* [Model Evaluation Metrics](#model-evaluation) (MSE, R², Accuracy, Confusion Matrix, etc.)

---

### 2. Fundamental Classification (Decision Tree)

* **File:** `Supervised_Learning/DecisionTree.ipynb`
* **Description:** A beginner-friendly implementation of a Decision Tree classifier using the classic Iris dataset.

**What I Implemented:**

* [Exploratory Data Analysis (EDA)](#eda)
* **Train-Test Split:** Correctly separating data to evaluate the model's performance on unseen data.
* [Decision Tree Classifier](#decision-tree)
* **Tree Visualization:** Plotted the final tree in a human-readable format.
* [Feature Importance](#feature-importance)
* [Model Evaluation Metrics](#model-evaluation) (Accuracy & Classification Report)

---

### ✅ 3. K-Means Clustering (From Scratch + Sklearn) — *Unsupervised Learning*

**File:**  
- `Unsupervised_Learning/KMeans_Clustering.ipynb`

**Description:**  
This notebook introduces **unsupervised learning** using the K-Means clustering algorithm. The algorithm is implemented **from scratch using NumPy** and also using **scikit-learn**, with visual comparison.

**What I Implemented:**
- What is clustering?
- What is K-Means algorithm?
- Euclidean distance calculation
- Random centroid initialization
- Cluster assignment using distance
- Centroid update using mean
- Full K-Means implementation **from scratch**
- K-Means using **scikit-learn**
- Data visualization using Matplotlib
- Visual comparison of:
  - From-scratch result
  - Sklearn result

## 📚 Key Concepts & Glossary

Here are simple definitions for the key terms used in my projects.

### <a name="decision-tree"></a>Decision Tree Classifier

A popular and intuitive **supervised learning** algorithm for **classification**. It works by learning a series of simple "if-then-else" rules from the data, creating a tree-like structure. For example, "Is the *petal length < 2.5cm*? If YES, predict *setosa*. If NO, ask another question."

### <a name="eda"></a>Exploratory Data Analysis (EDA)

The practice of using `pandas` commands (like `.head()`, `.info()`, `.describe()`) and visualizations to understand a dataset's main characteristics, find patterns, spot anomalies, and check for missing data *before* building a model.

### <a name="feature-importance"></a>Feature Importance

A score (usually from 0 to 1) that a model (like a Decision Tree or Random Forest) assigns to each input feature. It tells you which features the model found *most useful* for making its predictions. For example, the Iris model likely found "petal length" to be the most important feature.

### <a name="gridsearchcv"></a>GridSearchCV

Stands for "Grid Search Cross-Validation." This is an automated technique for **hyperparameter tuning**. You give it a "grid" of possible parameters (e.g., `alpha: [0.1, 1, 10]`), and it automatically tests every single combination to find the one that produces the best-performing model.

### <a name="linear-regression"></a>Linear Regression

A foundational **supervised learning** algorithm used for **regression** (predicting a continuous value, like a price). It works by finding the best possible straight line (or plane) that fits the relationship between the input features (e.g., "number of rooms") and the output target (e.g., "house price").

### <a name="logistic-regression"></a>Logistic Regression

A foundational **supervised learning** algorithm used for **classification** (predicting a category, like "has cancer" or "does not have cancer"). Despite its name, it's for classification, not regression. It works by fitting a logistic (S-shaped) curve to the data and predicting the *probability* (from 0 to 1) that an input belongs to a certain class.

### ✅ K-Means Clustering

A popular **unsupervised learning algorithm** used for grouping similar data points into clusters. The algorithm works by:
1. Choosing random centroids
2. Assigning each data point to the nearest centroid
3. Updating centroids using the mean of assigned points
4. Repeating until the clusters stabilize

### <a name="model-evaluation"></a>Model Evaluation Metrics

The numbers we use to measure how good a model is. The metric you choose depends on the task.

**For Regression (Predicting Values):**

* **Mean Squared Error (MSE):** The average of the *squared* differences between the actual and predicted values. A lower MSE is better.
* **R-squared ($R^2$):** A score between 0 and 1 that measures how much of the variation in the target value (e.g., price) our model can explain. A score of 1.0 is a perfect fit.

**For Classification (Predicting Categories):**

* **Accuracy:** The simplest metric. It's the percentage of predictions the model got right (e.g., 95% accuracy).
* **Confusion Matrix:** A table that shows *where* the model got confused. It breaks down predictions into True Positives, True Negatives, False Positives, and False Negatives.
* **Precision, Recall, F1-Score:**
    * **Precision:** Of all the times the model predicted "Yes," what percentage was correct? (Good for minimizing false positives).
    * **Recall:** Of all the *actual* "Yes" cases, what percentage did the model find? (Good for minimizing false negatives).
    * **F1-Score:** The harmonic mean (a special average) of Precision and Recall. It provides a single score that balances both.

### <a name="regularization"></a>Regularization (Ridge & Lasso)

A technique used to prevent **overfitting** (when a model learns the training data *too* well and fails on new data). It works by adding a small penalty to the model for having features with large, complex "weights".
* **Ridge (L2):** Shrinks large weights, but rarely makes them zero. Good all-around choice.
* **Lasso (L1):** Can shrink weights all the way to zero, effectively performing automatic feature selection.
