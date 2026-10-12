# Patient-Risk-Segmentation-using-the-Diabetes-Prediction-Dataset


# Patient Risk Segmentation and Diabetes Classification

## Project Overview

This project explores patient health data using **statistical analysis, data mining, unsupervised learning, and machine learning** to identify diabetes-related patterns, segment patients based on health characteristics, and classify diabetes status.

The analysis combines Exploratory Data Analysis (EDA), K-Means clustering, Principal Component Analysis (PCA), Hierarchical Clustering, Apriori Association Rule Mining, and supervised classification algorithms.

The objective is to understand relationships among demographic and physiological factors and evaluate different machine learning approaches for diabetes classification.

## Dataset

**Source:** [Diabetes Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

- **Original dataset:** 99,982 records and 9 columns
- **Adult population (age ≥ 18):** 82,767 records
- **Target variable:** `diabetes` (0 = Non-diabetic, 1 = Diabetic)
- **Class distribution:** Approximately 89.83% non-diabetic and 10.17% diabetic in the adult subset
- **Modeling features:** 8 predictors after preprocessing and feature engineering

**Key variables:** Age, Gender, Hypertension, Heart Disease, BMI, HbA1c Level, and Blood Glucose Level.

## Methodology

### 1. Data Preprocessing and Exploratory Analysis

- Filtered records to focus on adults aged 18 and above.
- Examined categorical variables, numerical distributions, and data completeness.
- Excluded smoking history because of the substantial proportion of records marked `No Info`.
- Created age-group features and encoded categorical variables.
- Analyzed class imbalance and relationships between health indicators and diabetes.
- Used histograms, boxplots, violin plots, correlation heatmaps, and profile visualizations.

### 2. Patient Segmentation — Unsupervised Learning

Applied **K-Means Clustering** using three RFM-inspired features:

- **Recency:** Patient age
- **Frequency:** Number of recorded comorbidities (hypertension and heart disease)
- **Monetary:** Composite metabolic indicator combining BMI, HbA1c, and blood glucose

StandardScaler was used before clustering, while the Elbow Method and Silhouette Score were examined to select the number of clusters.

Three clusters were selected to balance segmentation simplicity and interpretability.

**Observed diabetes prevalence across clusters:**

| Cluster | Diabetes Prevalence |
|---|---:|
| Cluster 0 | 1.78% |
| Cluster 1 | 14.40% |
| Cluster 2 | 28.54% |

PCA was used for two-dimensional cluster visualization, and Hierarchical Clustering with Ward linkage was explored as an additional structural comparison.

### 3. Association Rule Mining — Apriori

Applied the Apriori algorithm to discover frequently co-occurring health indicators, including:

- High blood glucose
- High HbA1c
- Obesity
- Older age
- Hypertension
- Heart disease

Association rules were evaluated using **Support, Confidence, and Lift**, with minimum support of 5% and confidence of 50%.

The analysis identified associations involving older age, hypertension, and elevated metabolic indicators.

### 4. Diabetes Classification — Supervised Learning

Implemented and compared seven machine learning configurations:

1. Linear Discriminant Analysis (LDA)
2. Logistic Regression
3. Support Vector Machine (RBF Kernel)
4. Decision Tree
5. Depth-Constrained CART Decision Tree
6. Random Forest
7. Multi-Layer Perceptron (MLP)

An **80:20 train-test split** was used. SMOTE was applied to training data to address class imbalance, and feature scaling was used in the modeling workflow.

Models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Model Performance

The saved experiment produced the following results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| MLP | 0.912 | 0.540 | 0.822 | 0.652 | **0.965** |
| Random Forest | 0.912 | 0.539 | 0.839 | 0.656 | 0.961 |
| CART | 0.871 | 0.430 | 0.865 | 0.575 | 0.953 |
| SVM | 0.884 | 0.458 | **0.869** | 0.600 | 0.952 |
| Logistic Regression | 0.884 | 0.457 | 0.836 | 0.591 | 0.948 |
| LDA | 0.887 | 0.466 | 0.830 | 0.597 | 0.946 |
| Decision Tree | **0.940** | **0.691** | 0.721 | **0.706** | 0.845 |

### Key Findings

- **MLP achieved the highest recorded ROC-AUC of 0.965**, indicating strong discrimination between the two diabetes classes.
- **SVM achieved the highest recall of approximately 86.9%**, identifying the greatest proportion of diabetic records among the compared configurations.
- **Decision Tree achieved the highest accuracy and F1-score** at its evaluated decision threshold.
- Blood glucose and HbA1c showed the strongest associations with the recorded diabetes status.
- K-Means segmentation identified patient groups with substantially different observed diabetes prevalence.
- Apriori revealed frequently co-occurring health indicators.

## Technologies Used

**Programming:** Python

**Data Analysis:** Pandas, NumPy

**Visualization:** Matplotlib, Seaborn

**Machine Learning:** Scikit-learn, Imbalanced-learn

**Association Rule Mining:** MLxtend

**Scientific Computing:** SciPy

**Development Environment:** Jupyter Notebook / Google Colab

## Running the Project

1. Clone or download this repository.
2. Install the required Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `mlxtend`, and `scipy`.
3. Download the dataset from Kaggle if it is not included in the repository.
4. Open `Data_Mining.ipynb` in Jupyter Notebook or Google Colab.
5. Update the dataset file path to the local location of `diabetes_prediction_dataset.csv`.
6. Ensure `DecisionTreeClassifier` is imported from `sklearn.tree` before running the model-comparison section.
7. Execute the notebook to explore the preprocessing, visualizations, clustering, association rules, and classification experiments.

## Limitations and Future Improvements

This is an exploratory data-mining project rather than a clinically validated diagnostic tool.

Important limitations and possible improvements include:

- Resolving duplicate-record handling and train-test overlap.
- Correcting age-group boundaries and simplifying the preprocessing pipeline.
- Improving the handling of categorical variables during oversampling.
- Comparing SMOTE with class-weighted approaches.
- Applying stratified cross-validation and systematic hyperparameter tuning.
- Evaluating Precision-Recall AUC and probability calibration.
- Testing model performance on independent datasets.
- Revalidating the clustering results and their clinical interpretations.

The classification models predict the **diabetes status recorded in the dataset**, not whether an individual will develop diabetes in the future.

## Project Files

- `Data_Mining.ipynb` — Python analysis, visualizations, clustering, association rules, and model comparison.
- `Data_MIning_Report.pdf` — Project report describing methods and findings.
- `diabetes_prediction_dataset.csv` — Dataset used in the analysis, if included in the repository.

**Note:** The notebook contains exploratory code and saved experiment outputs. Some narrative figures in the accompanying report differ from the notebook and require reconciliation. The performance figures above describe the saved experiment and should be regenerated after correcting the preprocessing and execution issues.

---

**Project Area:** Data Science | Machine Learning | Healthcare Analytics | Data Mining
