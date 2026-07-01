# Comprehensive Loan Approval Prediction System 

An end-to-end Machine Learning project developed to automate and optimize the loan underwriting process for financial institutions using predictive analytics. 

## Project Evolution & Context (Why this is unique!)
This project has been developed in two distinct phases, showcasing iterative improvement and model optimization:
1. **Phase 1 (Internship Baseline):** Initially developed during a data science internship using **Logistic Regression** and **K-Nearest Neighbors (KNN)**.
2. **Phase 2 (SMIT Assignment Advanced Upgrade - Current):** Upgraded as part of the **Saylani Mass IT Training (SMIT)** curriculum under the guidance of **Engr. Nayab**. In this phase, the low-performing KNN model was completely removed and replaced with robust tree-based ensemble methods: **Decision Tree** and **Random Forest Classifiers**, significantly improving the system's business utility and error minimization.

## Project Objective
Manual evaluation of loan applications is time-consuming, prone to human error, and vulnerable to subjective bias. This project builds a data-driven classification system that utilizes applicant demographic markers and financial metrics to predict whether a loan should be `Approved (1)` or `Rejected (0)`.

---

## Dataset Architecture
The model is trained on historical loan data containing **614 rows** and **13 features**:
* **Demographic Features:** `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`.
* **Financial Features:** `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`.
* **Target Variable:** `Loan_Status` (Y: Approved, N: Rejected).


## Data Preprocessing & Engineering Pipeline
To ensure high-quality training and eliminate bias, a strict preprocessing pipeline was executed:
* **Imputation:** Categorical missing values filled via **Mode**; numerical values filled via **Median** to protect against outlier distortion.
* **Encoding:** Binary variables processed via **Label Encoding**; multi-class variables handled using **One-Hot Encoding**.
* **Scaling:** Numerical columns (`ApplicantIncome`, `LoanAmount`) normalized using `StandardScaler`.
* **Class Imbalancing:** Addressed severe class imbalance by applying **SMOTE (Synthetic Minority Over-sampling Technique)** strictly on the training dataset to ensure unbiased minority predictions.

## Exploratory Data Analysis (EDA) Highlights
* **Credit History Gating:** Applicants with a positive `Credit_History` (1.0) showed an exponentially higher likelihood of approval, making it the most critical feature.
* **Income Distribution:** Right-skewed distribution indicating a majority of applicants fall into lower-to-middle income brackets.
* **Feature Correlation:** A strong positive correlation was noted between `ApplicantIncome` and `LoanAmount`, proving risk tolerance scales with income.

## Machine Learning Models & Performance Benchmark

The system evaluates three distinct models on a 30% unseen testing pool (123 samples):

| Classifier Model        | Overall Accuracy | Class 1 (Approved) F1-Score | Class 1 Recall | False Negatives (Critical Loss) |
|-------------------------|------------------|-----------------------------|----------------|---------------------------------|
| **Logistic Regression** | **76% (Best)**   | **0.83 (Best)**             | **90% (Best)** | **8 (Lowest)**                  |
| **Random Forest**       | 71%              | 0.79                        | 85%            | 12                              |
| **Decision Tree**       | 70%              | 0.77                        | 78%            | 18                              |

### Model Performance Insights:
* **Confusion Matrix Breakdown:** Logistic Regression successfully captured **72 out of 80** actual approved cases, minimizing False Negatives to just 8. 
* **The Tree-Based Approach:** While Random Forest and Decision Tree provided deep insights into non-linear splits, the linear decision boundaries of **Logistic Regression** yielded the most stable and commercially viable results for this dataset.

## Final Conclusion & Recommendation
**Logistic Regression** is selected as the champion model for production deployment. From a business perspective, it drastically reduces the risk of rejecting reliable clients (minimizing False Negatives), allowing financial institutions to safely automate workflows, reduce manual processing times, and scale credit lending securely.

## Project Repository Structure & Deliverables

This repository contains all the mandatory deliverables for the SMIT Assignment evaluation:
1. **`Loan_Approval_Prediction.ipynb`**: Complete Jupyter Notebook containing source code, data preprocessing pipelines (SMOTE, Scaling), EDA visualizations, and model training scripts.
2. **`Loan_Approval_Prediction_Report.pdf`**: A formal, comprehensive 2-3 page documentation report formatted according to the SMIT assessment criteria. *(Contains Executive Summary, Detailed Feature Matrix, Model Performance Tables, and Business Decisions).*

> **Note to Evaluator:** The PDF report has been uploaded directly to the root directory of this repository for quick review alongside the execution notebook.
