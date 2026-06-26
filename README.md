# 🏥 Federated and Explainable Learning for Predicting Antenatal Care Quality in Bangladesh

> **Using BDHS 2022 Data** | Presented at **BECITHCON 2026** — Track: *Big Data Analytics for Health Care*
>
> **Author:** Md. Sohag Hossain

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Research Objectives](#-research-objectives)
3. [Dataset](#-dataset)
4. [Target Variable — ANC Quality](#-target-variable--anc-quality)
5. [Features Used](#-features-used)
6. [Methodology](#-methodology)
7. [Project Structure (Notebook Parts)](#-project-structure-notebook-parts)
   - [Part 1 — Install Dependencies](#part-1--install-dependencies)
   - [Part 2 — Imports & Configuration](#part-2--imports--configuration)
   - [Part 3 — Mount Drive & Load Data](#part-3--mount-drive--load-data)
   - [Part 4 — Feature Selection](#part-4--feature-selection)
   - [Part 5 — Build ANC Quality Target Variable](#part-5--build-anc-quality-target-variable)
   - [Part 6 — Exploratory Data Analysis (EDA)](#part-6--exploratory-data-analysis-eda)
   - [Part 7 — Data Preprocessing](#part-7--data-preprocessing)
   - [Part 8 — Centralized Baseline Models](#part-8--centralized-baseline-models)
   - [Part 9 — Federated Learning: FedAvg](#part-9--federated-learning-fedavg)
   - [Part 10 — Federated Learning: FedProx](#part-10--federated-learning-fedprox)
   - [Part 11 — Comprehensive Model Comparison & Visualisation](#part-11--comprehensive-model-comparison--visualisation)
   - [Part 12 — FL Convergence Analysis](#part-12--fl-convergence-analysis)
   - [Part 13 — SHAP Explainability: Global](#part-13--shap-explainability-global)
   - [Part 14 — SHAP: Local Explanations & Dependence Plots](#part-14--shap-local-explanations--dependence-plots)
   - [Part 15 — Division-wise SHAP Analysis](#part-15--division-wise-shap-analysis)
   - [Part 16 — Confusion Matrices (All Models)](#part-16--confusion-matrices-all-models)
   - [Part 17 — Privacy Analysis & Communication Cost](#part-17--privacy-analysis--communication-cost)
   - [Part 18 — Final Results Table & Paper-Ready Summary](#part-18--final-results-table--paper-ready-summary)
   - [Part 19 — Save All Results to Excel](#part-19--save-all-results-to-excel)
8. [Installation & Setup](#-installation--setup)
9. [How to Run](#-how-to-run)
10. [Key Results](#-key-results)
11. [Technologies Used](#-technologies-used)
12. [Citation](#-citation)

---

## 🔍 Project Overview

This project applies **Federated Learning (FL)** and **Explainable AI (XAI)** to predict the quality of **Antenatal Care (ANC)** received by pregnant women across Bangladesh's 8 administrative divisions, using the **Bangladesh Demographic and Health Survey (BDHS) 2022** dataset.

The core idea is to simulate a privacy-preserving federated setup where each division acts as an independent client — training local models without sharing raw data — while still achieving competitive predictive performance compared to centralized approaches.

---

## 🎯 Research Objectives

1. Predict ANC quality (**High vs Low**) using BDHS 2022 data with optimized ML models
2. Implement privacy-preserving Federated Learning (**FedAvg**, **FedProx**) across 8 divisions
3. Provide explainable predictions using **SHAP** (global + local + regional)
4. Demonstrate that federated models **match or exceed centralized performance** while preserving data privacy

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | Bangladesh Demographic and Health Survey (BDHS) 2022 |
| **File** | `BDIR81FL.DTA` (Stata format) |
| **Population** | Women who received ≥ 1 ANC visit |
| **Geographic Scope** | 8 Divisions: Barisal, Chittagong, Dhaka, Khulna, Mymensingh, Rajshahi, Rangpur, Sylhet |

> ⚠️ The raw data file (`BDIR81FL.DTA`) is **not included** in this repository due to data access restrictions. It can be downloaded from the [DHS Program website](https://dhsprogram.com/) after registration.

---

## 🎯 Target Variable — ANC Quality

ANC Quality is constructed from **5 WHO-recommended indicators**, with a threshold of **≥ 4 out of 5** to classify as *High Quality ANC*.

| # | Component | BDHS Variable | Criterion |
|---|---|---|---|
| 1 | ANC Visit | `v238` | ≥ 1 visit |
| 2 | 1st Trimester Attendance | `m13_1` | ≤ 3 months gestation |
| 3 | Blood Pressure Checked | `m42a_1` | = 1 (Yes) |
| 4 | Iron Supplement Given | `m45_1` | = 1 (Yes) |
| 5 | Tetanus Toxoid Injection | `m1_1` | ≥ 1 injection |

> **Note:** `m43_1` (urine), `m44_1` (blood sample), and `m56_1` (complication counselling) are absent from the BDHS 2022 IR file. Variables `v467a` (permission) and `v467e` (alone) are 100% missing in the ANC sub-sample and are automatically dropped during preprocessing.

---

## 🧾 Features Used

Features are grouped into four categories:

**Sociodemographic** — age, education years/level, literacy, marital status, parity, birth order, wanted pregnancy, decision-making on healthcare, partner education & occupation

**Wealth & Household** — wealth index, urban/rural residence, division, electricity access, toilet type, water source, household members

**Media & Awareness** — TV access, radio access, mobile phone ownership, health insurance

**Healthcare Access Barriers** — distance problem, money problem, transport problem

**ANC & Delivery History** — ANC provider type, ANC place, delivery place, skilled birth attendant

---

## 🧠 Methodology

```
BDHS 2022 Data
      │
      ▼
Feature Engineering + ANC Quality Scoring
      │
      ▼
Preprocessing (Imputation, SMOTE, Scaling)
      │
      ├──────────────────────────────────────────────┐
      ▼                                              ▼
Centralized Models                        Federated Learning (8 Divisions)
(LR, RF, XGBoost, LightGBM)             (FedAvg & FedProx algorithms)
      │                                              │
      └──────────────────┬───────────────────────────┘
                         ▼
              Model Comparison & Evaluation
                (Accuracy, AUC, F1, etc.)
                         │
                         ▼
              SHAP Explainability
         (Global + Local + Division-wise)
```

---

## 📁 Project Structure (Notebook Parts)

### Part 1 — Install Dependencies
Installs all required Python packages: `shap`, `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`, `matplotlib`, `seaborn`, `openpyxl`, `pyreadstat`.

### Part 2 — Imports & Configuration
Sets up all library imports, global constants (random seed = 42), and the division code mapping for Bangladesh's 8 administrative divisions.

### Part 3 — Mount Drive & Load Data
Mounts Google Drive (when running in Colab) and loads the BDHS 2022 `.DTA` file using `pyreadstat`. Prints dataset shape and memory usage.

### Part 4 — Feature Selection
Defines the ANC quality scoring columns and the full set of predictor feature columns with their corresponding BDHS variable codes. Filters the raw dataframe to only the required columns.

### Part 5 — Build ANC Quality Target Variable
Constructs the 5 binary ANC quality components, calculates the composite ANC quality score, and applies the ≥ 4/5 threshold to create the binary outcome variable (`anc_quality`). Restricts the analysis sample to women with ≥ 1 ANC visit.

### Part 6 — Exploratory Data Analysis (EDA)
Produces a 2×3 panel of visualizations:
- Target distribution (High vs Low ANC quality)
- ANC quality score distribution
- High quality rate by Division
- Rate by Wealth Index
- Rate by Urban/Rural Residence
- Component satisfaction rates

### Part 7 — Data Preprocessing
- Renames columns from BDHS codes to readable labels
- Reports and handles missing values (drops columns ≥ 90% missing)
- Applies median imputation for remaining missingness
- Performs stratified 80/20 train-test split
- Applies **SMOTE** to balance the training set (minority class: ~36%)

### Part 8 — Centralized Baseline Models
Trains and evaluates four centralized ML models on the full dataset:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Outputs: Accuracy, AUC-ROC, F1-Score, Precision, Recall.

### Part 9 — Federated Learning: FedAvg
Implements the **Federated Averaging (FedAvg)** algorithm across 8 division-level clients. Each client trains locally on its division's data; model weights are aggregated at a central server. Simulates realistic FL communication rounds.

### Part 10 — Federated Learning: FedProx
Implements **FedProx**, an extension of FedAvg that adds a proximal regularization term (μ) to handle statistical heterogeneity across divisions. Particularly useful given the non-IID nature of health data across divisions.

### Part 11 — Comprehensive Model Comparison & Visualisation
Side-by-side performance comparison of all models (centralized + federated) using bar charts and summary tables across all evaluation metrics.

### Part 12 — FL Convergence Analysis
Plots training loss and accuracy curves across federated communication rounds for both FedAvg and FedProx, illustrating convergence behavior.

### Part 13 — SHAP Explainability: Global
Computes and visualizes global SHAP values:
- Summary (beeswarm) plot
- Feature importance bar chart
- Identifies the most influential predictors of ANC quality at the national level

### Part 14 — SHAP: Local Explanations & Dependence Plots
Generates individual-level SHAP waterfall plots (explaining single predictions) and SHAP dependence plots showing how key features interact with the model output.

### Part 15 — Division-wise SHAP Analysis
Computes SHAP values separately for each of the 8 divisions, revealing regional variation in the factors driving ANC quality predictions.

### Part 16 — Confusion Matrices (All Models)
Plots confusion matrices for every trained model (centralized and federated) to visualize classification performance across High and Low ANC quality classes.

### Part 17 — Privacy Analysis & Communication Cost
Quantifies the privacy-utility tradeoff and estimates communication overhead (bytes transmitted per round, total rounds) for the federated approaches.

### Part 18 — Final Results Table & Paper-Ready Summary
Generates a clean, publication-ready summary table of all model results suitable for inclusion in the conference paper.

### Part 19 — Save All Results to Excel
Exports all results — model metrics, SHAP values, division-wise summaries — to a structured `.xlsx` file for further analysis and reporting.

---

## ⚙️ Installation & Setup

### Requirements

- Python 3.8+
- Google Colab (recommended) or local Jupyter environment

### Install Dependencies

```bash
pip install shap scikit-learn xgboost lightgbm imbalanced-learn \
            matplotlib seaborn openpyxl pyreadstat
```

---

## ▶️ How to Run

1. **Get the data** — Download `BDIR81FL.DTA` from the [DHS Program](https://dhsprogram.com/) and upload it to your Google Drive.

2. **Open the notebook** — Upload `ANC_Federated_V4_Final.ipynb` to Google Colab.

3. **Update the file path** — In **Part 3**, set the correct path to your data file:
   ```python
   file_path = '/content/drive/MyDrive/YOUR_FOLDER/BDIR81FL.DTA'
   ```

4. **Run all cells** — Execute the notebook top to bottom (`Runtime → Run all`).

5. **Outputs** — Results will be saved as an Excel file and all figures will be displayed inline.

---

## 📈 Key Results

- Federated Learning (FedAvg & FedProx) across 8 divisions **matches or exceeds** centralized model performance
- SHAP analysis reveals **wealth index**, **education level**, and **ANC provider type** as the top national-level predictors of ANC quality
- Division-wise SHAP highlights meaningful **regional heterogeneity** in ANC quality drivers
- FedProx demonstrates improved convergence stability over FedAvg under non-IID conditions

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.x |
| **ML Frameworks** | scikit-learn, XGBoost, LightGBM |
| **Federated Learning** | Custom FedAvg & FedProx implementation |
| **Explainability** | SHAP |
| **Data Handling** | pandas, NumPy, pyreadstat |
| **Class Imbalance** | imbalanced-learn (SMOTE) |
| **Visualization** | matplotlib, seaborn |
| **Environment** | Google Colab / Jupyter Notebook |

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@inproceedings{hossain2026anc,
  author    = {Md Sohag Hossain},
  title     = {Federated Explainable Learning for Privacy-Preserving Antenatal Care Quality Prediction in Bangladesh},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Biomedical Engineering, Computer and Information Technology for Health (BECITHCON)},
  track     = {Big Data Analytics for Health Care},
  year      = {2026},
  note      = {Under Review}
}
```

---

*For questions or collaborations, feel free to open an issue or reach out via the conference contact.*
