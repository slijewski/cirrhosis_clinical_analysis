# Cirrhosis Clinical Analysis & Predictive Prognostics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

## Overview

**Primary biliary cirrhosis (PBC)** is a chronic liver disease associated with significant morbidity and mortality. This repository presents a comprehensive clinical data science project focused on the survival analysis and mortality risk prediction of patients with (PBC). Leveraging the landmark Mayo Clinic study dataset, the project integrates traditional biostatistical methods (Cox Proportional Hazards) with modern machine learning frameworks to provide a dual-perspective analysis of patient outcomes.

## 🧬 Scientific Methodology

### 1. Data Characterization & Preprocessing

The dataset comprises clinical and biochemical parameters from 418 patients, including laboratory measurements,
clinical symptoms, and survival outcomes.

Source: Kaggle – Cirrhosis Prediction Dataset (Mayo Clinic). The preprocessing pipeline ensures high-fidelity input for both statistical and ML models.

* **Temporal Normalization**: Patient age, originally recorded in days, is transformed into decimal years (Age/365.25) to align with standard clinical intervals.
* **Target Binarization**: For classification tasks, the outcome is binarized to represent mortality status ('D'). Statuses such as 'CL' (transplant) are treated as censored observations to maintain the integrity of the mortality prediction.
* **Robust Feature Engineering**:
  * **Numerical Features**: Median imputation is employed to handle missing values, mitigating the influence of extreme outliers common in cirrhosis biochemical markers (e.g., Bilirubin). Features undergo **Z-score standardization**, ensuring that variables with varying magnitudes (e.g., Platelet count vs. Bilirubin mg/dL) contribute equally to model convergence.
  * **Categorical Encoding**: Clinical signs such as Spiders, Hepatomegaly, and Ascites are transformed via One-Hot Encoding for mathematical compatibility.

### 2. Analytical Framework

#### Machine Learning Classification

We implement a comparative analysis between **Logistic Regression (LR)** and **Random Forest (RF)**:

* **Logistic Regression**: Serves as a baseline model to capture linear relationships between clinical markers and mortality risk.
* **Random Forest Classifier**: Utilized for its ability to capture high-order non-linear interactions (e.g., the synergistic effect of advanced age and low Albumin levels).
* **Performance Metrics**: The models are evaluated using the **Area Under the ROC Curve (AUC-ROC)**. An AUC of **0.87+** indicates high discriminative power in distinguishing between survivor and non-survivor cohorts.

#### Explainable AI (XAI) with SHAP

To bridge the "black-box" gap, we utilize **SHAP (SHapley Additive exPlanations)** values. This allows for:

* **Global Interpretability**: Identifying that **Bilirubin** and **Albumin** are the most significant clinical drivers across the entire population.
* **Clinical Coherence**: Validating that low Albumin (a markers of synthetic liver failure) and high Bilirubin (a marker of excretory dysfunction) correctly correlate with increased risk.

#### Survival Analysis (Cox Proportional Hazards)

Unlike traditional classification, the Cox model addresses the *rate* of event occurrence:

* **Hazard Ratios (HR)**: We calculate the exponentiated coefficients (exp(beta)).
  * An $HR > 1$ (e.g., Bilirubin) signifies an increased risk of mortality per unit increase.
  * An $HR < 1$ (e.g., Albumin) signifies a protective effect.
* **Statistical Significance**: All findings are validated using p-values (p < 0.05) to ensure results are not due to chance.

## 🖥️ Clinical Risk Calculator (Streamlit App)

The associated application (located in `app/app.py`) provides an interactive interface for clinicians and researchers to input patient parameters and obtain a real-time risk estimation based on the trained Random Forest pipeline.

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or higher
* Pip package manager

### Installation

1. Install dependencies:

   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

### Usage

To launch the interactive risk prediction tool:

```bash
# Using uv (recommended)
uv run streamlit run app/app.py

# Or using streamlit directly
streamlit run app/app.py
```

## 📓 Notebook Exploratory Analysis Summary

The core logic of this project is documented in [01_cirrhosis_analysis.ipynb](notebooks/01_cirrhosis_analysis.ipynb), covering the full data science lifecycle from cleaning to advanced modeling.

### Key Performance Metrics

* **Predictive Accuracy**: Both Logistic Regression and Random Forest achieved an **AUC-ROC of ~0.87**, showing high precision in identifying high-risk patients.
* **Survival Modeling**: The Cox Proportional Hazards model provided statistically significant (p < 0.05) coefficients.

### Primary Clinical Conclusions

* **High-Risk Indicators**: Elevated **Bilirubin**, **Copper**, and prolonged **Prothrombin time** are the strongest predictors of decreased survival probability.
* **Protective Factors**: Consistent with clinical literature, higher **Albumin** levels and **Platelet counts** correlate with significantly improved patient outcomes.
* **Interactions**: SHAP analysis reveals that advanced **Age** and higher **Disease Stage** significantly amplify the risk associated with biochemical imbalances.

## 📁 Repository Structure

```text
├── .python-version     # Python version pin (uv)
├── app/                # Streamlit application source
├── data/               # Clinical datasets (CSV)
├── models/             # Serialized ML pipelines (.pkl)
├── notebooks/          # Exploratory Data Analysis & Model Training
├── requirements.txt    # Project dependencies
├── uv.lock           # Lockfile for reproducible environment
└── src/                # Utility scripts and preprocessing modules
```

## 📃 Key Results

* Elevated bilirubin and reduced albumin were strongly associated with increased mortality risk
* Random Forest improved discrimination compared to logistic regression
* Cox regression identified bilirubin, albumin, and ascites as significant survival predictors

### Clinical Interpretation

Results align with known clinical risk factors in PBC and demonstrate how explainable ML can support
risk stratification while maintaining interpretability required in medical applications.

### Limitations

* Relatively small sample size
* Observational data with potential confounding
* No external validation cohort

### Future Work

* External validation on independent cohorts
* Time-dependent covariates in survival modeling
* Calibration and decision curve analysis
* Deployment as a clinical decision support prototype

## 📜 Acknowledgments

Data derived from the Mayo Clinic Trial in Primary Biliary Cirrhosis (1974-1984). This project is intended for **educational and research purposes only** and should not be used as a substitute for professional medical advice or clinical decision-making.

---

## 👨‍🔬Author

Sebastian Lijewski
PhD in Pharmaceutical Sciences
