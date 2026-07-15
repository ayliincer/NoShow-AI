# NoShow-AI
TÜBİTAK 2209-A - AI-based Patient No-Show Prediction System
# Explainable AI-Based Dynamic No-Show Prediction System

## Overview

This project presents an explainable machine learning pipeline for predicting outpatient appointment no-shows using electronic healthcare records.

The pipeline includes:

- Exploratory Data Analysis (EDA)
- Missing Value Analysis
- Duplicate Record Analysis
- Outlier Detection
- Descriptive Statistics
- Correlation Analysis
- Feature Engineering
- Missing Data Imputation
- Categorical Encoding
- Model Development
- Cross Validation
- External Validation
- Explainable Artificial Intelligence (SHAP)
- Model Serialization

The project follows a leakage-aware machine learning workflow to ensure scientific validity.

---

## Dataset

Primary Dataset

- Medical Appointment No Show Dataset

External Dataset

- KaggleV2-May-2016

---

## Machine Learning Models

The following algorithms are evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

---

## Explainable AI

Model interpretation is performed using SHAP.

Generated explanations include

- SHAP Summary Plot
- SHAP Bar Plot

---

## Project Structure

```
NoShow-AI
│
├── kodlar/
├── veriler/
├── modeller/
├── gorseller/
├── makale/
├── README.md
└── requirements.txt
```

---

## Installation

```bash
git clone <repository>

cd NoShow-AI-main

pip install -r requirements.txt
```

---

## Running

Execute the Python scripts sequentially.

```
01_veri_seti_genel_bakis.py

↓

02_eksik_veri_ve_tekrarlayan_kayitlar.py

↓

...

↓

20_modeli_diske_kaydet.py
```

---

## Output

The project produces

- Cleaned datasets
- Feature engineered datasets
- Cross-validation results
- External validation results
- SHAP visualizations
- Serialized production-ready model

---

## Author

Aylin Cer
Beyza Nur Dinçer

Management Information Systems (MIS)

Karadeniz Technical University

---

## License

This project is developed for academic research purposes.