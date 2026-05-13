<div align="center">

# Credit Scoring Model

### Predict Loan Default Risk Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**CodeAlpha Internship - Task 1**

---

*An end-to-end machine learning project that predicts whether a loan applicant will default, with a clean Jupyter notebook pipeline and a Streamlit web app for real-time predictions.*

</div>

---

## Project Overview

This project tackles the credit risk classification problem — predicting whether a borrower will default on a loan.

| Stage | Description |
|-------|-------------|
| EDA | Distributions, correlations, target balance |
| Preprocessing | Missing value imputation, outlier removal, label encoding |
| Feature Engineering | 3 custom features: `income_to_loan`, `age_emp_ratio`, `risk_score` |
| Model Training | Logistic Regression, Decision Tree, Random Forest |
| Evaluation | Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices |
| Deployment | Streamlit web app with single and batch prediction |

---

## Project Structure

```
credit_scoring_model/
├── credit_scoring.ipynb     # Full analysis notebook (EDA, training, evaluation)
├── app.py                   # Streamlit deployment app
├── requirements.txt         # Python dependencies
├── credit_risk_dataset.csv  # Dataset (from Kaggle)
├── saved_models/            # Trained model artifacts
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── encoders.pkl
├── .gitignore
├── LICENSE
└── README.md
```

---

## Dataset

**Source:** [Credit Risk Dataset - Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

| Feature | Description |
|---------|-------------|
| `person_age` | Age of the applicant |
| `person_income` | Annual income |
| `person_home_ownership` | Rent / Own / Mortgage / Other |
| `person_emp_length` | Employment length (years) |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade (A-G) |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_percent_income` | Loan as % of income |
| `cb_person_default_on_file` | Historical default (Y/N) |
| `cb_person_cred_hist_length` | Credit history length |
| **`loan_status`** | **Target - 0: Non-Default, 1: Default** |

---

## Models and Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 0.85 | 0.72 | 0.52 | 0.60 | 0.91 |
| Decision Tree | 0.88 | 0.76 | 0.67 | 0.71 | 0.83 |
| **Random Forest** | **0.93** | **0.88** | **0.73** | **0.80** | **0.97** |

> Best Model: Random Forest — selected by highest F1-Score and ROC-AUC.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/kareemaboalnoor/credit-scoring-model.git
cd credit-scoring-model
pip install -r requirements.txt
```

### 2. Run the notebook

Open `credit_scoring.ipynb` in Jupyter and run all cells to train the model.

### 3. Launch the app

```bash
python app.py
```

The app will open at `http://localhost:8501`.

---

## Tech Stack

| Technology | Purpose |
|:---:|:---:|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| scikit-learn | Machine learning |
| Matplotlib / Seaborn | Visualizations |
| Streamlit | Web deployment |

---

## Features

- **Single Prediction** — Enter applicant details and get instant risk assessment
- **Batch Upload** — Upload a CSV to predict risk for multiple applicants
- **Model Dashboard** — View performance metrics and visualizations
- **Clean UI** — Dark theme with gradient design

---

## License

This project is open source under the [MIT License](LICENSE).

---

<div align="center">

**Built by [Kareem Aboalnoor](https://github.com/kareemaboalnoor) | CodeAlpha Internship**

</div>
