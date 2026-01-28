# Customer Churn Prediction

## Overview

This repository contains a **telecom customer churn prediction project** built using Python and classical machine learning techniques. The project includes **data analysis notebooks** and a **Streamlit-based UI application** for interactive churn prediction and visualization.

---

## Repository Structure

```
CustomerChurn/
│
├── Customer_Churn.csv
├── project_with_UI.py
├── project_with_Visuals - manual.ipynb
├── project_with_Visuals - sklearn.ipynb
├── requirements.txt
└── README.md
```

---

## Dataset

* **File**: `Customer_Churn.csv`
* **Type**: Structured tabular data
* **Target column**: `Churn`

The dataset contains customer-level attributes related to service usage, account information, and demographics. The objective is to predict whether a customer will churn (`Yes` / `No`).

---

## Notebooks

### 1. `project_with_Visuals - manual.ipynb`

* Exploratory Data Analysis (EDA)
* Manual preprocessing and feature handling
* Visual analysis of churn patterns
* Understanding relationships between features and churn

### 2. `project_with_Visuals - sklearn.ipynb`

* Data preprocessing using scikit-learn utilities
* Model training using classical ML algorithms
* Model evaluation using standard classification metrics

These notebooks are intended for **analysis, experimentation, and learning**, not production deployment.

---

## Streamlit Application

### `project_with_UI.py`

This script provides an **interactive Streamlit UI** that allows users to:

* Input customer details
* Run churn prediction using a trained model
* Visualize insights related to churn

This is the main entry point if you want to *run the project as an application* rather than as notebooks.

---

## Installation

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt` should be generated from actual imports used in the notebooks and UI script.

---

## How to Run

### Run Streamlit App

```bash
streamlit run project_with_UI.py
```

Open either of the `.ipynb` files to explore data and models.

---

## Modeling Approach (High Level)

* Data cleaning and preprocessing
* Encoding categorical features
* Training supervised classification models
* Evaluating performance using accuracy and related metrics

Exact models and parameters are documented **inside the notebooks**, not duplicated here.

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn
* Streamlit
* Jupyter Notebook

---

## Results

<img width="2560" height="1440" alt="Screenshot (56)" src="https://github.com/user-attachments/assets/f346d9f4-9381-44f7-8ef7-d8baaaf748eb" />


This project is intended for learning and portfolio demonstration purposes.
