# Customer Churn Prediction using Machine Learning

## Table of Contents
- [Project Overview](#project-overview)
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [How to Run This Project](#how-to-run-this-project)
- [Interactive Dashboard Preview](#interactive-dashboard-preview)

---

## Project Overview

This project is an end-to-end data science solution aimed at predicting customer churn for a telecommunications company. By analyzing historical customer data, we can identify key factors that contribute to churn and build a machine learning model to predict which customers are at high risk of leaving. The final output is an interactive web-based dashboard that allows business users to get live churn predictions.

---

## Project Goal

The primary objective is to build a machine learning model that achieves at least **80% accuracy** in predicting customer churn. This tool is designed to help the business proactively engage with at-risk customers to improve retention rates and reduce revenue loss.

---

## Dataset

The project utilizes the **Telco Customer Churn** dataset, which is publicly available on Kaggle. It contains **7,043 customer records** with **21 distinct features** providing information on:
- **Customer Demographics:** Gender, Senior Citizen, Partner, Dependents.
- **Subscribed Services:** Phone Service, Internet Service, Online Security, etc.
- **Account Information:** Tenure, Contract type, Payment Method, Monthly Charges, Total Charges.
- **Target Variable:** `Churn` (Yes/No).

---

## Key Features

- **Data Cleaning & Preprocessing:** Handles missing values, corrects data types, and encodes categorical features for modeling.
- **Exploratory Data Analysis (EDA):** Visualizations to uncover insights and patterns related to customer churn.
- **Machine Learning Modeling:** Trains and evaluates three different classification models (Logistic Regression, Random Forest, XGBoost) to find the best performer.
- **Interactive Dashboard:** A user-friendly web interface built with Streamlit that displays KPIs, visualizations, and a live prediction system.

---

## Methodology

The project follows a structured data science workflow:

1.  **Data Loading and Initial Inspection:** The dataset is loaded into a Pandas DataFrame for initial analysis.
2.  **Data Preprocessing:**
    - `TotalCharges` column converted from object to numeric type.
    - Missing values in `TotalCharges` are imputed using the median.
    - Categorical features are converted into a machine-readable format using one-hot encoding.
3.  **Exploratory Data Analysis (EDA):**
    - Analysis of the churn rate (approx. 26.5%).
    - Visualizations reveal that customers with **month-to-month contracts**, **higher monthly charges**, and **lower tenure** are significantly more likely to churn.
4.  **Model Training and Evaluation:**
    - The data is split into 80% for training and 20% for testing.
    - Three models are trained and their performance is evaluated based on Accuracy, Precision, Recall, and F1-Score.
5.  **Model Deployment:**
    - The best-performing model (Logistic Regression) and its required column structure are saved using `joblib`.
    - An interactive dashboard is created with Streamlit to serve the model for live predictions.

---

## Model Performance

The models were evaluated on the unseen test set, with the following accuracy scores:

| Model | Accuracy Score | Met Objective (≥80%) |
| :--- | :---: | :---: |
| **Logistic Regression** | **82.11%** | ✔️ **Yes** |
| Random Forest | 78.92% | ❌ No |
| XGBoost | 78.92% | ❌ No |

**Logistic Regression** was selected as the final model due to its superior accuracy and high interpretability.

---

## Technologies Used

- **Programming Language:** Python 3.8+
- **Core Libraries:**
  - `pandas` for data manipulation
  - `numpy` for numerical operations
  - `scikit-learn` for machine learning
  - `xgboost` for the XGBoost model
- **Visualization:**
  - `matplotlib`
  - `seaborn`
- **Dashboard & Deployment:**
  - `streamlit` for the interactive web app
  - `joblib` for saving and loading the model

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # The raw dataset
├── churn_analysis.ipynb                   # Jupyter Notebook with all analysis and model training
├── dashboard.py                           # The Streamlit dashboard application script
├── churn_model.pkl                        # Saved Logistic Regression model
├── model_columns.pkl                      # Saved list of model feature columns
└── README.md                              # This README file
```

---

## How to Run This Project

Follow these steps to set up and run the project on your local machine.

**1. Clone the Repository:**
```bash
git clone https://github.com/<your-username>/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

**2. Create a Virtual Environment (Recommended):**
```bash
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

**3. Install Dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter xgboost streamlit joblib
```

**4. Run the Jupyter Notebook:**
First, run all the cells in the `churn_analysis.ipynb` notebook. This will train the models and generate the `churn_model.pkl` and `model_columns.pkl` files required by the dashboard.
```bash
jupyter notebook churn_analysis.ipynb
```

**5. Launch the Streamlit Dashboard:**
Once the model files are created, run the following command in your terminal:
```bash
python -m streamlit run dashboard.py
```
The application will open in a new tab in your web browser.

---

## Interactive Dashboard Preview

Here is a preview of the live prediction system on the dashboard:
https://ibb.co/zVVnSGsH
![Dashboard Preview](https://ibb.co/zVVnSGsH)

The dashboard allows users to input customer details and receive an instant prediction on whether the customer is likely to churn, along with a confidence score.
