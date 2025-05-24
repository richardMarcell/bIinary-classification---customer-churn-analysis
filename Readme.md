# 🧠 Customer Churn Prediction API

This project provides a **RESTful API** built with **Flask** for predicting customer churn using a **trained Random Forest machine learning model**. It takes customer profile data as input and returns a prediction indicating whether the customer is likely to churn, along with the probability.

---

## 🚀 Project Overview

Customer churn is a critical metric for any business. With this model, companies can **proactively identify customers at risk** of leaving and take actions to improve retention.

This API allows you to send customer data (e.g., credit score, balance, age, etc.) and receive a prediction of churn risk in real-time.

---

## 🛠️ Features

- 🔍 Input validation with clear error responses
- 🧪 Machine learning model (Random Forest) trained and deployed
- ⚖️ Scaler for numerical normalization
- 🧬 Label encoders for categorical variables (`Gender`, `Geography`)
- 🔁 Probability output (`predict_proba`) included
- 📦 Ready-to-deploy Flask application

---

## 🧰 Tech Stack

- Python 3.x
- Flask
- scikit-learn
- Pandas & NumPy
- joblib (for model serialization)

---

## 📦 Model Artifacts

Ensure the following `.joblib` files are in the same directory as `app.py`:

- `random_forest.joblib` – Trained Random Forest model
- `scaler.joblib` – Fitted Scaler for numerical features
- `Gender_encoder.joblib` – Label Encoder for Gender
- `Geography_encoder.joblib` – Label Encoder for Geography

---

## 🧪 Input Format

The API accepts a POST request with JSON input containing the following fields:

```json
{
  "CreditScore": 600,
  "Geography": "France",
  "Gender": "Male",
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000.0,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000.0
}
```

## 📤 Output Format

The API will respond with a JSON object like this:

```json
{
  "status": "success",
  "message": "Success to predict customer churn",
  "prediction": 1,
  "probability_churn": 0.837
}
```
