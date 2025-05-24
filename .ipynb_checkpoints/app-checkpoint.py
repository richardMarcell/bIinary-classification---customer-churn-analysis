from flask import Flask, request, jsonify
from joblib import load
import numpy as np
import pandas as pd

# Inisiasi Aplikasi atau API
app = Flask(__name__)

# Load Model, Scaler dan Encoder
model = load("random_forest.joblib")
scaler = load("scaler.joblib")
gender_encoder = load("Gender_encoder.joblib")
geo_encoder = load("Geography_encoder.joblib")


# Kolom yang dibutuhkan Untuk Prediksi (Harus Berurutan)
feature_order = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data dari request user
        data_request = request.get_json()

         # Validasi kolom yang dibutuhkan
        missing = [col for col in feature_order if col not in data_request]
        if missing:
            return jsonify({
                "status": "error",
                "message": f"Missing input fields: {', '.join(missing)}"
            }), 400

        # Validate numeric fields
        numeric_fields = [
            'CreditScore', 'Age', 'Tenure', 'Balance',
            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        for field in numeric_fields:
            if not isinstance(data_request[field], (int, float)):
                return jsonify({
                    "status": "error",
                    "message": f"Field '{field}' must be a number (int or float)."
                }), 400

        # Validate categorical values
        allowed_gender = ['Male', 'Female']
        allowed_geo = ['France', 'Spain', 'Germany']

        if data_request['Gender'] not in allowed_gender:
            return jsonify({
                "status": "error",
                "message": f"Gender must be one of the following values: {', '.join(allowed_gender)}"
            }), 400

        if data_request['Geography'] not in allowed_geo:
            return jsonify({
                "status": "error",
                "message": f"Geography must be one of the following values: {', '.join(allowed_geo)}"
            }), 400

        # Konversi data ke DataFrame
        data_frame = pd.DataFrame([data_request], columns=feature_order)

        # Encoding Request Kategorikal
        data_frame['Gender'] = gender_encoder.transform(data_frame['Gender'])
        data_frame['Geography'] = geo_encoder.transform(data_frame['Geography'])

        # Scaling Data
        scaled = scaler.transform(data_frame)

        # Prediksi
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        return jsonify({
            "status": "success",
            "message": "Success to predict customer churn",
            'prediction': int(prediction),
            'probability_churn': round(float(probability), 3)
        })
    except Exception as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
