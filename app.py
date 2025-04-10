from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the joblib-saved XGBoost model
model = joblib.load("XGBoost.pkl")  # Use your actual model filename

# Your feature list (must match training order exactly)
FEATURES = [
    "Alkalinity-total (as CaCO3)",
    "Ammonia-Total (as N)",
    "BOD - 5 days (Total)",
    "Chloride",
    "Conductivity @25°C",
    "Dissolved Oxygen",
    "ortho-Phosphate (as P) - unspecified",
    "pH",
    "Temperature",
    "Total Hardness (as CaCO3)",
    "True Colour"
]

@app.route("/")
def home():
    return "XGBoost WQI Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing features"}), 400

    try:
        # Ensure the order and format of features is correct
        df = pd.DataFrame([features])[FEATURES]
        prediction = model.predict(df)[0]
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
