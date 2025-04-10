from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import json

app = Flask(__name__)

# Load the trained XGBoost model
model = xgb.Booster()
model.load_model("XGBoost.json")  # Use your actual model file name here

# Your feature list (make sure the order is exactly as used during training)
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
        # Make sure data is in the correct feature order
        df = pd.DataFrame([features])[FEATURES]
        dmatrix = xgb.DMatrix(df)
        prediction = model.predict(dmatrix)[0]
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
