from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("XGBoost.pkl")

@app.route("/")
def home():
    return "WQI Predictor API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features")
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return jsonify({"prediction": float(prediction)})
