from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load the model from JSON
model = xgb.Booster()
model.load_model("XGBoost.json")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features", {})
    # Convert the features dictionary values to a numpy array
    input_array = np.array([list(features.values())])
    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(input_array)
    # Get prediction
    prediction = model.predict(dmatrix)
    return jsonify({"prediction": float(prediction[0])})
if __name__ == "__main__":
    app.run(debug=True)
