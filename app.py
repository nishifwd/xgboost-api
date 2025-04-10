from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model on startup (only once)
try:
    with open('your_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/', methods=['GET'])
def home():
    if model is None:
        return "XGBoost API is running, but model failed to load!"
    return "XGBoost Prediction API is running with model loaded successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST',
        'Access-Control-Allow-Headers': 'Content-Type'
    }
    
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return ('', 204, headers)
    
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500, headers
    
    # Get request data
    data = request.get_json()
    
    if not data or 'features' not in data:
        return jsonify({'error': 'Invalid request: missing features data'}), 400, headers
    
    try:
        # Process features - convert to numpy array
        features = np.array(data['features'], dtype=float)
        
        # Make predictions
        predictions = model.predict(features)
        
        # Return predictions
        return jsonify({'predictions': predictions.tolist()}), 200, headers
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500, headers

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)