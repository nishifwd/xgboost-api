from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np

app = FastAPI()

# Load the model from JSON
model = xgb.Booster()
model.load_model("XGBoost.json")

# Define input schema
class Features(BaseModel):
    features: dict

@app.post("/predict")
async def predict(data: Features):
    input_array = np.array([list(data.features.values())])
    dmatrix = xgb.DMatrix(input_array)
    prediction = model.predict(dmatrix)
    return {"prediction": float(prediction[0])}
