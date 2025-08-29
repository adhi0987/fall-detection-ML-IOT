from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd # Used for creating DataFrame to handle feature names for scaler
import uvicorn # To run the FastAPI app
import sklearn
print(sklearn.__version__)
# Initialize FastAPI app
app = FastAPI(
    title="SVM Fall Detection API",
    description="API for predicting fall events using a pre-trained SVM model."
)

# Load the trained model and scaler
try:
    with open('svm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Please ensure 'svm_model.pkl' and 'scaler.pkl' are in the same directory.")
    model = None # Set to None to indicate loading failure
    scaler = None
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Define the input data structure for prediction using Pydantic
class PredictionInput(BaseModel):
    max_Ax: float
    min_Ax: float
    var_Ax: float
    mean_Ax: float
    max_Ay: float
    min_Ay: float
    var_Ay: float
    mean_Ay: float
    max_Az: float
    min_Az: float
    var_Az: float
    mean_Az: float
    max_pitch: float
    min_pitch: float
    var_pitch: float
    mean_pitch: float

# Define a root endpoint for basic testing
@app.get("/")
async def read_root():
    return {"message": "Welcome to the SVM Fall Detection API!"}

# Define the prediction endpoint
@app.post("/predict/")
async def predict_fall(data: PredictionInput):
    if model is None or scaler is None:
        return {"error": "Model or scaler not loaded. Please check server logs."}

    try:
        # Convert input data to a pandas DataFrame for consistent scaling
        # Ensure the order of features matches the training order
        features = [
            data.max_Ax, data.min_Ax, data.var_Ax, data.mean_Ax,
            data.max_Ay, data.min_Ay, data.var_Ay, data.mean_Ay,
            data.max_Az, data.min_Az, data.var_Az, data.mean_Az,
            data.max_pitch, data.min_pitch, data.var_pitch, data.mean_pitch
        ]
        feature_names = [
            'max_Ax', 'min_Ax', 'var_Ax', 'mean_Ax',
            'max_Ay', 'min_Ay', 'var_Ay', 'mean_Ay',
            'max_Az', 'min_Az', 'var_Az', 'mean_Az',
            'max_pitch', 'min_pitch', 'var_pitch', 'mean_pitch'
        ]
        input_df = pd.DataFrame([features], columns=feature_names)

        # Scale the input data
        scaled_data = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_data)

        # The model predicts either 0 or 1.
        # Assuming 1 indicates a 'fall' and 0 indicates 'no fall'.
        prediction_label = "Fall" if prediction[0] == 1 else "No Fall"

        return {"prediction": int(prediction[0]), "prediction_label": prediction_label}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

# To run this API:
# 1. Save this code as a Python file (e.g., `app.py`).
# 2. Make sure `svm_model.pkl` and `scaler.pkl` are in the same directory.
# 3. Install necessary libraries: `pip install fastapi uvicorn pandas scikit-learn`
# 4. Run from your terminal: `uvicorn app:app --reload`
# 5. Access the API documentation at `http://127.0.0.1:8000/docs` or `http://127.0.0.1:8000/redoc`
