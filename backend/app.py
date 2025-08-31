import os
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import database
import models
from models import FallDetection, Base

# Initialize FastAPI app
app = FastAPI(
    title="SVM Fall Detection API",
    description="API for predicting fall events using a pre-trained SVM model."
)

origins = ["http://localhost:5173", "https://fall-ml-iot-interface.onrender.com"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and scaler
try:
    with open('new_svm_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('new_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Please ensure 'svm_model.pkl' and 'scaler.pkl' are in the same directory.")
    model = None
    scaler = None
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Create database tables on startup
@app.on_event("startup")
def on_startup():
    try:
        models.Base.metadata.create_all(bind=database.engine)
        print("Database tables created successfully.")
    except Exception as e:
        print(f"Error creating database tables: {e}")

# Define the input data structure for prediction using Pydantic
class PredictionInput(BaseModel):
    mac_addr: str
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

# Dependency to get a database session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Define a root endpoint for basic testing
@app.get("/")
async def read_root():
    return {"message": "Welcome to the SVM Fall Detection API!"}

# Define the prediction endpoint
@app.post("/predict/")
async def predict_fall(data: PredictionInput, db: Session = Depends(get_db)):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded. Please check server logs.")
    try:
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
        scaled_data = scaler.transform(input_df)
        print(input_df,"\n changed to :",scaled_data)
        prediction = model.predict(scaled_data)
        prediction_label = "Fall" if prediction[0] == 1 else "No Fall"

        db_fall_detection = models.FallDetection(
            mac_addr=data.mac_addr,
            max_Ax=data.max_Ax,
            min_Ax=data.min_Ax,
            var_Ax=data.var_Ax,
            mean_Ax=data.mean_Ax,
            max_Ay=data.max_Ay,
            min_Ay=data.min_Ay,
            var_Ay=data.var_Ay,
            mean_Ay=data.mean_Ay,
            max_Az=data.max_Az,
            min_Az=data.min_Az,
            var_Az=data.var_Az,
            mean_Az=data.mean_Az,
            max_pitch=data.max_pitch,
            min_pitch=data.min_pitch,
            var_pitch=data.var_pitch,
            mean_pitch=data.mean_pitch,
            prediction=int(prediction[0]),
            prediction_label=prediction_label
        )
        db.add(db_fall_detection)
        db.commit()
        db.refresh(db_fall_detection)

        return {"prediction": int(prediction[0]), "prediction_label": prediction_label, "record_id": db_fall_detection.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction or database operation failed: {e}")
    
# Define an endpoint to get all unique MAC addresses
@app.get("/getdevices")
def get_unique_devices(db: Session = Depends(get_db)):
    try:
        unique_macs = db.query(distinct(FallDetection.mac_addr)).all()
        mac_address_list = [mac[0] for mac in unique_macs]
        return {"unique_devices": mac_address_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve devices: {e}")
    
# Define a new endpoint to get data points for a specific MAC address
@app.get("/getdatapoints/{mac_address}")
def get_data_for_device(mac_address: str, db: Session = Depends(get_db)):
    try:
        records = db.query(FallDetection).filter(FallDetection.mac_addr == mac_address).all()
        if not records:
            raise HTTPException(status_code=404, detail=f"No records found for MAC address: {mac_address}")
        return records
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data for device: {e}")