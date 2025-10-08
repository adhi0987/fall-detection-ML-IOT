import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import distinct
from fastapi.middleware.cors import CORSMiddleware

# Assuming these modules are in your project structure
import database
import models
from models import FallDetection, Base

# --- 1. Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- 2. Lifespan Management (The New Way) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    """
    # === STARTUP LOGIC ===
    logger.info("Application startup...")

    # Load the trained model and scaler
    logger.info("Loading ML model and scaler...")
    try:
        with open('svm_model.pkl', 'rb') as model_file:
            app.state.model = pickle.load(model_file)
        with open('scalar.pkl', 'rb') as scaler_file:
            app.state.scaler = pickle.load(scaler_file)
        logger.info("Model and scaler loaded successfully and attached to app.state")
    except FileNotFoundError:
        logger.error("FATAL: Model or scaler file not found. The application cannot make predictions.")
        app.state.model = None
        app.state.scaler = None
    except Exception as e:
        logger.error(f"FATAL: Error loading model or scaler: {e}")
        app.state.model = None
        app.state.scaler = None

    # Create database tables
    logger.info("Creating database tables...")
    try:
        models.Base.metadata.create_all(bind=database.engine)
        logger.info("Database tables verified/created successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

    yield  # The application is now running

    # === SHUTDOWN LOGIC ===
    logger.info("Application shutdown...")
    # You can add cleanup code here if needed (e.g., closing connection pools)


# --- 3. FastAPI App Initialization ---
app = FastAPI(
    title="SVM Fall Detection API",
    description="API for predicting fall events using a pre-trained SVM model.",
    lifespan=lifespan  # Register the lifespan manager
)

# CORS Middleware
origins = ["http://localhost:5173", "https://fall-ml-iot-interface.onrender.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 4. Pydantic Models & Dependencies ---
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


# --- 5. API Endpoints ---
@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the SVM Fall Detection API!"}

@app.post("/predict/")
async def predict_fall(request: Request, data: PredictionInput, db: Session = Depends(get_db)):
    logger.info(f"Received prediction request for MAC address: {data.mac_addr}")
    
    # Access model and scaler from app.state
    model = request.app.state.model
    scaler = request.app.state.scaler

    if model is None or scaler is None:
        logger.error("Prediction failed: Model or scaler not loaded.")
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
        
        logger.info(f"Input data (first 16 features): {input_df.iloc[0].values[:]}")
        logger.info(f"Scaled Input data (first 16 features): {scaled_data[0]}")
        
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
        
        logger.info(f"Prediction for {data.mac_addr}: {prediction_label}. Record saved to DB with ID: {db_fall_detection.id}")

        return {"prediction": int(prediction[0]), "prediction_label": prediction_label, "record_id": db_fall_detection.id}
    except Exception as e:
        logger.error(f"Prediction or database operation failed for {data.mac_addr}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction or database operation failed: {e}")

@app.get("/getdevices")
def get_unique_devices(db: Session = Depends(get_db)):
    try:
        unique_macs = db.query(distinct(FallDetection.mac_addr)).all()
        mac_address_list = [mac[0] for mac in unique_macs]
        logger.info(f"Retrieved unique devices: {mac_address_list}")
        return {"unique_devices": mac_address_list}
    except Exception as e:
        logger.error(f"Failed to retrieve devices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve devices: {e}")

@app.get("/getdatapoints/{mac_address}")
def get_data_for_device(mac_address: str, db: Session = Depends(get_db)):
    try:
        records = db.query(FallDetection).filter(FallDetection.mac_addr == mac_address).all()
        if not records:
            logger.warning(f"No records found for MAC address: {mac_address}")
            raise HTTPException(status_code=404, detail=f"No records found for MAC address: {mac_address}")
        
        logger.info(f"Retrieved {len(records)} records for MAC address: {mac_address}")
        return records
    except HTTPException as e:
        # Re-raise HTTPException to preserve status code and detail
        raise e
    except Exception as e:
        logger.error(f"Failed to retrieve data for device {mac_address}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data for device: {e}")