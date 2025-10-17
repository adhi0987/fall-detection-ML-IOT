# backend/app.py

import os
import json
import logging
import threading
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import List
from dotenv import load_dotenv
import asyncio
from datetime import datetime # <-- ADD THIS

import paho.mqtt.client as mqtt
from fastapi import FastAPI, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# --- New scikit-learn imports for training ---
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

#modules in project structure
import database
import models
import schemas
from models import FallDetection

load_dotenv()

# --- 1. Logging and Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# MQTT Configuration
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
DATA_COLLECTION_TOPIC = "datacollection"
DATA_PREDICTION_TOPIC = "dataprediction"
FALL_ALERT_TOPIC = "fallalert"
# MQTT_CLIENT_ID = "fastapi-backend-client-rf" # unique id
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID") # unique id
# Model file path
MODEL_PIPELINE_PATH = 'fall_detection_model.joblib' # Path to  joblib file

# <-- CHANGE: Define feature names as a constant for consistency ---
FEATURE_NAMES = [
    'max_Ax', 'min_Ax', 'var_Ax', 'mean_Ax', 'max_Ay', 'min_Ay', 'var_Ay', 'mean_Ay',
    'max_Az', 'min_Az', 'var_Az', 'mean_Az', 'max_Gx', 'min_Gx', 'var_Gx', 'mean_Gx',
    'max_Gy', 'min_Gy', 'var_Gy', 'mean_Gy', 'max_Gz', 'min_Gz', 'var_Gz', 'mean_Gz'
]

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()


# --- 2. Database and Dependency Setup ---
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 3. MQTT Client Logic ---
mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected to MQTT Broker successfully!")
        client.subscribe([(DATA_COLLECTION_TOPIC, 0), (DATA_PREDICTION_TOPIC, 0)])
        logger.info(f"Subscribed to topics: {DATA_COLLECTION_TOPIC}, {DATA_PREDICTION_TOPIC}")
    else:
        logger.error(f"Failed to connect to MQTT Broker. Return code: {rc}")

# <-- CHANGE: Custom JSON encoder to handle datetime objects
def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

async def on_mqtt_message(client, userdata, message):
    payload_str = message.payload.decode("utf-8")
    topic = message.topic
    logger.info(f"Received MQTT message on topic '{topic}'")

    db = None # Initialize db to None
    try:
        data = json.loads(payload_str)
        db = database.SessionLocal()  # Independent session for thread
        prediction_value = 0  # Default

        if topic == DATA_COLLECTION_TOPIC:
            db_record = FallDetection(
                mac_addr=data.get("mac_addr"),
                max_Ax=data.get("max_Ax"), min_Ax=data.get("min_Ax"), var_Ax=data.get("var_Ax"), mean_Ax=data.get("mean_Ax"),
                max_Ay=data.get("max_Ay"), min_Ay=data.get("min_Ay"), var_Ay=data.get("var_Ay"), mean_Ay=data.get("mean_Ay"),
                max_Az=data.get("max_Az"), min_Az=data.get("min_Az"), var_Az=data.get("var_Az"), mean_Az=data.get("mean_Az"),
                max_Gx=data.get("max_Gx"), min_Gx=data.get("min_Gx"), var_Gx=data.get("var_Gx"), mean_Gx=data.get("mean_Gx"),
                max_Gy=data.get("max_Gy"), min_Gy=data.get("min_Gy"), var_Gy=data.get("var_Gy"), mean_Gy=data.get("mean_Gy"),
                max_Gz=data.get("max_Gz"), min_Gz=data.get("min_Gz"), var_Gz=data.get("var_Gz"), mean_Gz=data.get("mean_Gz"),
                prediction=-1,
                prediction_label="Unlabelled",
                source_type="labelled"
            )
            db.add(db_record)
            db.commit()
            db.refresh(db_record) # Refresh to get the timestamp from the db
            logger.info(f"Stored unlabelled data for MAC: {data.get('mac_addr')} (Record ID: {db_record.id})")
            
            # <-- CHANGE: Convert record to dict and then serialize with custom handler
            record_dict = schemas.FallDetection.from_orm(db_record).dict()
            await manager.broadcast(json.dumps(record_dict, default=json_serializer))


        elif topic == DATA_PREDICTION_TOPIC: # Prediction topic
            logger.info("Prediction topic received. Preparing features for prediction...")

            if not app.state.pipeline:
                logger.error("Model pipeline not loaded. Cannot perform prediction.")
                return

            # <-- CHANGE: Prepare features as a list first
            feature_values = [
                data.get("max_Ax"), data.get("min_Ax"), data.get("var_Ax"), data.get("mean_Ax"),
                data.get("max_Ay"), data.get("min_Ay"), data.get("var_Ay"), data.get("mean_Ay"),
                data.get("max_Az"), data.get("min_Az"), data.get("var_Az"), data.get("mean_Az"),
                data.get("max_Gx"), data.get("min_Gx"), data.get("var_Gx"), data.get("mean_Gx"),
                data.get("max_Gy"), data.get("min_Gy"), data.get("var_Gy"), data.get("mean_Gy"),
                data.get("max_Gz"), data.get("min_Gz"), data.get("var_Gz"), data.get("mean_Gz")
            ]

            # <-- CHANGE: Create a pandas DataFrame with feature names to resolve the warning
            prediction_df = pd.DataFrame([feature_values], columns=FEATURE_NAMES)

            # The pipeline handles scaling and prediction
            prediction = app.state.pipeline.predict(prediction_df)
            prediction_value = int(prediction[0])
            prediction_label = "Fall" if prediction_value == 1 else "No Fall"
            logger.info(f"Predicted value for MAC {data.get('mac_addr')}: {prediction_label}")

            # Store the prediction result in the database
            db_record = FallDetection(
                mac_addr=data.get("mac_addr"),
                max_Ax=data.get("max_Ax"), min_Ax=data.get("min_Ax"), var_Ax=data.get("var_Ax"), mean_Ax=data.get("mean_Ax"),
                max_Ay=data.get("max_Ay"), min_Ay=data.get("min_Ay"), var_Ay=data.get("var_Ay"), mean_Ay=data.get("mean_Ay"),
                max_Az=data.get("max_Az"), min_Az=data.get("min_Az"), var_Az=data.get("var_Az"), mean_Az=data.get("mean_Az"),
                max_Gx=data.get("max_Gx"), min_Gx=data.get("min_Gx"), var_Gx=data.get("var_Gx"), mean_Gx=data.get("mean_Gx"),
                max_Gy=data.get("max_Gy"), min_Gy=data.get("min_Gy"), var_Gy=data.get("var_Gy"), mean_Gy=data.get("mean_Gy"),
                max_Gz=data.get("max_Gz"), min_Gz=data.get("min_Gz"), var_Gz=data.get("var_Gz"), mean_Gz=data.get("mean_Gz"),
                prediction=prediction_value,
                prediction_label=prediction_label,
                source_type="predicted"
            )
            db.add(db_record)
            db.commit()
            db.refresh(db_record) # Refresh to get the timestamp from the db
            logger.info(f"Prediction stored for MAC {data.get('mac_addr')} (Record ID: {db_record.id})")

            # <-- CHANGE: Convert record to dict and then serialize with custom handler
            record_dict = schemas.FallDetection.from_orm(db_record).dict()
            await manager.broadcast(json.dumps(record_dict, default=json_serializer))


        # If a fall was detected (from prediction topic), publish an alert
        if prediction_value == 1:
            alert_payload = json.dumps({"mac_addr": data.get("mac_addr"), "alert": "Fall Detected!"})
            client.publish(FALL_ALERT_TOPIC, alert_payload)
            logger.info(f"Published fall alert for MAC: {data.get('mac_addr')}")

    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")
    finally:
        if db:
            db.close()

# Assign callbacks
mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = lambda client, userdata, message: asyncio.run(on_mqtt_message(client, userdata, message))


# --- 4. FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup...")

    # Load the scikit-learn pipeline
    try:
        logger.info(f"Loading model pipeline from: {MODEL_PIPELINE_PATH}")
        app.state.pipeline = joblib.load(MODEL_PIPELINE_PATH)
        logger.info("Model pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Could not load model pipeline: {e}")
        app.state.pipeline = None

    # Create database tables
    logger.info("Verifying database tables...")
    models.Base.metadata.create_all(bind=database.engine)
    logger.info("Database tables verified.")

    # Start MQTT client
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        # <-- CHANGE: Use loop_start() instead of creating a thread manually
        mqtt_client.loop_start() 
        logger.info("MQTT client loop started.")
    except Exception as e:
        logger.error(f"Could not connect to MQTT broker: {e}")

    yield

    logger.info("Application shutdown...")
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    logger.info("MQTT client disconnected.")

# --- 5. FastAPI App Initialization ---
app = FastAPI(
    title="Fall Detection API with MQTT and Model Training",
    description="An API to handle fall detection data, predictions via MQTT, and model retraining.",
    lifespan=lifespan
)

# CORS Middleware
origins = ["http://localhost:5173", "*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# --- 6. Pydantic Models ---
class LabelDataRequest(BaseModel):
    prediction: int

# --- 7. HTTP API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Fall Detection API!"}

@app.put("/labeldata/{record_id}")
def label_data(record_id: int, request: LabelDataRequest, db: Session = Depends(get_db)):
    db_record = db.query(FallDetection).filter(FallDetection.id == record_id).first()
    if not db_record:
        raise HTTPException(status_code=404, detail="Record not found")

    if request.prediction not in [0, 1]:
        raise HTTPException(status_code=400, detail="Prediction value must be 0 (No Fall) or 1 (Fall)")

    db_record.prediction = request.prediction
    db_record.prediction_label = "Fall" if request.prediction == 1 else "No Fall"
    db.commit()
    logger.info(f"Record {record_id} labeled manually as '{db_record.prediction_label}'")
    return {"message": f"Record {record_id} labeled successfully."}

@app.post("/trainmodel")
def train_model(db: Session = Depends(get_db)):
    logger.info("Starting model retraining process with RandomForest...")

    # 1. Fetch all labeled data from the database
    query = db.query(FallDetection).filter(FallDetection.source_type == "labelled").filter(FallDetection.prediction.in_([0, 1])).statement
    df = pd.read_sql(query, db.bind)

    if df.shape[0] < 20: # A reasonable minimum for training
        logger.warning(f"Not enough data to train. Found {df.shape[0]} records, need at least 20.")
        raise HTTPException(status_code=400, detail="Not enough labeled data to train the model.")

    # 2. <-- CHANGE: Use the global FEATURE_NAMES constant
    X = df[FEATURE_NAMES]
    y = df['prediction']
    logger.info(f"Training dataset prepared with {len(y)} samples.")

    try:
        # 3. Define the new pipeline
        new_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # 4. Train the pipeline
        new_pipeline.fit(X, y)
        logger.info("Model pipeline training completed.")

        # 5. Save the newly trained pipeline
        joblib.dump(new_pipeline, MODEL_PIPELINE_PATH)

        # 6. Reload the new pipeline into the app state
        app.state.pipeline = new_pipeline
        logger.info("New model pipeline saved and reloaded into application state.")

        return {"message": "Model training completed successfully.", "dataset_size": df.shape[0]}

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise HTTPException(status_code=500, detail=f"Model training failed: {e}")

@app.get("/fetchdata", response_model=List[schemas.FallDetection])
def fetch_data(source_type: str = Query(..., enum=["predicted", "labelled"]), db: Session = Depends(get_db)):
    data = db.query(FallDetection).filter(FallDetection.source_type == source_type).order_by(FallDetection.timestamp.desc()).all()
    return data

@app.get("/showdataset", response_model=List[schemas.FallDetection])
def show_dataset(db: Session = Depends(get_db)):
    dataset = db.query(FallDetection).filter(FallDetection.prediction.in_([0, 1])).order_by(FallDetection.timestamp.desc()).all()
    return dataset

# To run: uvicorn app:app --reload