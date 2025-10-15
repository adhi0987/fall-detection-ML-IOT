# # import os
# # import pickle
# # import logging
# # import numpy as np
# # import pandas as pd
# # from typing import List
# # from contextlib import asynccontextmanager

# # from fastapi import FastAPI, Depends, HTTPException, Request
# # from pydantic import BaseModel
# # from sqlalchemy.orm import Session
# # from sqlalchemy import distinct
# # from fastapi.middleware.cors import CORSMiddleware

# # # Assuming these modules are in your project structure
# # import database
# # import models
# # from models import FallDetection, Base

# # # --- 1. Logging Configuration ---
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)


# # # --- 2. Lifespan Management (The New Way) ---
# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     """
# #     Handles application startup and shutdown events.
# #     """
# #     # === STARTUP LOGIC ===
# #     logger.info("Application startup...")

# #     # Load the trained model and scaler
# #     logger.info("Loading ML model and scaler...")
# #     try:
# #         with open('svm_model.pkl', 'rb') as model_file:
# #             app.state.model = pickle.load(model_file)
# #         with open('scalar.pkl', 'rb') as scaler_file:
# #             app.state.scaler = pickle.load(scaler_file)
# #         logger.info("Model and scaler loaded successfully and attached to app.state")
# #     except FileNotFoundError:
# #         logger.error("FATAL: Model or scaler file not found. The application cannot make predictions.")
# #         app.state.model = None
# #         app.state.scaler = None
# #     except Exception as e:
# #         logger.error(f"FATAL: Error loading model or scaler: {e}")
# #         app.state.model = None
# #         app.state.scaler = None

# #     # Create database tables
# #     logger.info("Creating database tables...")
# #     try:
# #         models.Base.metadata.create_all(bind=database.engine)
# #         logger.info("Database tables verified/created successfully.")
# #     except Exception as e:
# #         logger.error(f"Error creating database tables: {e}")

# #     yield  # The application is now running

# #     # === SHUTDOWN LOGIC ===
# #     logger.info("Application shutdown...")
# #     # You can add cleanup code here if needed (e.g., closing connection pools)


# # # --- 3. FastAPI App Initialization ---
# # app = FastAPI(
# #     title="SVM Fall Detection API",
# #     description="API for predicting fall events using a pre-trained SVM model.",
# #     lifespan=lifespan  # Register the lifespan manager
# # )

# # # CORS Middleware
# # origins = ["http://localhost:5173", "https://fall-ml-iot-interface.onrender.com"]
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=origins,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )


# # # --- 4. Pydantic Models & Dependencies ---
# # class PredictionInput(BaseModel):
# #     mac_addr: str
# #     max_Ax: float
# #     min_Ax: float
# #     var_Ax: float
# #     mean_Ax: float
# #     max_Ay: float
# #     min_Ay: float
# #     var_Ay: float
# #     mean_Ay: float
# #     max_Az: float
# #     min_Az: float
# #     var_Az: float
# #     mean_Az: float
# #     max_pitch: float
# #     min_pitch: float
# #     var_pitch: float
# #     mean_pitch: float

# # # Dependency to get a database session
# # def get_db():
# #     db = database.SessionLocal()
# #     try:
# #         yield db
# #     finally:
# #         db.close()


# # # --- 5. API Endpoints ---
# # @app.get("/")
# # async def read_root():
# #     logger.info("Root endpoint accessed.")
# #     return {"message": "Welcome to the SVM Fall Detection API!"}

# # @app.post("/predict/")
# # async def predict_fall(request: Request, data: PredictionInput, db: Session = Depends(get_db)):
# #     logger.info(f"Received prediction request for MAC address: {data.mac_addr}")
    
# #     # Access model and scaler from app.state
# #     model = request.app.state.model
# #     scaler = request.app.state.scaler

# #     if model is None or scaler is None:
# #         logger.error("Prediction failed: Model or scaler not loaded.")
# #         raise HTTPException(status_code=500, detail="Model or scaler not loaded. Please check server logs.")
    
# #     try:
# #         features = [
# #             data.max_Ax, data.min_Ax, data.var_Ax, data.mean_Ax,
# #             data.max_Ay, data.min_Ay, data.var_Ay, data.mean_Ay,
# #             data.max_Az, data.min_Az, data.var_Az, data.mean_Az,
# #             data.max_pitch, data.min_pitch, data.var_pitch, data.mean_pitch
# #         ]
# #         feature_names = [
# #             'max_Ax', 'min_Ax', 'var_Ax', 'mean_Ax',
# #             'max_Ay', 'min_Ay', 'var_Ay', 'mean_Ay',
# #             'max_Az', 'min_Az', 'var_Az', 'mean_Az',
# #             'max_pitch', 'min_pitch', 'var_pitch', 'mean_pitch'
# #         ]
# #         input_df = pd.DataFrame([features], columns=feature_names)
# #         scaled_data = scaler.transform(input_df)
        
# #         logger.info(f"Input data (first 16 features): {input_df.iloc[0].values[:]}")
# #         logger.info(f"Scaled Input data (first 16 features): {scaled_data[0]}")
        
# #         prediction = model.predict(scaled_data)
# #         prediction_label = "Fall" if prediction[0] == 1 else "No Fall"

# #         db_fall_detection = models.FallDetection(
# #             mac_addr=data.mac_addr,
# #             max_Ax=data.max_Ax,
# #             min_Ax=data.min_Ax,
# #             var_Ax=data.var_Ax,
# #             mean_Ax=data.mean_Ax,
# #             max_Ay=data.max_Ay,
# #             min_Ay=data.min_Ay,
# #             var_Ay=data.var_Ay,
# #             mean_Ay=data.mean_Ay,
# #             max_Az=data.max_Az,
# #             min_Az=data.min_Az,
# #             var_Az=data.var_Az,
# #             mean_Az=data.mean_Az,
# #             max_pitch=data.max_pitch,
# #             min_pitch=data.min_pitch,
# #             var_pitch=data.var_pitch,
# #             mean_pitch=data.mean_pitch,
# #             prediction=int(prediction[0]),
# #             prediction_label=prediction_label
# #         )
# #         db.add(db_fall_detection)
# #         db.commit()
# #         db.refresh(db_fall_detection)
        
# #         logger.info(f"Prediction for {data.mac_addr}: {prediction_label}. Record saved to DB with ID: {db_fall_detection.id}")

# #         return {"prediction": int(prediction[0]), "prediction_label": prediction_label, "record_id": db_fall_detection.id}
# #     except Exception as e:
# #         logger.error(f"Prediction or database operation failed for {data.mac_addr}: {e}")
# #         raise HTTPException(status_code=500, detail=f"Prediction or database operation failed: {e}")

# # @app.get("/getdevices")
# # def get_unique_devices(db: Session = Depends(get_db)):
# #     try:
# #         unique_macs = db.query(distinct(FallDetection.mac_addr)).all()
# #         mac_address_list = [mac[0] for mac in unique_macs]
# #         logger.info(f"Retrieved unique devices: {mac_address_list}")
# #         return {"unique_devices": mac_address_list}
# #     except Exception as e:
# #         logger.error(f"Failed to retrieve devices: {e}")
# #         raise HTTPException(status_code=500, detail=f"Failed to retrieve devices: {e}")

# # @app.get("/getdatapoints/{mac_address}")
# # def get_data_for_device(mac_address: str, db: Session = Depends(get_db)):
# #     try:
# #         records = db.query(FallDetection).filter(FallDetection.mac_addr == mac_address).all()
# #         if not records:
# #             logger.warning(f"No records found for MAC address: {mac_address}")
# #             raise HTTPException(status_code=404, detail=f"No records found for MAC address: {mac_address}")
        
# #         logger.info(f"Retrieved {len(records)} records for MAC address: {mac_address}")
# #         return records
# #     except HTTPException as e:
# #         # Re-raise HTTPException to preserve status code and detail
# #         raise e
# #     except Exception as e:
# #         logger.error(f"Failed to retrieve data for device {mac_address}: {e}")
# #         raise HTTPException(status_code=500, detail=f"Failed to retrieve data for device: {e}")


# # import os
# # import json
# # import logging
# # import threading
# # import numpy as np
# # import pandas as pd
# # from contextlib import asynccontextmanager
# # from typing import List

# # import paho.mqtt.client as mqtt
# # from fastapi import FastAPI, Depends, HTTPException, Query
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel
# # from sqlalchemy.orm import Session
# # from tensorflow.keras.models import load_model
# # import pickle

# # # Assuming these modules are in your project structure
# # import database
# # import models
# # import schemas
# # from models import FallDetection

# # # --- 1. Logging and Configuration ---
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# # # MQTT Configuration
# # # MQTT_BROKER = "broker.hivemq.com"  # Using a public broker for development
# # MQTT_BROKER = "test.mosquitto.org"  # Alternative public broker
# # MQTT_PORT = 1883
# # DATA_COLLECTION_TOPIC = "datacollection"
# # DATA_PREDICTION_TOPIC = "dataprediction"
# # FALL_ALERT_TOPIC = "fallalert"
# # MQTT_CLIENT_ID = "fastapi-backend-client"

# # # Model and Scaler file paths
# # MODEL_PATH = 'neural_network_model.h5'
# # SCALER_PATH = 'scalar.pkl'

# # # --- 2. Database and Dependency Setup ---
# # # Dependency to get a database session for HTTP requests
# # def get_db():
# #     db = database.SessionLocal()
# #     try:
# #         yield db
# #     finally:
# #         db.close()

# # # --- 3. MQTT Client Logic ---
# # # Global MQTT client instance
# # mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)

# # def on_mqtt_message(client, userdata, message):
# #     """
# #     Handles incoming MQTT messages for data collection and prediction.
# #     """
# #     payload_str = message.payload.decode("utf-8")
# #     topic = message.topic
# #     logger.info(f"Received message on topic '{topic}'")

# #     try:
# #         data = json.loads(payload_str)
# #         # Create a new, independent database session for this thread
# #         db = database.SessionLocal()
# #         prediction_value =0
# #         if topic == DATA_COLLECTION_TOPIC:
# #             db_record = FallDetection(
# #                 mac_addr=data.get("mac_addr"),
# #                 max_Ax=data.get("max_Ax"), min_Ax=data.get("min_Ax"), var_Ax=data.get("var_Ax"), mean_Ax=data.get("mean_Ax"),
# #                 max_Ay=data.get("max_Ay"), min_Ay=data.get("min_Ay"), var_Ay=data.get("var_Ay"), mean_Ay=data.get("mean_Ay"),
# #                 max_Az=data.get("max_Az"), min_Az=data.get("min_Az"), var_Az=data.get("var_Az"), mean_Az=data.get("mean_Az"),
# #                 max_Gx=data.get("max_Gx"), min_Gx=data.get("min_Gx"), var_Gx=data.get("var_Gx"), mean_Gx=data.get("mean_Gx"),
# #                 max_Gy=data.get("max_Gy"), min_Gy=data.get("min_Gy"), var_Gy=data.get("var_Gy"), mean_Gy=data.get("mean_Gy"),
# #                 max_Gz=data.get("max_Gz"), min_Gz=data.get("min_Gz"), var_Gz=data.get("var_Gz"), mean_Gz=data.get("mean_Gz"),
# #                 prediction=-1,  # -1 signifies unlabelled data
# #                 prediction_label="Unlabelled",
# #                 source_type="labelled"
# #             )
# #             db.add(db_record)
# #             db.commit()
# #             logger.info(f"Stored unlabelled data for MAC: {data.get('mac_addr')}")

# #         # elif topic == DATA_PREDICTION_TOPIC:
# #         #     if not app.state.model or not app.state.scaler:
# #         #         logger.error("Model or scaler not loaded, cannot predict.")
# #         #         return

# #         #     # Prepare features for prediction
# #         #     features = [
# #         #         data.get("max_Ax"), data.get("min_Ax"), data.get("var_Ax"), data.get("mean_Ax"),
# #         #         data.get("max_Ay"), data.get("min_Ay"), data.get("var_Ay"), data.get("mean_Ay"),
# #         #         data.get("max_Az"), data.get("min_Az"), data.get("var_Az"), data.get("mean_Az"),
# #         #         data.get("max_Gx"), data.get("min_Gx"), data.get("var_Gx"), data.get("mean_Gx"),
# #         #         data.get("max_Gy"), data.get("min_Gy"), data.get("var_Gy"), data.get("mean_Gy"),
# #         #         data.get("max_Gz"), data.get("min_Gz"), data.get("var_Gz"), data.get("mean_Gz")
# #         #     ]
# #         #     scaled_features = app.state.scaler.transform([features])
            
# #         #     # Predict
# #         #     prediction_probs = app.state.model.predict(scaled_features)
# #         #     prediction_value = int(np.argmax(prediction_probs, axis=1)[0])
# #         #     prediction_label = "Fall" if prediction_value == 1 else "No Fall"

# #         #     db_record = FallDetection(
# #         #         mac_addr=data.get("mac_addr"),
# #         #         # ... (add all features as above) ...
# #         #         max_Ax=data.get("max_Ax"), min_Ax=data.get("min_Ax"), var_Ax=data.get("var_Ax"), mean_Ax=data.get("mean_Ax"),
# #         #         max_Ay=data.get("max_Ay"), min_Ay=data.get("min_Ay"), var_Ay=data.get("var_Ay"), mean_Ay=data.get("mean_Ay"),
# #         #         max_Az=data.get("max_Az"), min_Az=data.get("min_Az"), var_Az=data.get("var_Az"), mean_Az=data.get("mean_Az"),
# #         #         max_Gx=data.get("max_Gx"), min_Gx=data.get("min_Gx"), var_Gx=data.get("var_Gx"), mean_Gx=data.get("mean_Gx"),
# #         #         max_Gy=data.get("max_Gy"), min_Gy=data.get("min_Gy"), var_Gy=data.get("var_Gy"), mean_Gy=data.get("mean_Gy"),
# #         #         max_Gz=data.get("max_Gz"), min_Gz=data.get("min_Gz"), var_Gz=data.get("var_Gz"), mean_Gz=data.get("mean_Gz"),
# #         #         prediction=prediction_value,
# #         #         prediction_label=prediction_label,
# #         #         source_type="predicted"
# #         #     )
# #         #     db.add(db_record)
# #         #     db.commit()
# #         #     logger.info(f"Prediction for {data.get('mac_addr')}: {prediction_label}")

# #             # If a fall is detected, publish an alert
# #             if prediction_value == 1:
# #                 alert_payload = json.dumps({"mac_addr": data.get("mac_addr"), "alert": "Fall Detected!"})
# #                 client.publish(FALL_ALERT_TOPIC, alert_payload)
# #                 logger.info(f"Published fall alert for {data.get('mac_addr')}")

# #     except Exception as e:
# #         logger.error(f"Error processing MQTT message: {e}")
# #     finally:
# #         db.close() # Ensure the session is closed

# # def on_mqtt_connect(client, userdata, flags, rc):
# #     if rc == 0:
# #         logger.info("Connected to MQTT Broker!")
# #         client.subscribe([(DATA_COLLECTION_TOPIC, 0), (DATA_PREDICTION_TOPIC, 0)])
# #     else:
# #         logger.error(f"Failed to connect to MQTT, return code {rc}\n")

# # # Assign callbacks
# # mqtt_client.on_connect = on_mqtt_connect
# # mqtt_client.on_message = on_mqtt_message

# # # --- 4. FastAPI Lifespan Manager ---
# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     """
# #     Handles application startup and shutdown events.
# #     """
# #     # === STARTUP LOGIC ===
# #     logger.info("Application startup...")
    
# #     # Load ML model and scaler
# #     try:
# #         # logger.info(f"Loading model from: {MODEL_PATH}")
# #         # app.state.model = load_model(MODEL_PATH)
# #         logger.info(f"Loading scaler from: {SCALER_PATH}")
# #         with open(SCALER_PATH, 'rb') as f:
# #             app.state.scaler = pickle.load(f)
# #         logger.info("Model and scaler loaded successfully.")
# #     except Exception as e:
# #         logger.error(f"FATAL: Could not load model or scaler: {e}")
# #         app.state.model = None
# #         app.state.scaler = None

# #     # Create database tables
# #     logger.info("Verifying database tables...")
# #     models.Base.metadata.create_all(bind=database.engine)
# #     logger.info("Database tables verified.")
    
# #     # Start MQTT Client
# #     try:
# #         mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
# #         # Start the network loop in a separate thread
# #         mqtt_thread = threading.Thread(target=mqtt_client.loop_forever)
# #         mqtt_thread.daemon = True
# #         mqtt_thread.start()
# #     except Exception as e:
# #         logger.error(f"Could not connect to MQTT broker: {e}")

# #     yield

# #     # === SHUTDOWN LOGIC ===
# #     logger.info("Application shutdown...")
# #     mqtt_client.disconnect()
# #     logger.info("MQTT client disconnected.")

# # # --- 5. FastAPI App Initialization ---
# # app = FastAPI(
# #     title="Fall Detection API with MQTT and Model Training",
# #     description="An API to handle fall detection data, predictions via MQTT, and model retraining.",
# #     lifespan=lifespan
# # )

# # # CORS Middleware (update with your frontend URL)
# # origins = ["http://localhost:5173", "http://localhost:3000","*"] # Add your frontend's actual origin
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=origins,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # --- 6. Pydantic Models for HTTP Requests ---
# # class LabelDataRequest(BaseModel):
# #     prediction: int

# # # --- 7. HTTP API Endpoints ---
# # @app.get("/")
# # async def read_root():
# #     return {"message": "Welcome to the Fall Detection API!"}

# # @app.put("/labeldata/{record_id}")
# # def label_data(record_id: int, request: LabelDataRequest, db: Session = Depends(get_db)):
# #     db_record = db.query(FallDetection).filter(FallDetection.id == record_id).first()
# #     if not db_record:
# #         raise HTTPException(status_code=404, detail="Record not found")
    
# #     if request.prediction not in [0, 1]:
# #         raise HTTPException(status_code=400, detail="Prediction value must be 0 (No Fall) or 1 (Fall)")

# #     db_record.prediction = request.prediction
# #     db_record.prediction_label = "Fall" if request.prediction == 1 else "No Fall"
# #     db.commit()
# #     logger.info(f"Record {record_id} has been manually labeled as '{db_record.prediction_label}'")
# #     return {"message": f"Record {record_id} labeled successfully."}

# # @app.post("/trainmodel")
# # def train_model(db: Session = Depends(get_db)):
# #     # NOTE: In a production environment, this should be a background task (e.g., using Celery or FastAPI's BackgroundTasks)
# #     # to avoid blocking the server for a long time.
    
# #     logger.info("Starting model training process...")
    
# #     # 1. Fetch all labeled data
# #     query = db.query(FallDetection).filter(FallDetection.prediction.in_([0, 1])).statement
# #     df = pd.read_sql(query, db.bind)

# #     if df.shape[0] < 50: # Set a reasonable minimum for training
# #         logger.warning("Not enough labeled data to train a new model.")
# #         raise HTTPException(status_code=400, detail=f"Not enough data. Need at least 50 labeled records, but found {df.shape[0]}.")

# #     # 2. Prepare features (X) and labels (y)
# #     features = [
# #         'max_Ax', 'min_Ax', 'var_Ax', 'mean_Ax', 'max_Ay', 'min_Ay', 'var_Ay', 'mean_Ay', 'max_Az', 'min_Az', 'var_Az', 'mean_Az',
# #         'max_Gx', 'min_Gx', 'var_Gx', 'mean_Gx', 'max_Gy', 'min_Gy', 'var_Gy', 'mean_Gy', 'max_Gz', 'min_Gz', 'var_Gz', 'mean_Gz'
# #     ]
# #     X = df[features]
# #     y = df['prediction']

# #     # 3. Create and fit a new scaler
# #     from sklearn.preprocessing import StandardScaler
# #     scaler = StandardScaler()
# #     X_scaled = scaler.fit_transform(X)

# #     # 4. Define, compile, and train the model
# #     try:
# #         model = app.state.model
# #         # Re-compile in case of state loss, or define a new one
# #         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# #         model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# #         # 5. Save the newly trained model and scaler
# #         model.save(MODEL_PATH)
# #         with open(SCALER_PATH, 'wb') as f:
# #             pickle.dump(scaler, f)
        
# #         # Reload the model and scaler into the app state
# #         app.state.model = load_model(MODEL_PATH)
# #         app.state.scaler = scaler

# #         logger.info("Model and scaler have been retrained and saved successfully.")
# #         return {"message": "Model training completed successfully.", "dataset_size": df.shape[0]}

# #     except Exception as e:
# #         logger.error(f"An error occurred during model training: {e}")
# #         raise HTTPException(status_code=500, detail=f"Model training failed: {e}")

# # @app.get("/fetchdata", response_model=List[schemas.FallDetection])
# # def fetch_data(source_type: str = Query(..., enum=["predicted", "labelled"]), db: Session = Depends(get_db)):
# #     data = db.query(FallDetection).filter(FallDetection.source_type == source_type).order_by(FallDetection.timestamp.desc()).all()
# #     return data

# # @app.get("/showdataset", response_model=List[schemas.FallDetection])
# # def show_dataset(db: Session = Depends(get_db)):
# #     dataset = db.query(FallDetection).filter(FallDetection.prediction.in_([0, 1])).order_by(FallDetection.timestamp.desc()).all()
# #     return dataset

# # # To run this app: uvicorn app:app --reload


# import os
# import json
# import logging
# import threading
# import numpy as np
# import pandas as pd
# from contextlib import asynccontextmanager
# from typing import List

# import paho.mqtt.client as mqtt
# from fastapi import FastAPI, Depends, HTTPException, Query, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sqlalchemy.orm import Session
# from tensorflow.keras.models import load_model
# import pickle

# # Assuming these modules are in your project structure
# import database
# import models
# import schemas
# from models import FallDetection

# # --- 1. Logging and Configuration ---
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)

# # MQTT Configuration
# # MQTT_BROKER = "broker.hivemq.com"  # Using a public broker for development
# MQTT_BROKER = "test.mosquitto.org"  # Alternative public broker
# MQTT_PORT = 1883
# DATA_COLLECTION_TOPIC = "datacollection"
# DATA_PREDICTION_TOPIC = "dataprediction"
# FALL_ALERT_TOPIC = "fallalert"
# MQTT_CLIENT_ID = "fastapi-backend-client"

# # Model and Scaler file paths
# MODEL_PATH = 'neural_network_model.h5'
# SCALER_PATH = 'scalar.pkl'

# # --- 2. Database and Dependency Setup ---
# # Dependency to get a database session for HTTP requests
# def get_db():
#     db = database.SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # --- 3. MQTT Client Logic ---
# mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)

# def on_mqtt_connect(client, userdata, flags, rc):
#     if rc == 0:
#         logger.info("Connected to MQTT Broker successfully!")
#         # Subscribe to topics and log
#         client.subscribe([(DATA_COLLECTION_TOPIC, 0), (DATA_PREDICTION_TOPIC, 0)])
#         logger.info(f"Subscribed to topics: {DATA_COLLECTION_TOPIC}, {DATA_PREDICTION_TOPIC}")
#     else:
#         logger.error(f"Failed to connect to MQTT Broker. Return code: {rc}")

# def on_mqtt_message(client, userdata, message):
#     payload_str = message.payload.decode("utf-8")
#     topic = message.topic
#     logger.info(f"Received MQTT message on topic '{topic}': {payload_str}")

#     try:
#         data = json.loads(payload_str)
#         db = database.SessionLocal()  # Independent session for thread
#         prediction_value = 0  # Default

#         if topic == DATA_COLLECTION_TOPIC:
#             db_record = FallDetection(
#                 mac_addr=data.get("mac_addr"),
#                 max_Ax=data.get("max_Ax"), min_Ax=data.get("min_Ax"), var_Ax=data.get("var_Ax"), mean_Ax=data.get("mean_Ax"),
#                 max_Ay=data.get("max_Ay"), min_Ay=data.get("min_Ay"), var_Ay=data.get("var_Ay"), mean_Ay=data.get("mean_Ay"),
#                 max_Az=data.get("max_Az"), min_Az=data.get("min_Az"), var_Az=data.get("var_Az"), mean_Az=data.get("mean_Az"),
#                 max_Gx=data.get("max_Gx"), min_Gx=data.get("min_Gx"), var_Gx=data.get("var_Gx"), mean_Gx=data.get("mean_Gx"),
#                 max_Gy=data.get("max_Gy"), min_Gy=data.get("min_Gy"), var_Gy=data.get("var_Gy"), mean_Gy=data.get("mean_Gy"),
#                 max_Gz=data.get("max_Gz"), min_Gz=data.get("min_Gz"), var_Gz=data.get("var_Gz"), mean_Gz=data.get("mean_Gz"),
#                 prediction=-1,
#                 prediction_label="Unlabelled",
#                 source_type="labelled"
#             )
#             db.add(db_record)
#             db.commit()
#             logger.info(f"Stored unlabelled data for MAC: {data.get('mac_addr')} (Record ID: {db_record.id})")

#         # elif topic == DATA_PREDICTION_TOPIC:
#         #     logger.info("Prediction topic received. Preparing features for prediction...")
#         #     if not app.state.model or not app.state.scaler:
#         #         logger.error("Model or scaler not loaded. Cannot perform prediction.")
#         #         return
#         #
#         #     features = [
#         #         data.get("max_Ax"), data.get("min_Ax"), data.get("var_Ax"), data.get("mean_Ax"),
#         #         data.get("max_Ay"), data.get("min_Ay"), data.get("var_Ay"), data.get("mean_Ay"),
#         #         data.get("max_Az"), data.get("min_Az"), data.get("var_Az"), data.get("mean_Az"),
#         #         data.get("max_Gx"), data.get("min_Gx"), data.get("var_Gx"), data.get("mean_Gx"),
#         #         data.get("max_Gy"), data.get("min_Gy"), data.get("var_Gy"), data.get("mean_Gy"),
#         #         data.get("max_Gz"), data.get("min_Gz"), data.get("var_Gz"), data.get("mean_Gz")
#         #     ]
#         #     scaled_features = app.state.scaler.transform([features])
#         #     prediction_probs = app.state.model.predict(scaled_features)
#         #     prediction_value = int(np.argmax(prediction_probs, axis=1)[0])
#         #     prediction_label = "Fall" if prediction_value == 1 else "No Fall"
#         #     logger.info(f"Predicted value for MAC {data.get('mac_addr')}: {prediction_label}")
#         #
#         #     db_record = FallDetection(
#         #         mac_addr=data.get("mac_addr"),
#         #         max_Ax=data.get("max_Ax"), min_Ax=data.get("min_Ax"), var_Ax=data.get("var_Ax"), mean_Ax=data.get("mean_Ax"),
#         #         max_Ay=data.get("max_Ay"), min_Ay=data.get("min_Ay"), var_Ay=data.get("var_Ay"), mean_Ay=data.get("mean_Ay"),
#         #         max_Az=data.get("max_Az"), min_Az=data.get("min_Az"), var_Az=data.get("var_Az"), mean_Az=data.get("mean_Az"),
#         #         max_Gx=data.get("max_Gx"), min_Gx=data.get("min_Gx"), var_Gx=data.get("var_Gx"), mean_Gx=data.get("mean_Gx"),
#         #         max_Gy=data.get("max_Gy"), min_Gy=data.get("min_Gy"), var_Gy=data.get("var_Gy"), mean_Gy=data.get("mean_Gy"),
#         #         max_Gz=data.get("max_Gz"), min_Gz=data.get("min_Gz"), var_Gz=data.get("var_Gz"), mean_Gz=data.get("mean_Gz"),
#         #         prediction=prediction_value,
#         #         prediction_label=prediction_label,
#         #         source_type="predicted"
#         #     )
#         #     db.add(db_record)
#         #     db.commit()
#         #     logger.info(f"Prediction stored for MAC {data.get('mac_addr')} (Record ID: {db_record.id})")

#         # If fall detected, publish alert
#         if prediction_value == 1:
#             alert_payload = json.dumps({"mac_addr": data.get("mac_addr"), "alert": "Fall Detected!"})
#             client.publish(FALL_ALERT_TOPIC, alert_payload)
#             logger.info(f"Published fall alert for MAC: {data.get('mac_addr')}")

#     except Exception as e:
#         logger.error(f"Error processing MQTT message: {e}")
#     finally:
#         db.close()

# # Assign callbacks
# mqtt_client.on_connect = on_mqtt_connect
# mqtt_client.on_message = on_mqtt_message

# # --- 4. FastAPI Lifespan Manager ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("Application startup...")

#     # Load ML model and scaler
#     try:
#         # logger.info(f"Loading model from: {MODEL_PATH}")
#         # app.state.model = load_model(MODEL_PATH)
#         logger.info(f"Loading scaler from: {SCALER_PATH}")
#         with open(SCALER_PATH, 'rb') as f:
#             app.state.scaler = pickle.load(f)
#         logger.info("Scaler loaded successfully.")
#     except Exception as e:
#         logger.error(f"FATAL: Could not load model or scaler: {e}")
#         app.state.model = None
#         app.state.scaler = None

#     # Create database tables
#     logger.info("Verifying database tables...")
#     models.Base.metadata.create_all(bind=database.engine)
#     logger.info("Database tables verified.")

#     # Start MQTT client
#     try:
#         mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
#         mqtt_thread = threading.Thread(target=mqtt_client.loop_start)
#         mqtt_thread.daemon = True
#         mqtt_thread.start()
#         logger.info("MQTT client loop started in a separate thread.")
#     except Exception as e:
#         logger.error(f"Could not connect to MQTT broker: {e}")

#     yield

#     logger.info("Application shutdown...")
#     mqtt_client.disconnect()
#     logger.info("MQTT client disconnected.")

# # --- 5. FastAPI App Initialization ---
# app = FastAPI(
#     title="Fall Detection API with MQTT and Model Training",
#     description="An API to handle fall detection data, predictions via MQTT, and model retraining.",
#     lifespan=lifespan
# )

# # CORS Middleware
# origins = ["http://localhost:5173", "http://localhost:3000", "*"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- 6. Pydantic Models ---
# class LabelDataRequest(BaseModel):
#     prediction: int

# # --- 7. HTTP API Endpoints ---
# @app.get("/")
# async def read_root():
#     return {"message": "Welcome to the Fall Detection API!"}

# @app.put("/labeldata/{record_id}")
# def label_data(record_id: int, request: LabelDataRequest, db: Session = Depends(get_db)):
#     db_record = db.query(FallDetection).filter(FallDetection.id == record_id).first()
#     if not db_record:
#         raise HTTPException(status_code=404, detail="Record not found")
    
#     if request.prediction not in [0, 1]:
#         raise HTTPException(status_code=400, detail="Prediction value must be 0 (No Fall) or 1 (Fall)")

#     db_record.prediction = request.prediction
#     db_record.prediction_label = "Fall" if request.prediction == 1 else "No Fall"
#     db.commit()
#     logger.info(f"Record {record_id} labeled manually as '{db_record.prediction_label}'")
#     return {"message": f"Record {record_id} labeled successfully."}

# @app.post("/trainmodel")
# def train_model(db: Session = Depends(get_db)):
#     logger.info("Starting model training process...")

#     query = db.query(FallDetection).filter(FallDetection.prediction.in_([0, 1])).statement
#     df = pd.read_sql(query, db.bind)

#     if df.shape[0] < 50:
#         logger.warning(f"Not enough data to train. Found {df.shape[0]} records.")
#         raise HTTPException(status_code=400, detail="Not enough labeled data to train the model.")

#     features = [
#         'max_Ax', 'min_Ax', 'var_Ax', 'mean_Ax', 'max_Ay', 'min_Ay', 'var_Ay', 'mean_Ay', 
#         'max_Az', 'min_Az', 'var_Az', 'mean_Az', 
#         'max_Gx', 'min_Gx', 'var_Gx', 'mean_Gx', 'max_Gy', 'min_Gy', 'var_Gy', 'mean_Gy', 
#         'max_Gz', 'min_Gz', 'var_Gz', 'mean_Gz'
#     ]
#     X = df[features]
#     y = df['prediction']

#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     logger.info("Features scaled successfully.")

#     try:
#         model = app.state.model
#         model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#         model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
#         logger.info("Model training completed.")

#         model.save(MODEL_PATH)
#         with open(SCALER_PATH, 'wb') as f:
#             pickle.dump(scaler, f)
#         app.state.model = load_model(MODEL_PATH)
#         app.state.scaler = scaler
#         logger.info("Model and scaler saved and reloaded successfully.")

#         return {"message": "Model training completed successfully.", "dataset_size": df.shape[0]}

#     except Exception as e:
#         logger.error(f"Error during model training: {e}")
#         raise HTTPException(status_code=500, detail=f"Model training failed: {e}")

# @app.get("/fetchdata", response_model=List[schemas.FallDetection])
# def fetch_data(source_type: str = Query(..., enum=["predicted", "labelled"]), db: Session = Depends(get_db)):
#     data = db.query(FallDetection).filter(FallDetection.source_type == source_type).order_by(FallDetection.timestamp.desc()).all()
#     return data

# @app.get("/showdataset", response_model=List[schemas.FallDetection])
# def show_dataset(db: Session = Depends(get_db)):
#     dataset = db.query(FallDetection).filter(FallDetection.prediction.in_([0, 1])).order_by(FallDetection.timestamp.desc()).all()
#     return dataset

# # To run: uvicorn app:app --reload

import os
import json
import logging
import threading
import joblib  # <--- Use joblib instead of pickle/keras
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import List

import paho.mqtt.client as mqtt
from fastapi import FastAPI, Depends, HTTPException, Query, Request
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

# --- 1. Logging and Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# MQTT Configuration
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
DATA_COLLECTION_TOPIC = "datacollection"
DATA_PREDICTION_TOPIC = "dataprediction"
FALL_ALERT_TOPIC = "fallalert"
MQTT_CLIENT_ID = "fastapi-backend-client-rf" # unique id 

# Model file path
MODEL_PIPELINE_PATH = 'fall_detection_model.joblib' # Path to  joblib file

# --- 2. Database and Dependency Setup ---
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 3. MQTT Client Logic ---
mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID)

def on_mqtt_connect(client, rc):
    if rc == 0:
        logger.info("Connected to MQTT Broker successfully!")
        client.subscribe([(DATA_COLLECTION_TOPIC, 0), (DATA_PREDICTION_TOPIC, 0)])
        logger.info(f"Subscribed to topics: {DATA_COLLECTION_TOPIC}, {DATA_PREDICTION_TOPIC}")
    else:
        logger.error(f"Failed to connect to MQTT Broker. Return code: {rc}")

def on_mqtt_message(client, message):
    payload_str = message.payload.decode("utf-8")
    topic = message.topic
    logger.info(f"Received MQTT message on topic '{topic}'")

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
            logger.info(f"Stored unlabelled data for MAC: {data.get('mac_addr')} (Record ID: {db_record.id})")

        elif topic == DATA_PREDICTION_TOPIC: # Prediction topic
            logger.info("Prediction topic received. Preparing features for prediction...")
            
            # Check if the pipeline is loaded
            if not app.state.pipeline:
                logger.error("Model pipeline not loaded. Cannot perform prediction.")
                return

            # Prepare features for prediction 
            features = [
                data.get("max_Ax"), data.get("min_Ax"), data.get("var_Ax"), data.get("mean_Ax"),
                data.get("max_Ay"), data.get("min_Ay"), data.get("var_Ay"), data.get("mean_Ay"),
                data.get("max_Az"), data.get("min_Az"), data.get("var_Az"), data.get("mean_Az"),
                data.get("max_Gx"), data.get("min_Gx"), data.get("var_Gx"), data.get("mean_Gx"),
                data.get("max_Gy"), data.get("min_Gy"), data.get("var_Gy"), data.get("mean_Gy"),
                data.get("max_Gz"), data.get("min_Gz"), data.get("var_Gz"), data.get("mean_Gz")
            ]
            
            # The pipeline handles scaling and prediction in one step
            prediction = app.state.pipeline.predict([features])
            prediction_value = int(prediction[0]) # Get the single prediction result
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
            logger.info(f"Prediction stored for MAC {data.get('mac_addr')} (Record ID: {db_record.id})")

        # If a fall was detected (from prediction topic), publish an alert
        if prediction_value == 1:
            alert_payload = json.dumps({"mac_addr": data.get("mac_addr"), "alert": "Fall Detected!"})
            client.publish(FALL_ALERT_TOPIC, alert_payload)
            logger.info(f"Published fall alert for MAC: {data.get('mac_addr')}")

    except Exception as e:
        logger.error(f"Error processing MQTT message: {e}")
    finally:
        db.close()

# Assign callbacks
mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_message = on_mqtt_message

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
        mqtt_thread = threading.Thread(target=mqtt_client.loop_start)
        mqtt_thread.daemon = True
        mqtt_thread.start()
        logger.info("MQTT client loop started in a separate thread.")
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

    # 2. Prepare features (X) and labels (y)
    features = [
        'max_Ax', 'min_Ax', 'var_Ax', 'mean_Ax', 'max_Ay', 'min_Ay', 'var_Ay', 'mean_Ay', 
        'max_Az', 'min_Az', 'var_Az', 'mean_Az', 
        'max_Gx', 'min_Gx', 'var_Gx', 'mean_Gx', 'max_Gy', 'min_Gy', 'var_Gy', 'mean_Gy', 
        'max_Gz', 'min_Gz', 'var_Gz', 'mean_Gz'
    ]
    X = df[features]
    y = df['prediction']
    logger.info(f"Training dataset prepared with {len(y)} samples.")

    try:
        # 3. Define the new pipeline with a scaler and a RandomForest model
        new_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        # 4. Train the pipeline
        new_pipeline.fit(X, y)
        logger.info("Model pipeline training completed.")

        # 5. Save the newly trained pipeline over the old one
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