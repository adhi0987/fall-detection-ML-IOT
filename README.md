# Fall Detection ML IoT

A full-stack project for real-time fall detection using IoT devices and machine learning. This project collects sensor data from IoT devices, processes it with a backend API, and visualizes analytics in a web frontend.

## Features

- **IoT Device Integration:** Collects real-time sensor data from MPU6050 sensor
- **Machine Learning:** Predicts falls using trained ML models.
- **REST API:** Backend API for device management and analytics.
- **Frontend Dashboard:** Visualizes device data and analytics.

## Project Structure

```
fall-detection-ML-IOT/
│
├── backend/                # Backend API (FastAPI)
├── frontend/               # React frontend
│   └── iot-frontend/
│       ├── src/
│       │   ├── components/
│       │   │         └── Navbar.tsx
│       │   └── pages /
│       │             ├── PredictionAnalysis.tsx
│       │             └── Training.tsx
│       └── ...
├── hardware/                   # Hardware Code for ESP32 board
│         └── ESP32_code.ino
├── model_code/                 # Individual Model Training Code 
├── README.md
└── ...
```

## Getting Started

### Prerequisites

- Node.js & npm (for frontend)
- Python 3.x (for backend)
- (Optional) Virtual environment for Python

### Backend Setup

1. Navigate to the backend directory:
    ```sh
    cd backend
    ```
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the backend server:
    ```sh
    uvicorn  app:app --reload
    ```
   The backend should be available at `http://localhost:8000` or as configured.

### Frontend Setup

1. Navigate to the frontend directory:
    ```sh
    cd frontend/iot-frontend
    ```
2. Install dependencies:
    ```sh
    npm install
    ```
3. Start the development server:
    ```sh
    npm start
    ```
   The frontend will run at `http://localhost:5173/`.

### Device Data

- IoT device post the data to the `datacollection` or `dataprediction` topic.
- The frontend fetches device lists and analytics from the backend.

## Usage

- Open the frontend in your browser.
- `Analytics`: to see the analytics or predictions from the ml model for the last 10 minutes 
- ` Training & Labelling` : To Label the Raw Data Send from the IoT device and can to re train the model on the newly labelled data 

Below are the HTTP and WebSocket endpoints 

- GET /
  - Description: Health / welcome message.
  - Response (200): { "message": "Welcome to the Fall Detection API!" }

- WebSocket /ws
  - Description: Real-time broadcast channel for new records and predictions. Frontend connects to receive live updates.
  - Usage: open a WS connection to ws://{host}/ws and listen for JSON messages.

- PUT /labeldata/{record_id}
  - Description: Manually label a stored record.
  - Path params: record_id (int)
  - Body (application/json): { "prediction": 0 | 1 }
  - Success (200): { "message": "Record {id} labeled successfully." }
  - Errors: 400 if prediction not 0/1, 404 if record not found

- POST /trainmodel
  - Description: Trigger server-side retraining of the model using labelled data in the DB.
  - Body: none (optional params not implemented)
  - Success (200): { "message": "Model training completed successfully.", "dataset_size": <n> }
  - Errors: 400 if not enough labelled data, 500 on training failure
  - Notes: Trained pipeline is saved to disk and reloaded into app.state.pipeline.

- GET /fetchdata?source_type={predicted|labelled}
  - Description: Fetch recent records filtered by source_type.
  - Query params: source_type (required) — "predicted" or "labelled"
  - Response (200): JSON array of FallDetection records (see schemas for fields)
  - Example: GET /fetchdata?source_type=predicted

- GET /showdataset
  - Description: Return dataset of records that have a prediction value (0 or 1).
  - Response (200): JSON array of FallDetection records (ordered by newest first)

MQTT topics used by the backend (for devices / edge clients)
- datacollection — devices publish raw/label data here (backend stores as source_type="labelled")
- dataprediction — devices publish features for immediate prediction (backend predicts, stores as source_type="predicted")
- fallalert — backend publishes alerts here when a fall is predicted

