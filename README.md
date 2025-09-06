# Fall Detection ML IoT

A full-stack project for real-time fall detection using IoT devices and machine learning. This project collects sensor data from IoT devices, processes it with a backend API, and visualizes analytics in a web frontend.

## Features

- **IoT Device Integration:** Collects real-time sensor data (e.g., accelerometer, gyroscope) from multiple devices.
- **Machine Learning:** Predicts falls using trained ML models.
- **REST API:** Backend API for device management and analytics.
- **Frontend Dashboard:** Visualizes device data and analytics.
- **Device List:** View and select devices by MAC address.

## Project Structure

```
fall-detection-ML-IOT/
│
├── backend/                # Backend API (Python/FastAPI/Flask)
├── frontend/               # React frontend
│   └── iot-frontend/
│       ├── src/
│       │   └── components/
│       │       └── Devicelist.tsx
│       └── ...
├── data/                   # Sample datasets and logs
├── models/                 # Trained ML models
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
    python app.py
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
   The frontend will run at `http://localhost:3000`.

### Device Data

- Devices send sensor data to the backend API.
- The frontend fetches device lists and analytics from the backend.

## Usage

- Open the frontend in your browser.
- Select a device from the device list to view analytics.
- Monitor real-time fall detection results.

## API Endpoints

- `GET /getdevices` — List unique device MAC addresses.
- `GET /analytics?mac=...` — Get analytics for a specific device.
- (Add more endpoints as needed.)

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes
4. Push to your fork and open a pull request


## Acknowledgements

- Open source IoT and ML libraries
- Community