import os
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# --- Configuration ---
DATABASE_PATH = 'fall_detection.db'
MODEL_PATH = 'neural_network_model.h5'
SCALER_PATH = 'scalar.pkl'

def train_model():
    """
    Loads labeled data from the database, trains a neural network,
    and saves the model and data scaler.
    """
    # --- 1. Load Data From Database ---
    print(f"Loading data from '{DATABASE_PATH}'...")
    if not os.path.exists(DATABASE_PATH):
        print(f"Error: Database file not found at '{DATABASE_PATH}'.")
        print("Please ensure the backend has run and generated the database.")
        return

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        # Query for all data that has been manually labeled (prediction is 0 or 1)
        query = "SELECT * from fall_detections WHERE prediction IN (0, 1)"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        print(f"Error loading data from the database: {e}")
        return

    if df.empty:
        print("No labeled data found in the database. Cannot train model.")
        return
    
    print(f"Successfully loaded {df.shape[0]} labeled records.")

    # --- 2. Prepare Data ---
    print("Preparing data for training...")
    # Define the full list of 24 features
    features = [
        'max_Ax', 'min_Ax', 'var_Ax', 'mean_Ax',
        'max_Ay', 'min_Ay', 'var_Ay', 'mean_Ay',
        'max_Az', 'min_Az', 'var_Az', 'mean_Az',
        'max_Gx', 'min_Gx', 'var_Gx', 'mean_Gx',
        'max_Gy', 'min_Gy', 'var_Gy', 'mean_Gy',
        'max_Gz', 'min_Gz', 'var_Gz', 'mean_Gz'
    ]
    target = 'prediction'

    # Check if all feature columns exist in the DataFrame
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"Error: The following feature columns are missing from the database: {missing_cols}")
        return

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 3. Scale the Feature Data ---
    # Neural networks perform best when input features are scaled.
    print("Scaling feature data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 4. Define the Neural Network ---
    print("Defining the neural network architecture...")
    model = Sequential([
        Dense(24, activation='relu', input_shape=(24,)),
        Dropout(0.2),  # Dropout layer to prevent overfitting
        Dense(12, activation='relu'),
        Dropout(0.2),
        Dense(2, activation='softmax')  # 2 output nodes for 2 classes (0=No Fall, 1=Fall)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # --- 5. Train the Model ---
    print("\nStarting model training...")
    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )

    # --- 6. Evaluate the Model ---
    print("\nEvaluating model performance on the test set...")
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # --- 7. Save the Model and Scaler ---
    print("\nSaving the trained model and scaler...")
    # Save the TensorFlow/Keras model
    model.save(MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}'")

    # Save the scaler object using pickle
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to '{SCALER_PATH}'")
    
    print("\nTraining process complete.")

if __name__ == '__main__':
    train_model()
