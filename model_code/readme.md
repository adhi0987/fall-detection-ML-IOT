# model_maker.py Documentation

This script trains a Support Vector Machine (SVM) classifier for fall detection using features extracted from sensor data. It performs data preprocessing, feature selection, model training, evaluation, and saves the trained model and scaler for later use.

## Main Steps

1. **Import Libraries:**  
   Uses pandas, numpy, scikit-learn, and pickle for data handling, model training, and serialization.

2. **Load Dataset:**  
   Reads `augmented_dataset.csv` into a DataFrame.

3. **Feature Selection:**  
   Selects 17 columns (16 features + 1 label) from the dataset for model training.

4. **Split Data:**  
   Splits the data into training and test sets (80% train, 20% test).

5. **Standardization:**  
   Scales features using `StandardScaler` to improve model performance.

6. **(Optional) Validation Split:**  
   Optionally splits part of the training data for validation (not used in final training).

7. **Model Training:**  
   Trains an SVM classifier (`SVC` with RBF kernel and C=100) on the scaled training data.

8. **Model Evaluation:**  
   Predicts on the test set and prints the test accuracy.

9. **Save Model and Scaler:**  
   Saves the trained SVM model as `svm_model.pkl` and the scaler as `scaler.pkl` using pickle.

10. **(Optional) Additional Metrics:**  
    Includes commented-out code for confusion matrix, sensitivity, specificity, and accuracy calculations.

## Key Lines Explained

```python
# Load and select features
df = pd.read_csv('augmented_dataset.csv')
cols = [ ... ]  # 16 features + 'label'
df = df[cols]

# Split features and labels
y = df['label']
X = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM model
model = SVC(C=100, kernel='rbf')
model.fit(X_train_scaled, y_train)

# Evaluate model
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
print("Test accuracy:", accuracy)

# Save model and scaler
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)