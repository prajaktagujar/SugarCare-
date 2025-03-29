import numpy as np
import joblib
import os

# Define model paths
MODEL_DIR = "model"
MODEL_FILE = "svm_model.pkl"
SCALER_FILE = "scaler.pkl"

model_path = os.path.join(MODEL_DIR, MODEL_FILE)
scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)

# Ensure model and scaler exist before loading
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Error: Model or scaler file not found!")

# Load trained model & scaler
classifier = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Model and scaler loaded successfully!")

# Function to predict diabetes
def predict_diabetes(input_data):
    try:
        # Convert input to NumPy array
        input_data_np = np.asarray(input_data).reshape(1, -1)

        # Standardize input
        std_data = scaler.transform(input_data_np)

        # Predict
        prediction = classifier.predict(std_data)

        return "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    
    except Exception as e:
        return f"Error: {str(e)}"
