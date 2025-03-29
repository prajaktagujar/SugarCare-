from flask import Flask, request, render_template, jsonify, send_file
import os
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates", static_folder="static")

# Define model and scaler paths
MODEL_DIR = "model"
MODEL_FILE = "svm_model.pkl"
SCALER_FILE = "scaler.pkl"

model_path = os.path.join(MODEL_DIR, MODEL_FILE)
scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)

# Ensure model and scaler exist before loading
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Error: Model or scaler file not found! Ensure they are inside the 'model/' folder.")

# Load the trained SVM model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("Model and scaler loaded successfully!")

# Route for Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Route for About Page
@app.route("/about")
def about():
    return render_template("about.html")

# Route for Contact Page
@app.route("/contact")
def contact():
    return render_template("contact.html")

# Route for Diabetes Prediction Page
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        required_features = ["pregnancies", "glucose", "bloodPressure", "skinThickness",
                             "insulin", "bmi", "dpf", "age"]

        # Check if all required fields are in the request
        if not all(feature in data for feature in required_features):
            return jsonify({"error": "Missing input data fields."}), 400

        # Convert inputs to numpy array
        input_data = np.array([[float(data[feature]) for feature in required_features]])

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        output = bool(prediction[0])  # Convert to boolean for JSON response

        return jsonify({"prediction": output})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Log the error in the console
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    from waitress import serve  # For production servers
    serve(app, host="0.0.0.0", port=5000)
