from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load trained model & scaler
MODEL_DIR = "model"
MODEL_FILE = "svm_model.pkl"
SCALER_FILE = "scaler.pkl"

model_path = os.path.join(MODEL_DIR, MODEL_FILE)
scaler_path = os.path.join(MODEL_DIR, SCALER_FILE)

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Error: Model or scaler file not found!")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print("Model and scaler loaded successfully!")

# Route for home page
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
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("diabetes_prediction.html")

    try:
        # Get form data
        data = request.form

        # Extract features from form
        input_data = [
            float(data["pregnancies"]),
            float(data["glucose"]),
            float(data["bloodPressure"]),
            float(data["skinThickness"]),
            float(data["insulin"]),
            float(data["bmi"]),
            float(data["dpf"]),
            float(data["age"])
        ]

        # Convert input to NumPy array
        input_data_np = np.asarray(input_data).reshape(1, -1)

        # Standardize input data
        std_data = scaler.transform(input_data_np)

        # Make prediction
        prediction = model.predict(std_data)
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
