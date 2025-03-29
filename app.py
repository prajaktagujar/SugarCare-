from flask import Flask, request, render_template, jsonify
import numpy as np
from diabetes_prediction import predict_diabetes  # Import prediction function

app = Flask(__name__, template_folder="templates", static_folder="static")

# Route for Home Page
@app.route("/")
def home():
    return render_template("index.html")  # Ensure index.html is inside "templates" folder

# Route for About Page
@app.route("/about")
def about():
    return render_template("about.html")

# Route for Contact Page
@app.route("/contact")
def contact():
    return render_template("contact.html")

# Route for Diabetes Prediction
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict_form.html")  # Create a simple HTML form

    try:
        # Extract form data
        data = [
            float(request.form["pregnancies"]),
            float(request.form["glucose"]),
            float(request.form["bloodPressure"]),
            float(request.form["skinThickness"]),
            float(request.form["insulin"]),
            float(request.form["bmi"]),
            float(request.form["dpf"]),
            float(request.form["age"]),
        ]

        # Predict using the imported function
        result = predict_diabetes(data)

        return render_template("result.html", prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
