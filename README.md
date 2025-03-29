# Diabetes Prediction Web App

## Overview
The **Diabetes Prediction Web App** is an interactive machine learning-based tool that allows users to predict the likelihood of having diabetes based on medical parameters. It uses an **SVM (Support Vector Machine) model** trained on a diabetes dataset and is deployed using **Flask**.

## Features
- **User-friendly Web Interface**: Users can enter their medical details and get instant predictions.
- **Machine Learning Model**: Uses **SVM** for classification.
- **Interactive Diabetes Pedigree Calculation**: Users can check their diabetes pedigree risk using a pop-up form.
- **Flask Backend**: Serves the model and handles user input.
- **Bootstrap-based UI**: Clean and responsive design.

## File Structure
```
Diabetes_Prediction_App/
│── static/                     # Stores images, CSS, JavaScript
│   ├── styles.css              # Custom CSS for styling
│   ├── diabetes-image.png      # Images used in the project
│
│── templates/                  # HTML templates for Flask
│   ├── index.html              # Home page
│   ├── predict_form.html       # Form for user input
│   ├── result.html             # Displays prediction result
│
│── model/                      # ML Model and Scaler
│   ├── svm_model.pkl           # Trained SVM model
│   ├── scaler.pkl              # Standard Scaler for preprocessing
│
│── app.py                      # Flask backend
│── diabetes_prediction.py      # Contains prediction logic
│── requirements.txt            # Required Python packages
│── README.md                   # Project documentation
```

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Diabetes-Prediction-WebApp.git
cd Diabetes-Prediction-WebApp
```
### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Flask App
```bash
python app.py
```
### 5. Open in Browser
Go to `http://127.0.0.1:5000/` to access the web app.

## Usage
1. Enter medical parameters like glucose level, BMI, age, etc.
2. Click on "Predict Diabetes".
3. View the prediction result (Diabetic / Not Diabetic).

## Technologies Used
- **Python** (Flask, NumPy, Scikit-learn, Joblib)
- **HTML, CSS, JavaScript** (Frontend UI)
- **Bootstrap** (For responsive design)

## Deployment
This web app can be deployed on platforms like **Heroku**, **Render**, or **AWS**. i have deployed it on railway app
- `requirements.txt` is up to date.
- A `Procfile` is created (if using Heroku).
- The `model/` folder is included.


