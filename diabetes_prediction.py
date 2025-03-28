import numpy as np
import joblib

# Load trained model & scaler
classifier = joblib.load('model/svm_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Sample input data
input_data = (5,166,72,19,175,25.8,0.587,51)

# Preprocess input
input_data_np = np.asarray(input_data).reshape(1, -1)
std_data = scaler.transform(input_data_np)

# Predict
prediction = classifier.predict(std_data)

if prediction[0] == 0:
    print('The person is NOT diabetic')
else:
    print('The person IS diabetic')
