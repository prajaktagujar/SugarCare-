import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load dataset
diabetes_dataset = pd.read_csv('dataset/diabetes.csv')

# Preprocess data
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Save model & scaler
joblib.dump(classifier, 'model/svm_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Model and Scaler saved successfully!")
