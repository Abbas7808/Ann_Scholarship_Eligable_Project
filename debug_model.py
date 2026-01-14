import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load artifacts
print("Loading artifacts...")
try:
    model = tf.keras.models.load_model('scholarship_model.h5')
    scaler = joblib.load('scaler.pkl')
    print("Artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")
    exit()

# Define test cases
# gpa, attendance, study_hours, financial_score
test_cases = [
    ("Default (Good)", [3.5, 85.0, 6.0, 40.0]),
    ("Perfect", [4.0, 100.0, 10.0, 0.0]), # 0 financial score = High Need
    ("Poor", [2.0, 60.0, 2.0, 90.0]),    # 90 financial score = Low Need
    ("Average", [3.0, 80.0, 5.0, 50.0])
]

print("\nRunning Predictions:")
print("-" * 50)
print(f"{'Case':<15} | {'Prob':<10} | {'Prediction':<15}")
print("-" * 50)

for name, input_vals in test_cases:
    # Scale
    input_scaled = scaler.transform([input_vals])
    
    # Predict
    prob = model.predict(input_scaled, verbose=0)[0][0]
    pred = "Eligible" if prob > 0.5 else "Not Eligible"
    
    print(f"{name:<15} | {prob:.4f}     | {pred:<15}")

# Check Dataset Balance
print("\nDataset Balance:")
df = pd.read_csv('dataset.csv')
print(df['scholarship'].value_counts(normalize=True))
