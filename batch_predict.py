import pandas as pd
import numpy as np
# Patch for compatibility with newer NumPy versions
if not hasattr(np, 'object'):
    np.object = object

import tensorflow as tf
import joblib
import argparse
import sys
import os

def batch_predict(input_file, output_file, model_path='scholarship_model.h5', scaler_path='scaler.pkl'):
    """
    Determines eligibility for a batch of students.
    """
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    required_columns = ['gpa', 'attendance', 'financial_score', 'credit_hours']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Input CSV must contain columns: {required_columns}")
        return

    print("Loading model and scaler...")
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        print("Please ensure 'scholarship_model.h5' and 'scaler.pkl' exist.")
        return

    # Initialize prediction column
    df['predicted_eligibility'] = 'Pending'
    df['reason'] = ''

    # --- Step 1: Apply Hard Rules ---
    # Rule: Credit Hours < 100 OR GPA < 2.5 => Not Eligible
    print("Applying hard eligibility rules...")
    mask_fail_rules = (df['credit_hours'] < 100) | (df['gpa'] < 2.5)
    
    df.loc[mask_fail_rules, 'predicted_eligibility'] = 'Not Eligible'
    df.loc[mask_fail_rules, 'reason'] = 'Did not meet minimum GPA (2.5) or Credit Hours (100) requirement'

    # --- Step 2: Apply AI Model to remaining candidates ---
    mask_candidates = ~mask_fail_rules
    candidate_count = mask_candidates.sum()
    
    if candidate_count > 0:
        print(f"Evaluating {candidate_count} candidates with AI model...")
        
        # Prepare features for model
        features = df.loc[mask_candidates, required_columns]
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        # Sigmoid output is probability (0 to 1)
        pred_probs = model.predict(features_scaled, verbose=0)
        
        # Convert probability to class (Threshold 0.5)
        # 1 = Eligible, 0 = Not Eligible
        pred_classes = (pred_probs > 0.5).astype(int).flatten()
        
        # Map back to string labels
        pred_labels = np.where(pred_classes == 1, 'Eligible', 'Not Eligible')
        
        df.loc[mask_candidates, 'predicted_eligibility'] = pred_labels
        df.loc[mask_candidates, 'reason'] = 'Evaluated by AI Model'
        
    else:
        print("No candidates met the minimum hard requirements for AI evaluation.")

    # --- Save Results ---
    print(f"Saving results to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")
    
    # Summary
    print("\nSummary Results:")
    print(df['predicted_eligibility'].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Student Eligibility Predictor')
    parser.add_argument('--input', type=str, default='dataset.csv', help='Path to input CSV file')
    parser.add_argument('--output', type=str, default='batch_results.csv', help='Path to output CSV file')
    
    args = parser.parse_args()
    
    batch_predict(args.input, args.output)
