import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 50000

# Generate random features first
# GPA: 0.0 to 4.0
gpa = np.round(np.random.uniform(2.0, 4.0, n_samples).astype(np.float32), 2)
# Attendance: 50 to 100
attendance = np.round(np.random.uniform(60, 100, n_samples).astype(np.float32), 1)
# Financial Score: 0 to 100 (Lower score = Higher Need)
financial_score = np.round(np.random.uniform(0, 100, n_samples).astype(np.float32), 1)
# Credit Hours: 0 to 130 (assuming typical undergrad program)
credit_hours = np.round(np.random.uniform(0, 130, n_samples).astype(np.float32), 0)

# Logic for Scholarship (Relaxed & Credit Based)
# Score components:
# GPA (High weight): 40%
# Credit Hours (Progress weight): 30% (More credits = likely closer to grad/proven track record)
# Attendance (Moderate weight): 20%
# Financial Score (Need based): 10% (Lower score = Higher Need)

norm_gpa = (gpa / 4.0) * 100        # 0-100
norm_att = attendance               # 0-100
norm_financial = 100 - financial_score # 0(Rich) to 100(Poor/High Need)
norm_credits = (credit_hours / 130.0) * 100 # 0-100

# Weights
score = (0.4 * norm_gpa) + (0.3 * norm_credits) + (0.2 * norm_att) + (0.1 * norm_financial)

# Add small noise
score += np.random.normal(0, 2, n_samples)

# Threshold at 20th percentile (Relaxed - 80% Acceptance)
threshold = np.percentile(score, 20)

scholarship = (score > threshold).astype(int)

# HARD RULES: Overwrite eligibility based on user constraints
# Must have >= 100 Credit Hours AND >= 2.5 GPA
scholarship = np.where((credit_hours < 100) | (gpa < 2.5), 0, scholarship)

scholarship_labels = np.where(scholarship == 1, 'Eligible', 'Not Eligible')

# Create DataFrame
df = pd.DataFrame({
    'gpa': gpa,
    'attendance': attendance,
    'financial_score': financial_score,
    'credit_hours': credit_hours,
    'scholarship': scholarship_labels
})

# Save to CSV
df.to_csv('dataset.csv', index=False)
print(f"Dataset generated with {n_samples} samples.")
print(f"Balance:\n{df['scholarship'].value_counts(normalize=True)}")

