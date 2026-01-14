import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# 2. Preprocessing
print("Preprocessing data...")
# Features and Target
# Features and Target
X = df[['gpa', 'attendance', 'financial_score', 'credit_hours']]
y = df['scholarship']

# Encode Labels (Eligible -> 1, Not Eligible -> 0)
# We want Eligible=1.
y = df['scholarship'].apply(lambda x: 1 if x == 'Eligible' else 0).values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

# 3. Build ANN Model (MLP)
print("Building model...")
model = tf.keras.models.Sequential([
    # Input Layer (4 features) -> Hidden Layer 1
    tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
    # Hidden Layer 2
    tf.keras.layers.Dense(4, activation='relu'),
    # Output Layer (Binary Classification)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 4. Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train Model
print("Training model...")
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.1)

# 6. Evaluate
print("Evaluating model...")
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Not Eligible', 'Eligible']))

# 7. Save Model
model.save('scholarship_model.h5')
print("Model saved as scholarship_model.h5")
