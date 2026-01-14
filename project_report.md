# Scholarship Eligibility Predictor - Project Report

## 1. Problem Statement
Universities receive thousands of scholarship applications annually. Manual processing is time-consuming and prone to human error or bias. There is a need for an automated, data-driven system to filter candidates fairly based on academic merit and financial need.

## 2. Proposed Solution
An **Artificial Neural Network (ANN)** based system that learns patterns from historical student data to predict eligibility.
- **Input**: GPA, Attendance, Study Hours, Financial Score.
- **Output**: Classification (Eligible / Not Eligible).
- **Interface**: A user-friendly Streamlit web application for real-time demonstration.

## 3. Dataset Description
The system is trained on a synthetic dataset (`dataset.csv`) comprising 1000 student records.
- **gpa** (0.0-4.0): Academic performance.
- **attendance** (0-100): Class consistency.
- **study_hours** (0-10): Effort indicator.
- **financial_score** (0-100): Economic status (Lower score = Higher need).

## 4. ANN Architecture (MLP)
We utilized a Multi-Layer Perceptron (MLP) implemented in TensorFlow/Keras.

- **Input Layer**: 4 Neurons (matching the 4 input features).
- **Hidden Layer 1**: 8 Neurons, **ReLU** activation (captures non-linear patterns).
- **Hidden Layer 2**: 4 Neurons, **ReLU** activation (further abstraction).
- **Output Layer**: 1 Neuron, **Sigmoid** activation (outputs probability between 0 and 1).

**Training Configuration:**
- **Optimizer**: Adam (Adaptive Moment Estimation).
- **Loss Function**: Binary Crossentropy (standard for binary classification).
- **Epochs**: 50.

## 5. Technical Stack
- **Python**: Core programming language.
- **TensorFlow/Keras**: Deep Learning framework.
- **Scikit-learn**: Data preprocessing (StandardScaler).
- **Pandas/NumPy**: Data manipulation.
- **Streamlit**: Web interface for demonstration.

## 6. Viva Questions & Answers
**Q: Why use an ANN instead of Logistic Regression?**
A: ANN can capture complex, non-linear relationships between features (e.g., a student with low GPA might still be eligible if their financial need is extreme, but only if attendance is high). Linear models might miss these nuances.

**Q: What is the purpose of the Sigmoid function?**
A: It squashes the final output to a range of 0 to 1, representing the probability of being "Eligible".

**Q: Why do we scale the features?**
A: Features like GPA (0-4) and Attendance (0-100) have different scales. Scaling ensures the ANN converges faster and isn't biased towards larger numbers.

## 7. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train model (optional if already saved): `python train_model.py`
3. Run App: `streamlit run app.py`
