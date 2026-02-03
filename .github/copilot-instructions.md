# AI Coding Agent Instructions for Scholarship Eligibility System

## Project Overview
This is a **scholarship eligibility prediction system** that combines **hard rule validation** with **AI/ML classification**. It features a Streamlit web interface for interactive predictions, batch processing capabilities, and model training pipelines.

## Architecture & Data Flow

### Core Components
1. **Model Training** (`train_model.py`) - TensorFlow/Keras neural network trained on synthetic scholarship data
2. **Web Interface** (`app.py`) - Streamlit dashboard for individual predictions and analytics
3. **Batch Processing** (`batch_predict.py`) - Command-line tool for bulk eligibility evaluation
4. **Data Generation** (`generate_data.py`) - Creates synthetic training dataset with realistic scholarship patterns

### Prediction Pipeline (Critical Pattern)
The system applies a **two-stage eligibility filter**:
- **Stage 1 (Hard Rules)**: Filter applicants by minimum thresholds (GPA ≥ 2.5, Credit Hours ≥ 100)
- **Stage 2 (AI Model)**: Run remaining candidates through trained neural network with sigmoid threshold (0.5)

**Key file**: [batch_predict.py](batch_predict.py#L39-L48) shows this hybrid approach clearly. Always maintain this order—hard rules must be applied before AI evaluation.

### Data Flow
```
generate_data.py → dataset.csv → train_model.py → scholarship_model.h5
                                                  ↓
dataset.csv → batch_predict.py → batch_results.csv
                                  ↓ (displays via)
                              app.py (Streamlit UI)
```

## Configuration System
The `scholarship_config.json` file defines three scholarship types with eligibility criteria:
- **Merit Based**: High GPA (3.5+), attendance (85+)
- **Need Based**: Lower GPA (2.5+), prioritizes financial_score ≤ 40
- **Sports Excellence**: Lower GPA (2.2+), attendance (60+)

When modifying rules, **update this JSON file** to keep criteria centralized. See [app.py](app.py#L14-L22) for config loading pattern.

## Key Features & Workflows

### Development Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Generate training data: `python generate_data.py`
3. Train model: `python train_model.py` (outputs `scholarship_model.h5` and `scaler.pkl`)
4. Run web app: `streamlit run app.py`
5. Batch predictions: `python batch_predict.py --input input.csv --output output.csv`

### Scaling & Preprocessing
- **StandardScaler** (fit on training data) scales features to 0-1 range
- Scaler is saved as `scaler.pkl` and must be used for both batch and web predictions
- Features: `['gpa', 'attendance', 'financial_score', 'credit_hours']`

### Important: Model Inputs
All predictions expect exactly 4 features with correct scaling. Missing or incorrect feature order will cause silent failures. See [train_model.py](train_model.py#L16-L17) for canonical feature order.

## Docker Deployment
Build and run: `docker build -t scholarship-system . && docker run -p 8501:8501 scholarship-system`

The Dockerfile exposes port 8501 (Streamlit default) and respects the `PORT` environment variable for cloud deployments.

## Common Patterns to Preserve

1. **CSV Column Validation** - Both `batch_predict.py` and `app.py` validate required columns exist before processing
2. **Error Handling** - Missing model/scaler files should fail gracefully with clear error messages
3. **Probability Thresholding** - Always use 0.5 threshold for converting sigmoid output to binary class
4. **Numpy Compatibility** - `batch_predict.py` includes numpy version patch for newer environments

## Testing & Debugging
- `debug_model.py` - Available for model inspection and diagnostics
- Test batch predictions with `dataset.csv` as input
- Check `batch_results.csv` for result format validation

## Git Workflow
Always push changes with clear commit messages describing which component was modified (e.g., "Update AI model training hyperparameters" or "Add new scholarship category to config").
