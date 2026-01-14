FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Streamlit default)
EXPOSE 8501

# Run the application
# Note: For Google Cloud Run, it will provide a PORT env var.
# We use shell formatting to allow variable expansion, defaulting to 8501.
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
