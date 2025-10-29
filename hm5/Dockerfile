# Dockerfile
FROM agrigorev/zoomcamp-model:2025

# Install dependencies
COPY pyproject.toml .
RUN pip install uv
RUN uv pip install --system fastapi uvicorn scikit-learn

# Copy FastAPI script
COPY app.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]