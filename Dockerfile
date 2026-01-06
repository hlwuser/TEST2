FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first (for better caching)
COPY requirement.txt .

# Install only production dependencies
RUN pip install --no-cache-dir \
    flask \
    joblib \
    pandas \
    numpy \
    scikit-learn

# Copy only necessary files for the app
COPY app.py .
COPY templates/ templates/
COPY model/model.pkl model/

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]