# Use Python 3.11 as base image (compatible with your newer library versions)
FROM python:3.12-slim

# Set working directory
WORKDIR /observe-custom-ml-service

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables (optional)
ENV PYTHONUNBUFFERED=1

# Expose port for Flask application
EXPOSE 5000

# Modify this based on your actual entry point file
CMD ["python", "app.py"]

