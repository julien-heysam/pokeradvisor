# Base image with Python 3.11 installed
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the package code into the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Run the application
CMD uvicorn src.main:app --host 0.0.0.0 --port 8000
