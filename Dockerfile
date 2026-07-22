# Use the official Python 3.10 image
FROM python:3.10-slim

# Set the working directory
WORKDIR /code

# Install system dependencies required for MRZ and Barcode parsing (Layer 0)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Change the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the rest of the application code
COPY --chown=user . $HOME/app

# Hugging Face Spaces routes traffic to port 7860 by default
EXPOSE 7860

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]