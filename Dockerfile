# Use Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and DeepFace
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FaceNet weights (Facenet models are about 90MB)
# We can use a script to trigger the download during build
COPY src/ /app/src/
COPY scripts/ /app/scripts/
RUN mkdir -p /root/.deepface/weights
ADD https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5 /root/.deepface/weights/facenet_weights.h5

# Copy the rest of the application
COPY . .

# Set entrypoint to the inference CLI
ENTRYPOINT ["python", "scripts/inference.py"]
CMD ["--help"]
