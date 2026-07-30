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

# Create user with UID 1000 for Hugging Face Spaces security
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FaceNet weights (Facenet models are about 90MB)
# We can use a script to trigger the download during build
COPY --chown=user src/ /app/src/
COPY --chown=user scripts/ /app/scripts/
RUN mkdir -p /home/user/.deepface/weights
ADD --chown=user https://github.com/serengil/deepface_models/releases/download/v1.0/facenet_weights.h5 /home/user/.deepface/weights/facenet_weights.h5

# Copy the rest of the application
COPY --chown=user . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
