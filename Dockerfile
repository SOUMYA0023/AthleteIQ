FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_unified.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_unified.txt

# Clean install of mediapipe to ensure proper setup
RUN pip uninstall -y mediapipe opencv-python opencv-python-headless || true
RUN pip install --no-cache-dir opencv-python-headless>=4.8.0
RUN pip install --no-cache-dir --force-reinstall mediapipe>=0.10.30
RUN python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__); print('Solutions module accessible:', hasattr(mp, 'solutions')); print('Pose available:', hasattr(mp.solutions, 'pose'))"

# Copy application code
COPY . .

# Expose port
EXPOSE 8501