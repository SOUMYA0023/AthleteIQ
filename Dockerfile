FROM python:3.10-slim

# Set environment variables for CPU-only mode
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV CUDA_VISIBLE_DEVICES=""
ENV MEDIAPIPE_DISABLE_GPU=1

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

# Copy pinned requirements
COPY requirements_unified.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements_unified.txt

# Verify MediaPipe installation
RUN python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__); print('Solutions module accessible:', hasattr(mp, 'solutions')); print('Pose available:', hasattr(getattr(mp, 'solutions', {}), 'pose'))"

# Copy application code
COPY . .

# Expose port
EXPOSE 8501