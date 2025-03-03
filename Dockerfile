FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10.6 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    libvips \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10.6 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt 

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p /app/uploads /app/downloads /app/static /app/templates

# Copy the application code
COPY . .

# Command to run the Flask app
CMD ["python3", "gradio_app.py"] 