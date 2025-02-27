FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libvips \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt \
    && pip install accelerate einops streamlit

# Copy the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"] 