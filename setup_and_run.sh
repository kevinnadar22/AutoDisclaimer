#!/bin/bash
set -e  # Exit on error

# Print colored messages
print_message() {
    echo -e "\e[1;34m>> $1\e[0m"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_message "Docker is not installed. Please install Docker first."
        print_message "Visit https://docs.docker.com/get-docker/ for installation instructions."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_message "Docker Compose is not installed. Please install Docker Compose first."
        print_message "Visit https://docs.docker.com/compose/install/ for installation instructions."
        exit 1
    fi
}

# Check if NVIDIA drivers are installed (for GPU support)
check_nvidia() {
    if command -v nvidia-smi &> /dev/null; then
        print_message "NVIDIA GPU detected. Will use GPU acceleration."
        HAS_GPU=true
    else
        print_message "No NVIDIA GPU detected. Will run in CPU mode."
        HAS_GPU=false
    fi
}

# Main setup function
setup_and_run() {
    print_message "Setting up Smoking Detection and Disclaimer App"
    
    # Check prerequisites
    check_docker
    check_nvidia
    
    # Create necessary directories
    print_message "Creating necessary directories..."
    mkdir -p output_collages
    mkdir -p annotated_images
    
    # Build and run with Docker Compose
    print_message "Building Docker image (this may take a few minutes)..."
    docker-compose build --no-cache
    
    print_message "Starting the application..."
    docker-compose up
}

# Show usage instructions after starting
show_usage() {
    print_message "Application is running!"
    print_message "Open your web browser and go to: http://localhost:8501"
    print_message ""
    print_message "Usage instructions:"
    print_message "1. Upload a video file"
    print_message "2. Configure processing options"
    print_message "3. Click 'Process Video' to detect smoking and add disclaimers"
    print_message "4. Download the processed video when complete"
    print_message ""
    print_message "To stop the application, press Ctrl+C"
}

# Run the setup
setup_and_run
show_usage 