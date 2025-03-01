"""
Smoking Detection & Disclaimer Adder - Utility Functions

This package contains utility functions for video processing, smoking detection, and disclaimer addition.
"""

# Import key functions for easier access
from .split import split_video_and_create_collages
from .process_smoking_detection import load_model, process_image
from .add_smoking_disclaimer import add_disclaimer_to_video as process_video_with_disclaimer 