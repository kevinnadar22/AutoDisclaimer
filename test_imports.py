#!/usr/bin/env python3
"""
Test script to verify that imports from the utils package are working correctly.
"""

print("Testing imports from utils package...")

try:
    from utils.split import split_video_and_create_collages
    print("✓ Successfully imported split_video_and_create_collages")
except ImportError as e:
    print(f"✗ Failed to import split_video_and_create_collages: {e}")

try:
    from utils.process_smoking_detection import load_model, process_image
    print("✓ Successfully imported load_model and process_image")
except ImportError as e:
    print(f"✗ Failed to import load_model or process_image: {e}")

try:
    from utils.add_smoking_disclaimer import add_disclaimer_to_video
    print("✓ Successfully imported add_disclaimer_to_video")
except ImportError as e:
    print(f"✗ Failed to import add_disclaimer_to_video: {e}")

try:
    from utils import process_video_with_disclaimer
    print("✓ Successfully imported process_video_with_disclaimer from utils")
except ImportError as e:
    print(f"✗ Failed to import process_video_with_disclaimer from utils: {e}")

print("\nImport test completed.") 