# Import necessary libraries


# Check for TensorFlow GPU support
print("Checking TensorFlow GPU support...")
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # List physical devices
    physical_devices = tf.config.list_physical_devices()
    print("Physical devices:", physical_devices)
    
    # Check specifically for GPU devices
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"TensorFlow is using GPU: {len(gpu_devices)} GPU(s) available")
        for i, gpu in enumerate(gpu_devices):
            print(f"  GPU {i}: {gpu}")
    else:
        print("TensorFlow is NOT using GPU")
        
    # Additional TensorFlow GPU info
    if tf.test.is_built_with_cuda():
        print("TensorFlow is built with CUDA support")
    else:
        print("TensorFlow is NOT built with CUDA support")
except ImportError:
    print("TensorFlow is not installed")
except Exception as e:
    print(f"Error checking TensorFlow GPU: {e}")

# Check for PyTorch GPU support
print("\nChecking PyTorch GPU support...")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("PyTorch is using GPU")
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Number of available GPU devices: {device_count}")
        
        # Print information for each GPU
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
            print(f"    Memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    else:
        print("PyTorch is NOT using GPU")
except ImportError:
    print("PyTorch is not installed")
except Exception as e:
    print(f"Error checking PyTorch GPU: {e}")
