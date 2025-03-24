import sys
import torch
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from models.colorization import ColorizationModel
from utils.video_processing import VideoColorizer

def test_imports():
    """Test if all required packages are installed."""
    print("Testing imports...")
    try:
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        import streamlit as st
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Error importing packages: {e}")
        return False

def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"✓ CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("⚠ CUDA is not available. Using CPU (processing will be slower)")
        return False

def test_model_loading():
    """Test if the colorization model can be loaded."""
    print("\nTesting model loading...")
    try:
        model = ColorizationModel()
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_video_processing():
    """Test video processing functionality."""
    print("\nTesting video processing...")
    try:
        model = ColorizationModel()
        video_colorizer = VideoColorizer(model)
        print("✓ Video processing components initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Error initializing video processing: {e}")
        return False

def main():
    print("Running project setup tests...\n")
    
    tests = [
        ("Package Imports", test_imports),
        ("CUDA Availability", test_cuda),
        ("Model Loading", test_model_loading),
        ("Video Processing", test_video_processing)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        if not test_func():
            all_passed = False
            print(f"✗ {test_name} failed")
        else:
            print(f"✓ {test_name} passed")
    
    if all_passed:
        print("\n✓ All tests passed! The project is ready to run.")
        print("\nTo start the application:")
        print("1. Activate your virtual environment (if using one)")
        print("2. Run: streamlit run app.py")
    else:
        print("\n✗ Some tests failed. Please check the errors above and fix them before running the project.")
        sys.exit(1)

if __name__ == "__main__":
    main() 