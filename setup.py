import os
import sys
import subprocess
import torch

def check_cuda():
    """Check if CUDA is available and print GPU information."""
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA is not available. Using CPU (processing will be slower)")
        return False

def create_directories():
    """Create necessary directories for the project."""
    directories = [
        'models',
        'utils',
        'temp',
        'output',
        'model_checkpoints'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def install_dependencies():
    """Install required dependencies."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed dependencies")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def main():
    print("Setting up AI Image Colorization project...")
    
    # Check CUDA availability
    check_cuda()
    
    # Create necessary directories
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    print("\nSetup completed successfully!")
    print("\nTo run the project:")
    print("1. Activate your virtual environment (if using one)")
    print("2. Run: streamlit run app.py")
    print("\nNote: The first run may take longer as it downloads the AI models.")

if __name__ == "__main__":
    main() 