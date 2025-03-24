import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STYLES_DIR = os.path.join(BASE_DIR, 'styles')

# Model paths
MODEL_PATHS = {
    'default': {
        'prototxt': os.path.join(MODELS_DIR, 'models_colorization_deploy_v2.prototxt'),
        'model': os.path.join(MODELS_DIR, 'colorization_release_v2.caffemodel'),
        'points': os.path.join(MODELS_DIR, 'pts_in_hull.npy')
    }
}

# Style transfer models
STYLE_MODELS = {
    'vgg19': os.path.join(MODELS_DIR, 'vgg19.h5'),
    'resnet50': os.path.join(MODELS_DIR, 'resnet50.h5')
}

# Image processing settings
IMAGE_SETTINGS = {
    'max_size': 800,
    'quality': 95,
    'supported_formats': ['jpg', 'jpeg', 'png', 'tiff']
}

# Color adjustment ranges
COLOR_RANGES = {
    'saturation': (0.0, 2.0),
    'brightness': (-50, 50),
    'contrast': (0.5, 2.0)
}

# Batch processing settings
BATCH_SETTINGS = {
    'max_images': 10,
    'max_video_duration': 30  # seconds
} 