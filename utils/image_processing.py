import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
from config import IMAGE_SETTINGS, COLOR_RANGES

def preprocess_image(image):
    """Preprocess image for colorization."""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize if too large
    height, width = image.shape[:2]
    if max(height, width) > IMAGE_SETTINGS['max_size']:
        scale = IMAGE_SETTINGS['max_size'] / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    
    return image

def enhance_image(image, brightness=0, contrast=1.0, saturation=1.0):
    """Apply image enhancements."""
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Apply enhancements
    if brightness != 0:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.0 + brightness/100.0)
    
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast)
    
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(saturation)
    
    return np.array(pil_image)

def detect_faces(image):
    """Detect faces in the image."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def apply_style_transfer(content_image, style_image, model_name='vgg19'):
    """Apply neural style transfer."""
    # Load the style transfer model
    model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    
    # Preprocess images
    content_image = tf.keras.applications.vgg19.preprocess_input(content_image)
    style_image = tf.keras.applications.vgg19.preprocess_input(style_image)
    
    # Get style and content features
    content_features = model(content_image)
    style_features = model(style_image)
    
    # Apply style transfer (simplified version)
    # In a real implementation, you would use a more sophisticated style transfer algorithm
    mixed_features = 0.7 * content_features + 0.3 * style_features
    
    # Convert back to image
    result = tf.keras.applications.vgg19.deprocess_input(mixed_features)
    return result

def batch_process_images(images):
    """Process multiple images."""
    results = []
    for image in images:
        processed = preprocess_image(image)
        results.append(processed)
    return results

def create_video_from_images(images, fps=30):
    """Create a video from a sequence of images."""
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    
    for image in images:
        out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    out.release() 