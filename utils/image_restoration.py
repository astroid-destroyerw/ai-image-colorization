import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf

def denoise_image(image, strength=10):
    """Apply non-local means denoising to the image."""
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

def sharpen_image(image, amount=1.5):
    """Sharpen the image using unsharp masking."""
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def upscale_image(image, scale_factor=2):
    """Upscale the image using super-resolution."""
    # Convert to float32
    img = image.astype(np.float32)
    
    # Apply bicubic interpolation
    height, width = img.shape[:2]
    upscaled = cv2.resize(img, (width * scale_factor, height * scale_factor),
                         interpolation=cv2.INTER_CUBIC)
    
    return upscaled.astype(np.uint8)

def enhance_details(image, amount=1.2):
    """Enhance image details using edge detection and blending."""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 100, 200)
    
    # Create a mask from edges
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)
    
    # Enhance the image
    enhanced = cv2.convertScaleAbs(image, alpha=amount, beta=0)
    
    # Blend enhanced image with original using edge mask
    result = cv2.addWeighted(image, 1, enhanced, amount-1, 0)
    
    return result

def restore_old_photo(image, denoise_strength=10, sharpen_amount=1.5):
    """Apply restoration techniques specifically for old photos."""
    # Denoise
    denoised = denoise_image(image, denoise_strength)
    
    # Sharpen
    sharpened = sharpen_image(denoised, sharpen_amount)
    
    # Enhance details
    enhanced = enhance_details(sharpened)
    
    return enhanced

def auto_enhance(image):
    """Apply automatic image enhancement."""
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Auto contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    # Auto brightness
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.1)
    
    # Auto color
    enhancer = ImageEnhance.Color(pil_image)
    pil_image = enhancer.enhance(1.1)
    
    # Auto sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    return np.array(pil_image)

def apply_restoration_pipeline(image, options):
    """Apply a series of restoration techniques based on options."""
    result = image.copy()
    
    if options.get('denoise', False):
        result = denoise_image(result, options.get('denoise_strength', 10))
    
    if options.get('sharpen', False):
        result = sharpen_image(result, options.get('sharpen_amount', 1.5))
    
    if options.get('upscale', False):
        result = upscale_image(result, options.get('scale_factor', 2))
    
    if options.get('enhance_details', False):
        result = enhance_details(result, options.get('detail_amount', 1.2))
    
    if options.get('auto_enhance', False):
        result = auto_enhance(result)
    
    return result 