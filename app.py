#!/usr/bin/env python
# coding: utf-8

# In[6]:



# import the necessary packages
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import os
from models.colorization import ColorizationModel, HistoricalColorizationModel, PortraitColorizationModel
from utils.image_processing import (
    preprocess_image, enhance_image, detect_faces,
    apply_style_transfer, batch_process_images, create_video_from_images
)
from utils.image_restoration import (
    denoise_image, sharpen_image, upscale_image,
    enhance_details, restore_old_photo, auto_enhance,
    apply_restoration_pipeline
)
from utils.ui_components import (
    create_image_comparison,
    create_dark_mode_toggle,
    create_feedback_section,
    create_version_history
)
from config import IMAGE_SETTINGS, COLOR_RANGES, BATCH_SETTINGS
from datetime import datetime
import shutil
import urllib.request

# Set page config
st.set_page_config(
    page_title="Black and White Image Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for consistent styling
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    /* Info message styling */
    .stAlert {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 4px;
        padding: 10px;
        margin: 10px 0;
    }
    .stAlert p {
        color: #1976d2;
        margin: 0;
    }
    /* Warning message styling */
    .stAlert[data-testid="stAlert"] {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
    }
    .stAlert[data-testid="stAlert"] p {
        color: #f57c00;
    }
    /* Error message styling */
    .stAlert[data-testid="stAlert"] {
        background-color: #ffebee;
        border: 1px solid #f44336;
    }
    .stAlert[data-testid="stAlert"] p {
        color: #d32f2f;
    }
    /* Success message styling */
    .stAlert[data-testid="stAlert"] {
        background-color: #e8f5e9;
        border: 1px solid #4caf50;
    }
    .stAlert[data-testid="stAlert"] p {
        color: #388e3c;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = ColorizationModel()
if 'version_history' not in st.session_state:
    st.session_state.version_history = []
if 'current_version' not in st.session_state:
    st.session_state.current_version = 0
if 'restoration_options' not in st.session_state:
    st.session_state.restoration_options = {
        'denoise': False,
        'sharpen': False,
        'upscale': False,
        'enhance_details': False,
        'auto_enhance': False
    }

# Apply dark mode
create_dark_mode_toggle()

# Sidebar
st.sidebar.title("Settings")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Colorization Model",
    ["Default", "Historical", "Portrait"]
)

if model_type == "Historical":
    st.session_state.current_model = HistoricalColorizationModel()
elif model_type == "Portrait":
    st.session_state.current_model = PortraitColorizationModel()
else:
    st.session_state.current_model = ColorizationModel()

# Image restoration options
st.sidebar.subheader("Image Restoration")
st.session_state.restoration_options['denoise'] = st.sidebar.checkbox("Denoise", value=False)
if st.session_state.restoration_options['denoise']:
    st.session_state.restoration_options['denoise_strength'] = st.sidebar.slider(
        "Denoise Strength",
        min_value=1,
        max_value=20,
        value=10
    )

st.session_state.restoration_options['sharpen'] = st.sidebar.checkbox("Sharpen", value=False)
if st.session_state.restoration_options['sharpen']:
    st.session_state.restoration_options['sharpen_amount'] = st.sidebar.slider(
        "Sharpen Amount",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1
    )

st.session_state.restoration_options['upscale'] = st.sidebar.checkbox("Upscale", value=False)
if st.session_state.restoration_options['upscale']:
    st.session_state.restoration_options['scale_factor'] = st.sidebar.slider(
        "Scale Factor",
        min_value=1,
        max_value=4,
        value=2
    )

st.session_state.restoration_options['enhance_details'] = st.sidebar.checkbox("Enhance Details", value=False)
if st.session_state.restoration_options['enhance_details']:
    st.session_state.restoration_options['detail_amount'] = st.sidebar.slider(
        "Detail Amount",
        min_value=1.0,
        max_value=2.0,
        value=1.2,
        step=0.1
    )

st.session_state.restoration_options['auto_enhance'] = st.sidebar.checkbox("Auto Enhance", value=False)

# Color adjustment
st.sidebar.subheader("Color Adjustment")
color_intensity = st.sidebar.slider(
    "Color Intensity",
    min_value=0.0,
    max_value=2.0,
    value=1.0,
    step=0.1
)

# Image enhancement
st.sidebar.subheader("Image Enhancement")
brightness = st.sidebar.slider(
    "Brightness",
    min_value=COLOR_RANGES['brightness'][0],
    max_value=COLOR_RANGES['brightness'][1],
    value=0,
    step=1
)

contrast = st.sidebar.slider(
    "Contrast",
    min_value=COLOR_RANGES['contrast'][0],
    max_value=COLOR_RANGES['contrast'][1],
    value=1.0,
    step=0.1
)

saturation = st.sidebar.slider(
    "Saturation",
    min_value=COLOR_RANGES['saturation'][0],
    max_value=COLOR_RANGES['saturation'][1],
    value=1.0,
    step=0.1
)

# Style transfer
st.sidebar.subheader("Style Transfer")
use_style_transfer = st.sidebar.checkbox("Apply Style Transfer")
style_image = None
if use_style_transfer:
    style_file = st.sidebar.file_uploader(
        "Upload Style Image",
        type=IMAGE_SETTINGS['supported_formats']
    )
    if style_file:
        style_image = Image.open(style_file)
        style_image = np.array(style_image)

# Feedback section
create_feedback_section()

# Main content
st.title("Black and White Image Colorization")
st.write("Transform your black and white images into vibrant color photos!")

# File uploader with drag and drop support
st.markdown("""
    <style>
        .uploadedFile {
            background-color: #f0f2f6;
            border: 2px dashed #4CAF50;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop an image here or click to upload",
    type=IMAGE_SETTINGS['supported_formats']
)

def download_model():
    """Download the model file if it doesn't exist."""
    model_path = "models/colorization_release_v2.caffemodel"
    if not os.path.exists("models"):
        os.makedirs("models")
    
    if not os.path.exists(model_path):
        st.info("Downloading model file... This may take a few minutes.")
        model_url = st.secrets.get("MODEL_URL", "https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            st.stop()

# Download model at startup
download_model()

if uploaded_file:
    try:
        # Read image
        image = Image.open(uploaded_file)
        img = np.array(image)
        
        # Ensure image is in RGB format
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Preprocess
        img = preprocess_image(img)
        
        # Apply restoration if options are selected
        if any(st.session_state.restoration_options.values()):
            img = apply_restoration_pipeline(img, st.session_state.restoration_options)
        
        # Enhance
        img = enhance_image(img, brightness, contrast, saturation)
        
        # Colorize
        colorized = st.session_state.current_model.colorize(img, color_intensity)
        
        # Ensure colorized image is in RGB format
        if len(colorized.shape) == 2:
            colorized = cv2.cvtColor(colorized, cv2.COLOR_GRAY2RGB)
        
        # Ensure both images have the same size
        if img.shape != colorized.shape:
            colorized = cv2.resize(colorized, (img.shape[1], img.shape[0]))
        
        # Apply style transfer if selected
        if use_style_transfer and style_image is not None:
            colorized = apply_style_transfer(colorized, style_image)
        
        # Display results
        st.subheader("Results")
        
        # Create image comparison
        create_image_comparison(img, colorized)
        
        # Download button
        buffered = BytesIO()
        Image.fromarray(colorized).save(buffered, format="PNG")
        st.download_button(
            label="Download Colorized Image",
            data=buffered.getvalue(),
            file_name=f"colorized_{uploaded_file.name}",
            mime="image/png",
            key="download_button"
        )
        
        # Show settings used
        st.subheader("Settings Used")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Color Intensity:", color_intensity)
            st.write("Brightness:", brightness)
        with col2:
            st.write("Contrast:", contrast)
            st.write("Saturation:", saturation)
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Please upload an image to begin colorization.")


# In[ ]:




