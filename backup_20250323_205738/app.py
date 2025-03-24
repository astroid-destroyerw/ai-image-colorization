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
    create_image_comparison, create_progress_ring,
    create_color_palette, create_dark_mode_toggle,
    create_feedback_section, create_version_history
)
from config import IMAGE_SETTINGS, COLOR_RANGES, BATCH_SETTINGS

# Set page config
st.set_page_config(
    page_title="Advanced Image Colorization",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = ColorizationModel()
if 'version_history' not in st.session_state:
    st.session_state.version_history = []
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
st.title("Advanced Image Colorization")
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

uploaded_files = st.file_uploader(
    "Drag and drop images here or click to upload",
    type=IMAGE_SETTINGS['supported_formats'],
    accept_multiple_files=True
)

if uploaded_files:
    # Process images
    for uploaded_file in uploaded_files:
        # Read image
        image = Image.open(uploaded_file)
        img = np.array(image)
        
        # Preprocess
        img = preprocess_image(img)
        
        # Apply restoration if options are selected
        if any(st.session_state.restoration_options.values()):
            img = apply_restoration_pipeline(img, st.session_state.restoration_options)
        
        # Enhance
        img = enhance_image(img, brightness, contrast, saturation)
        
        # Show progress ring
        progress_ring = create_progress_ring(0)
        
        # Colorize
        colorized = st.session_state.current_model.colorize(img, color_intensity)
        
        # Apply style transfer if selected
        if use_style_transfer and style_image is not None:
            colorized = apply_style_transfer(colorized, style_image)
        
        # Update progress
        progress_ring = create_progress_ring(100)
        
        # Store processed image
        st.session_state.processed_images.append({
            'original': img,
            'colorized': colorized,
            'filename': uploaded_file.name,
            'settings': {
                'color_intensity': color_intensity,
                'brightness': brightness,
                'contrast': contrast,
                'saturation': saturation,
                'restoration_options': st.session_state.restoration_options
            }
        })
        
        # Add to version history
        st.session_state.version_history.append({
            'image': colorized,
            'settings': st.session_state.processed_images[-1]['settings']
        })
    
    # Display results
    st.subheader("Results")
    
    # Create columns for each image
    cols = st.columns(len(st.session_state.processed_images))
    
    for idx, (col, processed) in enumerate(zip(cols, st.session_state.processed_images)):
        with col:
            st.write(f"Image {idx + 1}: {processed['filename']}")
            
            # Create image comparison
            create_image_comparison(processed['original'], processed['colorized'])
            
            # Show color palette
            create_color_palette(processed['colorized'])
            
            # Download button
            buffered = BytesIO()
            Image.fromarray(processed['colorized']).save(buffered, format="PNG")
            st.download_button(
                label=f"Download {processed['filename']}",
                data=buffered.getvalue(),
                file_name=f"colorized_{processed['filename']}",
                mime="image/png"
            )
    
    # Batch download option
    if len(st.session_state.processed_images) > 1:
        st.subheader("Batch Download")
        if st.button("Download All Images"):
            for processed in st.session_state.processed_images:
                buffered = BytesIO()
                Image.fromarray(processed['colorized']).save(buffered, format="PNG")
                st.download_button(
                    label=f"Download {processed['filename']}",
                    data=buffered.getvalue(),
                    file_name=f"colorized_{processed['filename']}",
                    mime="image/png"
                )
    
    # Version history
    st.subheader("Version History")
    restored_version = create_version_history(st.session_state.version_history)
    if restored_version:
        # Update the current image with the restored version
        st.session_state.processed_images[-1]['colorized'] = restored_version['image']
else:
    st.info("Please upload one or more images to begin colorization.")


# In[ ]:




