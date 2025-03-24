import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import base64

def create_image_comparison(original, colorized, width=400):
    """Create a side-by-side comparison with a draggable slider."""
    # Convert images to PIL
    original_pil = Image.fromarray(original)
    colorized_pil = Image.fromarray(colorized)
    
    # Resize images to match width
    height = int(width * original.shape[0] / original.shape[1])
    original_pil = original_pil.resize((width, height))
    colorized_pil = colorized_pil.resize((width, height))
    
    # Create a container for the comparison
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(original_pil, caption="Original", use_column_width=True)
    with col2:
        st.image(colorized_pil, caption="Colorized", use_column_width=True)
    
    # Add a slider for interactive comparison
    comparison_value = st.slider(
        "Slide to compare",
        min_value=0,
        max_value=100,
        value=50,
        key="comparison_slider"
    )
    
    # Create a blended image based on slider value
    blend_ratio = comparison_value / 100
    blended = cv2.addWeighted(original, 1 - blend_ratio, colorized, blend_ratio, 0)
    st.image(blended, caption="Interactive Comparison", use_column_width=True)

def create_progress_ring(progress):
    """Create a circular progress indicator."""
    # Create a canvas for the progress ring
    canvas = st.empty()
    
    # Update the progress ring
    canvas.markdown(f"""
        <div style="
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: conic-gradient(
                from 0deg,
                #4CAF50 {progress}%,
                #E0E0E0 {progress}%
            );
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <div style="
                width: 80px;
                height: 80px;
                background: white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            ">
                {progress}%
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    return canvas

def create_color_palette(image):
    """Extract and display dominant colors from the image."""
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Convert to float32
    pixels = np.float32(pixels)
    
    # Define criteria for k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    # Apply k-means clustering
    _, labels, palette = cv2.kmeans(pixels, 5, None, criteria, 10, flags)
    
    # Convert palette to uint8
    palette = np.uint8(palette)
    
    # Create a row of color swatches
    color_swatches = ""
    for color in palette:
        rgb = tuple(color)
        color_swatches += f"""
            <div style="
                display: inline-block;
                width: 40px;
                height: 40px;
                background-color: rgb{rgb};
                margin: 0 5px;
                border-radius: 5px;
                border: 1px solid #ccc;
            "></div>
        """
    
    st.markdown(f"""
        <div style="text-align: center; margin: 20px 0;">
            <h4>Dominant Colors</h4>
            {color_swatches}
        </div>
    """, unsafe_allow_html=True)

def create_dark_mode_toggle():
    """Create a dark mode toggle button."""
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    if st.sidebar.button("🌙 Dark Mode"):
        st.session_state.dark_mode = not st.session_state.dark_mode
    
    if st.session_state.dark_mode:
        st.markdown("""
            <style>
                .stApp {
                    background-color: #1E1E1E;
                    color: white;
                }
                .stButton>button {
                    background-color: #4CAF50;
                    color: white;
                }
                .stSlider>div>div>div {
                    background-color: #4CAF50;
                }
            </style>
        """, unsafe_allow_html=True)

def create_feedback_section():
    """Create a feedback section with thumbs up/down and comments."""
    st.sidebar.subheader("Feedback")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("👍"):
            st.success("Thank you for your positive feedback!")
    with col2:
        if st.button("👎"):
            st.warning("Thank you for your feedback. We'll work on improving!")
    
    feedback = st.sidebar.text_area("Additional comments:")
    if st.sidebar.button("Submit Feedback"):
        if feedback:
            st.success("Thank you for your detailed feedback!")
        else:
            st.info("Please provide some feedback to help us improve!")

def create_version_history(versions):
    """Display version history of processed images."""
    st.sidebar.subheader("Version History")
    
    for i, version in enumerate(versions):
        with st.sidebar.expander(f"Version {i+1}"):
            st.image(version['image'], caption=f"Version {i+1}", use_column_width=True)
            st.write(f"Settings: {version['settings']}")
            if st.button(f"Restore Version {i+1}", key=f"restore_{i}"):
                return version
    return None 