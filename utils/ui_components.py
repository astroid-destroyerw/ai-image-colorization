import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import base64

def create_image_comparison(original, colorized, width=400):
    """Create a side-by-side comparison with a draggable slider."""
    try:
        # Convert images to RGB if they aren't already
        if len(original.shape) == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        if len(colorized.shape) == 2:
            colorized = cv2.cvtColor(colorized, cv2.COLOR_GRAY2RGB)
        
        # Ensure both images have exactly 3 channels (RGB)
        if original.shape[-1] == 4:
            original = original[:, :, :3]
        if colorized.shape[-1] == 4:
            colorized = colorized[:, :, :3]
        
        # Ensure both images have the same size
        if original.shape != colorized.shape:
            # Calculate the new dimensions while maintaining aspect ratio
            height = int(width * original.shape[0] / original.shape[1])
            original = cv2.resize(original, (width, height))
            colorized = cv2.resize(colorized, (width, height))
        
        # Convert images to PIL for display
        original_pil = Image.fromarray(original)
        colorized_pil = Image.fromarray(colorized)
        
        # Create a container for the comparison
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(original_pil, caption="Original", use_container_width=True)
        
        with col2:
            st.image(colorized_pil, caption="Colorized", use_container_width=True)
        
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
        
        # Convert images to float32 and normalize
        original_float = original.astype(np.float32) / 255.0
        colorized_float = colorized.astype(np.float32) / 255.0
        
        # Create a mask for blending
        mask = np.zeros_like(original_float)
        mask[:, :int(width * blend_ratio)] = 1
        
        # Blend the images using the mask
        blended = original_float * (1 - mask) + colorized_float * mask
        
        # Convert back to uint8
        blended = (blended * 255).astype(np.uint8)
        
        # Convert blended image to PIL for display
        blended_pil = Image.fromarray(blended)
        st.image(blended_pil, caption="Interactive Comparison", use_container_width=True)
    except Exception as e:
        st.error(f"Error creating image comparison: {str(e)}")
        # Show a simple side-by-side comparison as fallback
        try:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(original, caption="Original", use_container_width=True)
            with col2:
                st.image(colorized, caption="Colorized", use_container_width=True)
        except:
            pass

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

def create_version_history(version_history):
    """Create version history section with slider and restore functionality."""
    if not version_history:
        return None
        
    # Create version selector
    version = st.select_slider(
        "Select Version",
        options=range(len(version_history)),
        value=len(version_history) - 1,
        format_func=lambda x: f"Version {x+1}"
    )
    
    # Display selected version
    selected_version = version_history[version]
    
    # Show settings used
    st.write("Settings used:")
    settings = selected_version['settings']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("Color Intensity:", settings['color_intensity'])
        st.write("Brightness:", settings['brightness'])
    with col2:
        st.write("Contrast:", settings['contrast'])
        st.write("Saturation:", settings['saturation'])
    with col3:
        st.write("Restoration:", settings['restoration_options'])
    
    # Restore button
    if st.button("Restore This Version", key="restore_version"):
        return selected_version
    
    return None 