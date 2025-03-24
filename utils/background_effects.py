import streamlit as st
import numpy as np
from PIL import Image
import cv2

def create_dynamic_background(image_path=None, effect_type="gradient"):
    """
    Create a dynamic background effect for the application.
    
    Args:
        image_path (str, optional): Path to the background image
        effect_type (str): Type of background effect ("gradient", "pattern", "image")
    
    Returns:
        str: HTML/CSS for the background effect
    """
    if effect_type == "gradient":
        return """
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a237e, #0d47a1, #1565c0);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        </style>
        """
    
    elif effect_type == "pattern":
        return """
        <style>
        .stApp {
            background-color: #ffffff;
            background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%239C92AC' fill-opacity='0.1'%3E%3Cpath d='M36 34v-4l-2-2V24h-2v4l-2 2v4h6zm0-30V0h-2v4h-2v2h2v4h2V6h2V4h-2zM6 34v-4l-2-2V24H2v4l-2 2v4h6zM6 4V0H4v4H2v2h2v4h2V6h2V4H6zM34 0h-2v4h-2v2h2v4h2V6h2V4h-2V0zM4 0H2v4H0v2h2v4h2V6h2V4H4V0z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        }
        </style>
        """
    
    elif effect_type == "image" and image_path:
        # Process the image to create a blurred, low-opacity background
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1920, 1080))
        img = cv2.GaussianBlur(img, (21, 21), 0)
        
        # Convert to base64 for embedding
        import base64
        from io import BytesIO
        buffered = BytesIO()
        Image.fromarray(img).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.9);
            z-index: -1;
        }}
        </style>
        """
    
    return ""

def create_parallax_effect(element_id, speed=0.5):
    """
    Create a parallax scrolling effect for an element.
    
    Args:
        element_id (str): ID of the element to apply parallax to
        speed (float): Speed of the parallax effect (0.0 to 1.0)
    
    Returns:
        str: JavaScript code for the parallax effect
    """
    return f"""
    <script>
    window.addEventListener('scroll', function() {{
        const element = document.getElementById('{element_id}');
        const scrolled = window.pageYOffset;
        const rate = scrolled * {speed};
        element.style.transform = `translateY(${{rate}}px)`;
    }});
    </script>
    """

def create_image_carousel(images, interval=5000):
    """
    Create an image carousel with automatic rotation.
    
    Args:
        images (list): List of image paths
        interval (int): Rotation interval in milliseconds
    
    Returns:
        str: HTML/CSS/JavaScript for the carousel
    """
    carousel_html = """
    <div class="carousel-container">
        <div class="carousel-slides">
    """
    
    for i, img_path in enumerate(images):
        carousel_html += f"""
            <div class="carousel-slide" style="display: {'block' if i == 0 else 'none'}">
                <img src="{img_path}" alt="Carousel image {i+1}">
            </div>
        """
    
    carousel_html += """
        </div>
        <button class="carousel-prev">❮</button>
        <button class="carousel-next">❯</button>
    </div>
    """
    
    carousel_css = """
    <style>
    .carousel-container {
        position: relative;
        width: 100%;
        height: 400px;
        overflow: hidden;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .carousel-slides {
        width: 100%;
        height: 100%;
    }
    
    .carousel-slide {
        position: absolute;
        width: 100%;
        height: 100%;
        opacity: 0;
        transition: opacity 0.5s ease-in-out;
    }
    
    .carousel-slide.active {
        opacity: 1;
    }
    
    .carousel-slide img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .carousel-prev,
    .carousel-next {
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        background: rgba(0, 0, 0, 0.5);
        color: white;
        border: none;
        padding: 1rem;
        cursor: pointer;
        font-size: 1.5rem;
        border-radius: 50%;
        transition: background 0.3s ease;
    }
    
    .carousel-prev:hover,
    .carousel-next:hover {
        background: rgba(0, 0, 0, 0.8);
    }
    
    .carousel-prev {
        left: 1rem;
    }
    
    .carousel-next {
        right: 1rem;
    }
    </style>
    """
    
    carousel_js = f"""
    <script>
    let currentSlide = 0;
    const slides = document.querySelectorAll('.carousel-slide');
    const prevBtn = document.querySelector('.carousel-prev');
    const nextBtn = document.querySelector('.carousel-next');
    
    function showSlide(n) {{
        slides.forEach(slide => slide.style.display = 'none');
        currentSlide = (n + slides.length) % slides.length;
        slides[currentSlide].style.display = 'block';
    }}
    
    function nextSlide() {{
        showSlide(currentSlide + 1);
    }}
    
    function prevSlide() {{
        showSlide(currentSlide - 1);
    }}
    
    prevBtn.addEventListener('click', prevSlide);
    nextBtn.addEventListener('click', nextSlide);
    
    setInterval(nextSlide, {interval});
    </script>
    """
    
    return carousel_html + carousel_css + carousel_js

def create_animated_graphics():
    """
    Create animated graphics for the background.
    
    Returns:
        str: HTML/CSS/JavaScript for animated graphics
    """
    return """
    <style>
    .animated-graphics {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .graphic {
        position: absolute;
        width: 50px;
        height: 50px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        animation: float 20s infinite linear;
    }
    
    @keyframes float {
        0% {
            transform: translateY(100vh) scale(0);
            opacity: 0;
        }
        50% {
            opacity: 0.5;
        }
        100% {
            transform: translateY(-100px) scale(1);
            opacity: 0;
        }
    }
    </style>
    
    <div class="animated-graphics" id="graphics-container"></div>
    
    <script>
    function createGraphics() {
        const container = document.getElementById('graphics-container');
        const numGraphics = 20;
        
        for (let i = 0; i < numGraphics; i++) {
            const graphic = document.createElement('div');
            graphic.className = 'graphic';
            graphic.style.left = `${Math.random() * 100}%`;
            graphic.style.animationDelay = `${Math.random() * 20}s`;
            container.appendChild(graphic);
        }
    }
    
    createGraphics();
    </script>
    """ 