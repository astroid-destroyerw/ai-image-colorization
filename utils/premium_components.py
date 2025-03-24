import streamlit as st
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import base64

def create_premium_navigation():
    """Create a premium navigation bar with sticky behavior."""
    return """
    <nav class="premium-nav" id="premium-nav">
        <div class="nav-brand">
            <h1 style="margin: 0; font-size: 1.5rem; color: var(--primary-color);">AI Colorization</h1>
        </div>
        <div class="nav-links">
            <a href="#about" class="nav-link">About</a>
            <a href="#projects" class="nav-link">Projects</a>
            <a href="#apply" class="nav-link">Apply</a>
            <a href="#contact" class="nav-link">Contact</a>
        </div>
    </nav>
    <script>
    window.addEventListener('scroll', function() {
        const nav = document.getElementById('premium-nav');
        if (window.scrollY > 50) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    });
    </script>
    """

def create_premium_hero():
    """Create a premium hero section with animated content."""
    return """
    <section class="premium-hero">
        <div class="hero-content">
            <h1 class="hero-title">AI Image Colorization</h1>
            <p class="hero-subtitle">Transform your black and white images into vibrant color photos</p>
            <a href="#start" class="cta-button">Get Started</a>
        </div>
    </section>
    """

def create_premium_project_card(image, title, description, tags=None):
    """Create a premium project card with hover effects."""
    if isinstance(image, np.ndarray):
        # Convert numpy array to base64
        buffered = BytesIO()
        Image.fromarray(image).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_src = f"data:image/png;base64,{img_str}"
    else:
        image_src = image

    tags_html = ""
    if tags:
        tags_html = '<div class="project-tags">' + \
                   ''.join([f'<span class="tag">{tag}</span>' for tag in tags]) + \
                   '</div>'

    return f"""
    <div class="project-card">
        <img src="{image_src}" alt="{title}" class="project-image">
        <div class="project-content">
            <h3>{title}</h3>
            <p>{description}</p>
            {tags_html}
        </div>
    </div>
    """

def create_premium_search():
    """Create a premium search bar with dynamic results."""
    return """
    <div class="search-container">
        <input type="text" class="search-input" placeholder="Search projects, research areas, or student work...">
    </div>
    """

def create_premium_lightbox(image, title=None):
    """Create a premium lightbox for image viewing."""
    if isinstance(image, np.ndarray):
        # Convert numpy array to base64
        buffered = BytesIO()
        Image.fromarray(image).save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_src = f"data:image/png;base64,{img_str}"
    else:
        image_src = image

    title_html = f'<h3>{title}</h3>' if title else ''

    return f"""
    <div class="lightbox" id="lightbox">
        <div class="lightbox-content">
            <span class="lightbox-close">&times;</span>
            {title_html}
            <img src="{image_src}" alt="{title}" class="lightbox-image">
        </div>
    </div>
    <script>
    document.querySelector('.lightbox-close').addEventListener('click', function() {{
        document.getElementById('lightbox').classList.remove('active');
    }});
    </script>
    """

def create_premium_modal(title, content):
    """Create a premium modal window."""
    return f"""
    <div class="modal" id="modal">
        <div class="modal-content">
            <span class="modal-close">&times;</span>
            <h2>{title}</h2>
            {content}
        </div>
    </div>
    <script>
    document.querySelector('.modal-close').addEventListener('click', function() {{
        document.getElementById('modal').classList.remove('active');
    }});
    </script>
    """

def create_premium_footer():
    """Create a premium footer with social links."""
    return """
    <footer class="premium-footer">
        <div class="footer-content">
            <div class="footer-section">
                <h3>About Us</h3>
                <p>Advanced AI Image Colorization Platform</p>
            </div>
            <div class="footer-section">
                <h3>Quick Links</h3>
                <ul>
                    <li><a href="#about">About</a></li>
                    <li><a href="#projects">Projects</a></li>
                    <li><a href="#apply">Apply</a></li>
                    <li><a href="#contact">Contact</a></li>
                </ul>
            </div>
            <div class="footer-section">
                <h3>Connect With Us</h3>
                <div class="social-links">
                    <a href="#" class="social-link">Twitter</a>
                    <a href="#" class="social-link">LinkedIn</a>
                    <a href="#" class="social-link">GitHub</a>
                </div>
            </div>
        </div>
    </footer>
    """

def create_premium_loading():
    """Create a premium loading spinner."""
    return """
    <div class="loading-spinner"></div>
    """

def create_premium_scroll_animation(element_id):
    """Add scroll animation to an element."""
    return f"""
    <script>
    const observer = new IntersectionObserver((entries) => {{
        entries.forEach(entry => {{
            if (entry.isIntersecting) {{
                entry.target.classList.add('visible');
            }}
        }});
    }});
    
    document.getElementById('{element_id}').classList.add('scroll-animate');
    observer.observe(document.getElementById('{element_id}'));
    </script>
    """ 