# AI Image Colorization

A Streamlit application that uses deep learning to colorize black and white images. The application provides an intuitive interface for uploading images, adjusting colorization parameters, and downloading the results.

## Features

- Upload and process black and white images
- Multiple colorization models (Standard, Historical, Portrait)
- Interactive image comparison with slider
- Image enhancement options (brightness, contrast, saturation)
- Image restoration features (denoising, sharpening, upscaling)
- Style transfer capability
- Dark mode support
- Version history tracking
- Download processed images

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-image-colorization.git
cd ai-image-colorization
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload a black and white image using the file uploader

4. Adjust the colorization parameters in the sidebar:
   - Select a colorization model
   - Adjust color intensity
   - Modify image enhancement settings
   - Enable/disable restoration options
   - Apply style transfer (optional)

5. View the results with the interactive comparison slider

6. Download the colorized image using the download button

## Project Structure

```
ai-image-colorization/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── models/              # Colorization models
│   └── colorization.py
├── utils/               # Utility functions
│   ├── image_processing.py
│   ├── image_restoration.py
│   └── ui_components.py
├── Input_images/        # Directory for uploaded images
└── Result_images/       # Directory for processed images
```

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- OpenCV for image processing
- PyTorch for deep learning models
- Streamlit for the web interface 