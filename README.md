# AI Image Colorization

A Streamlit web application that colorizes black and white images using deep learning.

## Features

- Upload and colorize black and white images
- Interactive comparison between original and colorized images
- Image enhancement options
- Support for multiple image formats
- User-friendly interface

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/astroid-destroyerw/ai-image-colorization.git
cd ai-image-colorization
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the model file:
- Download the `colorization_release_v2.caffemodel` from [here](https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1)
- Place it in the `models/` directory

5. Run the application:
```bash
streamlit run app.py
```

## Deployment on Streamlit Cloud

1. Fork this repository
2. Sign up for [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and connect it to your forked repository
4. Add the following secrets in Streamlit Cloud settings:
   - Navigate to your app settings
   - Add the model URL as a secret:
     - Name: `MODEL_URL`
     - Value: `https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1`

## Model Files
The following files are required but not included in the repository due to size:
- `models/colorization_release_v2.caffemodel` (122.97 MB)

These files will be downloaded automatically when deploying to Streamlit Cloud.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

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