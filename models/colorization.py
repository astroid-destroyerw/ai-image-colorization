import cv2
import numpy as np
from config import MODEL_PATHS

class ColorizationModel:
    def __init__(self, model_name='default'):
        self.model_name = model_name
        self.model_paths = MODEL_PATHS[model_name]
        self.net = None
        self.pts = None
        self._load_model()

    def _load_model(self):
        """Load the colorization model."""
        # Load the model
        self.net = cv2.dnn.readNetFromCaffe(
            self.model_paths['prototxt'],
            self.model_paths['model']
        )
        
        # Load the cluster centers
        self.pts = np.load(self.model_paths['points'])
        
        # Add the cluster centers as 1x1 convolutions to the model
        class8 = self.net.getLayerId("class8_ab")
        conv8 = self.net.getLayerId("conv8_313_rh")
        pts = self.pts.transpose().reshape(2, 313, 1, 1)
        self.net.getLayer(class8).blobs = [pts.astype("float32")]
        self.net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    def colorize(self, image, color_intensity=1.0):
        """Colorize the input image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Scale the pixel intensities to the range [0, 1]
        scaled = gray.astype("float32") / 255.0
        
        # Convert the image from the BGR to Lab color space
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        
        # Resize the Lab image to 224x224
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        
        # Pass the L channel through the network
        self.net.setInput(cv2.dnn.blobFromImage(L))
        ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
        
        # Resize the predicted 'ab' volume
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        
        # Grab the 'L' channel from the original input image
        L = cv2.split(lab)[0]
        
        # Concatenate the original 'L' channel with the predicted 'ab' channels
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        
        # Apply color intensity
        colorized[:, :, 1:] *= color_intensity
        
        # Convert the output image from the Lab color space to RGB
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        
        # Convert to uint8
        colorized = (255 * colorized).astype("uint8")
        
        return colorized

class HistoricalColorizationModel(ColorizationModel):
    def __init__(self):
        super().__init__('default')
        # Add historical photo specific processing
        self.vintage_effect = True

    def colorize(self, image, color_intensity=1.0, vintage_strength=0.3):
        """Colorize with historical photo effects."""
        # First do normal colorization
        colorized = super().colorize(image, color_intensity)
        
        if self.vintage_effect:
            # Apply sepia effect
            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            
            # Blend original colorization with sepia effect
            sepia = cv2.transform(colorized, sepia_matrix)
            colorized = cv2.addWeighted(colorized, 1-vintage_strength, sepia, vintage_strength, 0)
        
        return colorized

class PortraitColorizationModel(ColorizationModel):
    def __init__(self):
        super().__init__('default')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def colorize(self, image, color_intensity=1.0):
        """Colorize with special handling for portraits."""
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Do normal colorization
        colorized = super().colorize(image, color_intensity)
        
        if len(faces) > 0:
            # Enhance face regions
            for (x, y, w, h) in faces:
                face_region = colorized[y:y+h, x:x+w]
                # Apply additional enhancement to face region
                face_region = cv2.convertScaleAbs(face_region, alpha=1.1, beta=10)
                colorized[y:y+h, x:x+w] = face_region
        
        return colorized 