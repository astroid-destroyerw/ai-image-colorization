import torch
import torch.nn as nn
import torch.nn.functional as F

class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Decoder
        output = self.decoder(features)
        return output

class CustomColorizationModel:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = ColorizationNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (grayscale, color) in enumerate(train_loader):
                grayscale = grayscale.to(self.device)
                color = color.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(grayscale)
                loss = self.criterion(output, color)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
    
    def colorize(self, image, color_intensity=1.0):
        self.model.eval()
        with torch.no_grad():
            # Preprocess image
            if len(image.shape) == 2:
                image = image.reshape(1, 1, image.shape[0], image.shape[1])
            elif len(image.shape) == 3:
                image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            
            # Convert to tensor and normalize
            image = torch.FloatTensor(image).to(self.device)
            image = (image - 0.5) / 0.5
            
            # Get colorization
            output = self.model(image)
            
            # Denormalize and apply color intensity
            output = output * 0.5 + 0.5
            output = output * color_intensity
            
            # Convert back to numpy
            output = output.cpu().numpy()
            output = output[0].transpose(1, 2, 0)
            
            return output
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 