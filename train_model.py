import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from models.custom_model import CustomColorizationModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        # Convert to grayscale
        grayscale = image.convert('L')
        
        # Convert to LAB color space
        lab_image = image.convert('LAB')
        l, a, b = lab_image.split()
        
        if self.transform:
            grayscale = self.transform(grayscale)
            a = self.transform(a)
            b = self.transform(b)
        
        return grayscale, torch.cat([a, b], dim=0)

def train_model(model, train_loader, num_epochs, save_dir):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (grayscale, color) in enumerate(progress_bar):
            grayscale = grayscale.to(model.device)
            color = color.to(model.device)
            
            model.optimizer.zero_grad()
            output = model.model(grayscale)
            loss = model.criterion(output, color)
            loss.backward()
            model.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
        
        # Save model checkpoint
        model.save_model(os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
        
        # Save sample results
        if (epoch + 1) % 5 == 0:
            save_sample_results(model, train_loader, epoch, save_dir)

def save_sample_results(model, train_loader, epoch, save_dir):
    model.model.eval()
    with torch.no_grad():
        # Get a batch of images
        grayscale, color = next(iter(train_loader))
        grayscale = grayscale.to(model.device)
        
        # Generate colorization
        output = model.model(grayscale)
        
        # Convert to RGB
        grayscale = grayscale.cpu().numpy()
        output = output.cpu().numpy()
        color = color.cpu().numpy()
        
        # Plot results
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i in range(3):
            # Grayscale
            axes[0, i].imshow(grayscale[i, 0], cmap='gray')
            axes[0, i].set_title('Grayscale')
            axes[0, i].axis('off')
            
            # Colorized
            lab = np.zeros((grayscale[i, 0].shape[0], grayscale[i, 0].shape[1], 3))
            lab[:, :, 0] = grayscale[i, 0]
            lab[:, :, 1:] = output[i].transpose(1, 2, 0)
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            axes[1, i].imshow(rgb)
            axes[1, i].set_title('Colorized')
            axes[1, i].axis('off')
            
            # Ground Truth
            lab = np.zeros((grayscale[i, 0].shape[0], grayscale[i, 0].shape[1], 3))
            lab[:, :, 0] = grayscale[i, 0]
            lab[:, :, 1:] = color[i].transpose(1, 2, 0)
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            axes[2, i].imshow(rgb)
            axes[2, i].set_title('Ground Truth')
            axes[2, i].axis('off')
        
        plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch+1}.png'))
        plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset and dataloader
    dataset = ColorizationDataset('path/to/training/data', transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Initialize model
    model = CustomColorizationModel()
    
    # Train model
    train_model(model, train_loader, num_epochs=50, save_dir='model_checkpoints')

if __name__ == '__main__':
    main() 