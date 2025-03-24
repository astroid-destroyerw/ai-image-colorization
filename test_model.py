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
from sklearn.metrics import mean_squared_error, structural_similarity_index
import pandas as pd

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
        
        return grayscale, torch.cat([a, b], dim=0), img_name

def evaluate_model(model, test_loader, save_dir):
    model.model.eval()
    metrics = []
    
    with torch.no_grad():
        for grayscale, color, img_names in tqdm(test_loader, desc='Testing'):
            grayscale = grayscale.to(model.device)
            
            # Generate colorization
            output = model.model(grayscale)
            
            # Convert to numpy
            grayscale = grayscale.cpu().numpy()
            output = output.cpu().numpy()
            color = color.cpu().numpy()
            
            # Calculate metrics for each image
            for i in range(len(grayscale)):
                # Convert to RGB for visualization
                lab = np.zeros((grayscale[i, 0].shape[0], grayscale[i, 0].shape[1], 3))
                lab[:, :, 0] = grayscale[i, 0]
                lab[:, :, 1:] = output[i].transpose(1, 2, 0)
                rgb_output = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                
                lab = np.zeros((grayscale[i, 0].shape[0], grayscale[i, 0].shape[1], 3))
                lab[:, :, 0] = grayscale[i, 0]
                lab[:, :, 1:] = color[i].transpose(1, 2, 0)
                rgb_gt = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                
                # Calculate metrics
                mse = mean_squared_error(rgb_gt, rgb_output)
                ssim = structural_similarity_index(rgb_gt, rgb_output, multichannel=True)
                
                metrics.append({
                    'image': img_names[i],
                    'mse': mse,
                    'ssim': ssim
                })
                
                # Save visualization
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(grayscale[i, 0], cmap='gray')
                axes[0].set_title('Grayscale')
                axes[0].axis('off')
                
                axes[1].imshow(rgb_output)
                axes[1].set_title('Colorized')
                axes[1].axis('off')
                
                axes[2].imshow(rgb_gt)
                axes[2].set_title('Ground Truth')
                axes[2].axis('off')
                
                plt.savefig(os.path.join(save_dir, f'result_{os.path.basename(img_names[i])}'))
                plt.close()
    
    # Calculate average metrics
    df = pd.DataFrame(metrics)
    avg_metrics = df.mean()
    
    # Save metrics to CSV
    df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
    
    return avg_metrics

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
    
    # Create test dataset and dataloader
    test_dataset = ColorizationDataset('path/to/test/data', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize model and load weights
    model = CustomColorizationModel()
    model.load_model('model_checkpoints/model_epoch_50.pth')
    
    # Create results directory
    save_dir = 'test_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # Evaluate model
    avg_metrics = evaluate_model(model, test_loader, save_dir)
    
    # Print results
    print("\nTest Results:")
    print(f"Average MSE: {avg_metrics['mse']:.4f}")
    print(f"Average SSIM: {avg_metrics['ssim']:.4f}")

if __name__ == '__main__':
    main() 