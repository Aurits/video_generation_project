import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.keyframe_autoencoder import KeyframeAutoencoder
from models.interpolation_module import InterpolationModule
from data.dataset import VideoFrameDataset
from torchvision.transforms import Compose, Resize, ToTensor

# Hyperparameters
batch_size = 8
num_epochs = 50
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    # Prepare dataset and dataloader
    transform = Compose([Resize((64, 64)), ToTensor()])
    dataset = VideoFrameDataset(video_dir='data/videos', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    autoencoder = KeyframeAutoencoder().to(device)
    interp_module = InterpolationModule().to(device)
    
    # Optimizers
    optimizer = optim.Adam(list(autoencoder.parameters()) + list(interp_module.parameters()), lr=learning_rate)
    
    # Loss function: a simple L1 loss for reconstruction and interpolation
    criterion = nn.L1Loss()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for keyframe1, keyframe2 in dataloader:
            keyframe1, keyframe2 = keyframe1.to(device), keyframe2.to(device)
            
            # Forward pass: reconstruct keyframes
            reconstructed = autoencoder(keyframe1)
            loss_recon = criterion(reconstructed, keyframe1)
            
            # Interpolate between keyframes (using a midpoint t=0.5)
            interpolated = interp_module(keyframe1, keyframe2, t=0.5)
            # Use keyframe2 as a rough target for the interpolation module (this can be improved with better targets)
            loss_interp = criterion(interpolated, keyframe2)
            
            loss = loss_recon + loss_interp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
        
        # Optionally save models every few epochs
        if (epoch + 1) % 10 == 0:
            torch.save(autoencoder.state_dict(), f'checkpoints/autoencoder_epoch{epoch+1}.pth')
            torch.save(interp_module.state_dict(), f'checkpoints/interp_module_epoch{epoch+1}.pth')

if __name__ == '__main__':
    train()
