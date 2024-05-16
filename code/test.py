import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
import numpy as np
from dataset import SatelliteImageSegmentation  # Ensure this path is correct for your setup
from model import UNet  # Ensure this path is correct for your setup

# Define the dataset path and hyperparameters
dataset_path = '../data/dubai_dataset'
batch_size = 8
num_epochs = 2
learning_rate = 1e-4
patch_size = 256

# Instantiate the dataset
dataset = SatelliteImageSegmentation(dataset_path, patch_size)
image_dataset, mask_dataset, labels = dataset.load_dataset()

# Create DataLoader
train_dataset = TensorDataset(torch.tensor(image_dataset, dtype=torch.float32).permute(0, 3, 1, 2),
                              torch.tensor(labels, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model, loss function, and optimizer
model = UNet(n_classes=6)  # 6 classes for segmentation
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Log average loss per epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

print('Training complete')

# Save the model
model_save_path = 'unet_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')






