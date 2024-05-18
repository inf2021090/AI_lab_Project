import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import csv
import matplotlib.pyplot as plt
from dataset import SatelliteImageSegmentation 
from model import UNet 
from utils import create_dir

# Dataset
dataset_path = '../data/dubai_dataset'

# Hyperparameters
batch_size = 8
num_epochs = 2
learning_rate = 1e-4
patch_size = 256

# Folders
model_checkpoint_path = "../saved_models"
metrics_files_path = "../docs/logs/metrics"

# Mk directory if not exist
create_dir(model_checkpoint_path)
create_dir(metrics_files_path)

# Instantiate the dataset
dataset = SatelliteImageSegmentation(dataset_path, patch_size)
image_dataset, mask_dataset, labels = dataset.load_dataset()

# Create DataLoader
train_dataset = TensorDataset(torch.tensor(image_dataset, dtype=torch.float32).permute(0, 3, 1, 2),
                              torch.tensor(labels, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

total_classes = len(np.unique(labels)) # 6 classes for segmentation

# Init model, loss function, optimizer
model = UNet(total_classes)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# CSV file path for logging loss
csv_file_path = os.path.join(metrics_files_path, 'training_loss_steps.csv')
file_exists = os.path.isfile(csv_file_path)

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
        
        # Save the loss in CSV file
        with open(csv_file_path, mode='w', newline='') as csv_file:
            fieldnames = ['epoch', 'step', 'loss']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            # Write the header only if the file is new
            if not file_exists:
                writer.writeheader()
                file_exists = True
            
            # Write the metrics
            writer.writerow({'epoch': epoch + 1, 'step': i + 1, 'loss': loss.item()})
        
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Log average loss per epoch
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    
    # Save the model checkpoint after each epoch
    model_save_path = os.path.join(model_checkpoint_path, f'unet_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), model_save_path)

# Save the final trained model
final_model_save_path = os.path.join(model_checkpoint_path, 'unet_final.pth')
torch.save(model.state_dict(), final_model_save_path)

print('Training complete')




