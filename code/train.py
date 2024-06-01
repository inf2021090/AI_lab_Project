import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import csv
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score
from dataset import SatelliteImageSegmentation 
from model import UNet 
from utils import create_dir

# Dataset
dataset_path = '../data/dubai_dataset'

# Hyperparameters
batch_size = 8
num_epochs = 25 # modify accordingly
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

# Split dataset into training and validation sets
num_samples = len(image_dataset)
split_ratio = 0.8
split_index = int(num_samples * split_ratio)

train_images = image_dataset[:split_index]
train_labels = labels[:split_index]

validation_images = image_dataset[split_index:]
validation_labels = labels[split_index:]

# Create DataLoader
train_dataset = TensorDataset(torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2),
                              torch.tensor(train_labels, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_dataset = TensorDataset(torch.tensor(validation_images, dtype=torch.float32).permute(0, 3, 1, 2),
                                   torch.tensor(validation_labels, dtype=torch.long))
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

total_classes = len(np.unique(labels)) # 6 classes for segmentation

# Init model, loss function, optimizer
model = UNet(total_classes)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device being used: {device}")
model.to(device)

# CSV file path for logging loss and metrics
csv_file_path = os.path.join(metrics_files_path, 'training_metrics.csv')
file_exists = os.path.isfile(csv_file_path)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_jaccard = 0.0
    running_dice = 0.0
    running_pixel_accuracy = 0.0
    running_precision = 0.0
    running_recall = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, labels.squeeze(1))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate metrics for the batch
        preds = torch.argmax(outputs, dim=1)
        batch_jaccard = jaccard_score(labels.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy(), average='micro')
        batch_dice = f1_score(labels.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy(), average='micro')
        batch_pixel_accuracy = (preds == labels.squeeze(1)).float().mean().item()
        batch_precision = precision_score(labels.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy(), average='micro')
        batch_recall = recall_score(labels.view(-1).cpu().numpy(), preds.view(-1).cpu().numpy(), average='micro')
        
        # Accumulate metrics
        running_jaccard += batch_jaccard
        running_dice += batch_dice
        running_pixel_accuracy += batch_pixel_accuracy
        running_precision += batch_precision
        running_recall += batch_recall
        
        # Print the loss and metrics every step
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}, Jaccard: {batch_jaccard:.4f}, Dice: {batch_dice:.4f}, Pixel Accuracy: {batch_pixel_accuracy:.4f}, Precision: {batch_precision:.4f}, Recall: {batch_recall:.4f}')
    
    # Calculate average metrics for the epoch
    avg_loss = running_loss / len(train_loader)
    avg_jaccard = running_jaccard / len(train_loader)
    avg_dice = running_dice / len(train_loader)
    avg_pixel_accuracy = running_pixel_accuracy / len(train_loader)
    avg_precision = running_precision / len(train_loader)
    avg_recall = running_recall / len(train_loader)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    validation_loss = 0.0

    with torch.no_grad():  # No need to compute gradients during validation
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(1))
            validation_loss += loss.item() * images.size(0)  # Accumulate validation loss

    # Calculate average validation loss
    avg_validation_loss = validation_loss / len(validation_loader.dataset)
    print(    f'Epoch [{epoch + 1}/{num_epochs}], Average Validation Loss: {avg_validation_loss:.4f}')
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Average Jaccard: {avg_jaccard:.4f}, Average Dice: {avg_dice:.4f}, Average Pixel Accuracy: {avg_pixel_accuracy:.4f}, Average Precision: {avg_precision:.4f}, Average Recall: {avg_recall:.4f}, Average Validation Loss: {avg_validation_loss:.4f}')
    
    # Validation loop
    model.eval()  # Set model to evaluation mode
    validation_loss = 0.0

    with torch.no_grad():  # No need to compute gradients during validation
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.squeeze(1))
            validation_loss += loss.item() * images.size(0)  # Accumulate validation loss

    # Calculate average validation loss
    avg_validation_loss = validation_loss / len(validation_loader.dataset)
    print(    f'Epoch [{epoch + 1}/{num_epochs}], Average Validation Loss: {avg_validation_loss:.4f}')

    # Save validation loss to CSV file
    # Add a new field in the CSV file to store the validation loss
    with open(csv_file_path, mode='a', newline='') as csv_file:
        fieldnames = ['epoch', 'average_loss', 'average_jaccard', 'average_dice', 'average_pixel_accuracy', 'average_precision', 'average_recall', 'average_validation_loss']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()
            file_exists = True
        
        # Write the average loss and metrics
        writer.writerow({'epoch': epoch + 1, 'average_loss': avg_loss, 'average_jaccard': avg_jaccard, 'average_dice': avg_dice, 'average_pixel_accuracy': avg_pixel_accuracy, 'average_precision': avg_precision, 'average_recall': avg_recall, 'average_validation_loss': avg_validation_loss})

    
    # Save the model checkpoint after each epoch
    model_save_path = os.path.join(model_checkpoint_path, f'unet_epoch_{epoch + 1}.pth')
    torch.save(model.state_dict(), model_save_path)

# Save the final trained model
final_model_save_path = os.path.join(model_checkpoint_path, 'unet_final.pth')
torch.save(model.state_dict(), final_model_save_path)

print('Training complete')

print('Training complete')



