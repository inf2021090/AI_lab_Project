import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import csv
from dataset import SatelliteImageSegmentation 
from model import UNet 
from utils import create_dir
import evaluation

def main():
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

    # Split dataset into train and validation
    split_ratio = 0.8
    train_size = int(split_ratio * len(image_dataset))
    val_size = len(image_dataset) - train_size

    train_images, val_images = torch.utils.data.random_split(image_dataset, [train_size, val_size])
    train_labels, val_labels = torch.utils.data.random_split(labels, [train_size, val_size])

    # Create DataLoader for training and validation
    train_dataset = TensorDataset(torch.tensor(train_images, dtype=torch.float32).permute(0, 3, 1, 2),
                                  torch.tensor(train_labels, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.tensor(val_images, dtype=torch.float32).permute(0, 3, 1, 2),
                                torch.tensor(val_labels, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    total_classes = len(np.unique(labels)) # 6 classes for segmentation

    # Init model, loss function, optimizer
    model = UNet(total_classes)  
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # CSV file path for logging loss and evaluation metrics
    csv_file_path = os.path.join(metrics_files_path, 'training_loss_metrics.csv')
    file_exists = os.path.isfile(csv_file_path)

    # Training loop
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ['epoch', 'train_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_jaccard']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

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

            # Log average loss per epoch
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

            # Evaluation after each epoch
            model.eval()  # Switch model to evaluation mode
            with torch.no_grad():  # No gradient calculation during evaluation
                val_loss = 0.0
                total_samples = 0
                correct_predictions = 0
                precisions = []
                recalls = []
                f1_scores = []
                jaccard_coeffs = []

                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    predicted_labels = torch.argmax(outputs, dim=1)

                    # Calculate evaluation metrics
                    val_loss += criterion(outputs, labels.squeeze(1)).item()
                    total_samples += labels.size(0)
                    correct_predictions += (predicted_labels == labels.squeeze(1)).sum().item()

                    acc = evaluation.accuracy(labels, predicted_labels)
                    prec = evaluation.precision(labels, predicted_labels)
                    rec = evaluation.recall(labels, predicted_labels)
                    f1 = evaluation.f1_score(labels, predicted_labels)
                    jaccard = evaluation.jaccard_coef(labels, predicted_labels)

                    precisions.append(prec)
                    recalls.append(rec)
                    f1_scores.append(f1)
                    jaccard_coeffs.append(jaccard)

                val_accuracy = correct_predictions / total_samples
                val_precision = np.mean(precisions)
                val_recall = np.mean(recalls)
                val_f1 = np.mean(f1_scores)
                val_jaccard = np.mean(jaccard_coeffs)

                # Print or log the metrics
                print(f"Validation Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}, Jaccard Coefficient: {val_jaccard:.4f}")

                # Write the metrics to CSV file
                writer.writerow({'epoch': epoch + 1,
                                 'train_loss': avg_loss,
                                 'val_accuracy': val_accuracy,
                                 'val_precision': val_precision,
                                 'val_recall': val_recall,
                                 'val_f1': val_f1,
                                 'val_jaccard': val_jaccard})

            # Save the model checkpoint after each epoch
            model_save_path = os.path.join(model_checkpoint_path, f'unet_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_save_path)

    # Save the final trained model
    final_model_save_path = os.path.join(model_checkpoint_path, 'unet_final.pth')
    torch.save(model.state_dict(), final_model_save_path)

    print('Training complete')

if __name__ == "__main__":
    main()






