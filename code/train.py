import os
import glob
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from dataset import SatelliteImageSegmentation
from utils import to_categorical
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path = "../data/dubai_dataset"
    segmentation = SatelliteImageSegmentation(path)
    image_dataset, mask_dataset, labels = segmentation.load_dataset()
    print("Image dataset shape:", image_dataset.shape)
    print("Mask dataset shape:", mask_dataset.shape)
    print("Labels shape:", labels.shape)

    n_classes = len(np.unique(labels))

    labels_categorical_dataset = to_categorical(labels, n_classes)
    print(type(labels_categorical_dataset))

    master_trianing_dataset = image_dataset

    X_train, X_test, y_train, y_test = train_test_split(master_trianing_dataset, labels_categorical_dataset, test_size=0.15, random_state=100)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    random_image_id = random.randint(0, len(image_dataset))
    plt.figure(figsize=(14,8))
    plt.subplot(121)
    plt.imshow(image_dataset[random_image_id])
    plt.subplot(122)
    #plt.imshow(mask_dataset[random_image_id])
    plt.imshow(labels[random_image_id][:,:,0])
    plt.show()






