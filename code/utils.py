import os
import torch
import numpy as np
import torch
import shutil

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def remove_dir(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print(f"Directory '{directory_path}' and its contents have been deleted.")
    else:
        print(f"Directory '{directory_path}' does not exist.")

def hex_2_array(hex_color):
    hex_color = hex_color.lstrip('#')
    color_array = np.array(tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))
    print(color_array)
    return color_array

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y.reshape(-1)].reshape(y.shape[:-1] + (num_classes,))


def jaccard_coef(y_true, y_pred):
    # flatten the tensors
    y_true_flatten = y_true.view(-1)
    y_pred_flatten = y_pred.view(-1)

    # calculate I and U
    intersection = torch.sum(y_true_flatten * y_pred_flatten)
    union = torch.sum(y_true_flatten) + torch.sum(y_pred_flatten) - intersection

    # calculate Jaccard coefficient
    jaccard = (intersection + 1.0) / (union + 1.0)

    return jaccard





