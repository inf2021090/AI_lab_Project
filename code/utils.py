import os
import numpy as np

def hex_2_array(hex_color):
    hex_color = hex_color.lstrip('#')
    color_array = np.array(tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))
    print(color_array)
    return color_array

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y.reshape(-1)].reshape(y.shape[:-1] + (num_classes,))


