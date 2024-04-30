import os
import numpy as np

def hex_2_array(hex_color):
    hex_color = hex_color.lstrip('#')
    color_array = np.array(tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4)))
    print(color_array)
    return color_array


