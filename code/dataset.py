import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from torch.utils.data import Dataset

from matplotlib import pyplot as plt
import random
import logging

class SatelliteImageSegmentation(Dataset):
    def __init__(self, dataset_path, image_patch_size=256):
        self.dataset_path = dataset_path
        self.image_patch_size = image_patch_size
        self.logger = logging.getLogger(__name__)  # Initialize logger

    def load_image(self, path):
        return cv2.imread(path, 1)

    def normalize_image(self, image):
         image = image/255.0
         return image

    def load_dataset(self):
        image_dataset = []
        log_path = "../docs/logs/img_shape.log"
        logging.basicConfig(filename=log_path, level=logging.INFO)  # Set up logging configuration
        for image_type in ['images']:
            if image_type == 'images':
                image_extension = 'jpg'
            for tile_id in range(1, 8):
                for image_id in range(1, 20):
                    image_path = f'{self.dataset_path}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}'
                    if os.path.exists(image_path):  # Check if image path exists
                        image = self.load_image(image_path)
                        if image is not None:
                            size_x = (image.shape[1] // self.image_patch_size) * self.image_patch_size
                            size_y = (image.shape[0] // self.image_patch_size) * self.image_patch_size
                            image = Image.fromarray(image)
                            image = image.crop((0, 0, size_x, size_y))
                            image = np.array(image)
                            patched_images = patchify(image, (self.image_patch_size, self.image_patch_size, 3), step=self.image_patch_size)
                            for i in range(patched_images.shape[0]):
                                for j in range(patched_images.shape[1]):
                                    individual_patched_image = patched_images[i, j, :, :]
                                    individual_patched_image = self.normalize_image(individual_patched_image)
                                    individual_patched_image = individual_patched_image[0]#
                                    image_dataset.append(individual_patched_image)
                                    # Log the shape of the individual patched image
                                    self.logger.info(f"Shape of patched image {tile_id}-{image_id}-{i}-{j}: {individual_patched_image.shape}")

        return np.array(image_dataset)


path = "../data/dubai_dataset"
segmentation = SatelliteImageSegmentation(path)
image_dataset = segmentation.load_dataset()
print("Image dataset shape:", image_dataset.shape)


print(len(image_dataset))
print(type(image_dataset[0]))


print(type(np.reshape(image_dataset[0], (256, 256, 3))))


random_image_id = random.randint(0, len(image_dataset))
plt.figure(figsize=(14,8))
plt.subplot(121)
plt.imshow(image_dataset[random_image_id])
plt.subplot(122)