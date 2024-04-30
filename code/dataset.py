import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from torch.utils.data import Dataset
from utils import hex_2_array

from matplotlib import pyplot as plt
import random
import logging

class SatelliteImageSegmentation(Dataset):
    def __init__(self, dataset_path, patch_size=256):
        self.dataset_path = dataset_path
        self.patch_size = patch_size
        self.logger = logging.getLogger(__name__)  # Initialize logger
        self.class_colors = {
            'building': hex_2_array('#3C1098'),
            'land': hex_2_array('#8429F6'),
            'road': hex_2_array('#6EC1E4'),
            'vegetation': hex_2_array('#FEDD3A'),
            'water': hex_2_array('#E2A929'),
            'unlabeled': hex_2_array('#9B9B9B')
        }

    def load_image(self, path):
        return cv2.imread(path, 1)

    def normalize_image(self, image):
         image = image/255.0
         return image
    
    def rgb_2_label(self, label):
        label_segment = np.zeros(label.shape, dtype=np.uint8)

        label_segment[np.all(label == self.class_colors['water'], axis=-1)] = 0
        label_segment[np.all(label == self.class_colors['land'], axis=-1)] = 1
        label_segment[np.all(label == self.class_colors['road'], axis=-1)] = 2
        label_segment[np.all(label == self.class_colors['building'], axis=-1)] = 3
        label_segment[np.all(label == self.class_colors['vegetation'], axis=-1)] = 4
        label_segment[np.all(label == self.class_colors['unlabeled'], axis=-1)] = 5
    
        #print(label_segment)
        label_segment = label_segment[:,:,0]
        #print(label_segment)
        return label_segment

    def load_dataset(self):
        image_dataset = []
        mask_dataset = []
        labels = []
        img_log_path = "../docs/logs/img_shape.log"
        msk_log_path = "../docs/logs/msk_shape.log"
        logging.basicConfig(filename=img_log_path, level=logging.INFO)  # Set up logging configuration
        for image_type in ['images', 'masks']:
            if image_type == 'images':
                image_extension = 'jpg'
            else:
                image_extension = 'png'
            for tile_id in range(1, 8):
                for image_id in range(1, 20):
                    image_path = f'{self.dataset_path}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}'
                    if os.path.exists(image_path):  # Check if image path exists
                        image = self.load_image(image_path)
                        if image is not None:
                            if image_type == 'masks':
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            size_x = (image.shape[1] // self.patch_size) * self.patch_size
                            size_y = (image.shape[0] // self.patch_size) * self.patch_size
                            image = Image.fromarray(image)
                            image = image.crop((0, 0, size_x, size_y))
                            image = np.array(image)
                            patches = patchify(image, (self.patch_size, self.patch_size, 3), step=self.patch_size)
                            for i in range(patches.shape[0]):
                                for j in range(patches.shape[1]):
                                    if image_type == 'images':
                                        patch_image = patches[i, j, :, :]
                                        patch_image = self.normalize_image(patch_image)
                                        patch_image = patch_image[0]#
                                        image_dataset.append(patch_image)
                                    elif image_type == 'masks':
                                        patch_mask = patches[i,j,:,:]
                                        patch_mask = patch_mask[0]
                                        mask_dataset.append(patch_mask)
                                        #labels
                                    # Log the shape of the individual patched image
                                    self.logger.info(f"Shape of patched image {tile_id}-{image_id}-{i}-{j}: {patch_image.shape}")

        # Process masks to labels
        for i in range(len(mask_dataset)):
            label = self.rgb_2_label(mask_dataset[i])
            labels.append(label)

        labels = np.expand_dims(np.array(labels), axis=3)                           
        mask_dataset = np.array(mask_dataset)
        return np.array(image_dataset), mask_dataset, labels


path = "../data/dubai_dataset"
segmentation = SatelliteImageSegmentation(path)
image_dataset, mask_dataset, labels = segmentation.load_dataset()
print("Image dataset shape:", image_dataset.shape)
print("Mask dataset shape:", mask_dataset.shape)
print("Labelst shape:", labels.shape)

print(len(image_dataset))
print(type(image_dataset[0]))
print(len(labels))


print(type(np.reshape(image_dataset[0], (256, 256, 3))))


random_image_id = random.randint(0, len(image_dataset))
plt.figure(figsize=(14,8))
plt.subplot(121)
plt.imshow(image_dataset[random_image_id])
plt.subplot(122)
#plt.imshow(mask_dataset[random_image_id])
plt.imshow(labels[random_image_id][:,:,0])
plt.show()