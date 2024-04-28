import os
import cv2
import numpy as np
from patchify import patchify
import logging
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.tiles_dirs = sorted(os.listdir(dataset_dir))

        # print the names of the files in the dataset folder
        print("Files in dataset folder:")
        for file_name in os.listdir(dataset_dir):
            print(f"  {file_name}")

    def __len__(self):
        return len(self.tiles_dirs)

    def __getitem__(self, idx):
        tile_dir = self.tiles_dirs[idx]
        images_dir = os.path.join(self.dataset_dir, tile_dir, 'images')
        masks_dir = os.path.join(self.dataset_dir, tile_dir, 'masks')

        image_data = []
        mask_data = []

        # collect images
        for image_file in sorted(os.listdir(images_dir)):
            if image_file.endswith('.jpg'):
                img_path = os.path.join(images_dir, image_file)
                image = cv2.imread(img_path, 1)
                image = np.array(image)
                if image is not None:

                    size_x = (image.shape[1]//256*256)
                    size_y = (image.shape[0]//256*256)
                    image = image[0:size_y, 0:size_x]  # Crop the image to a multiple of patch size

                    patches = patchify(image, (256, 256, 3), step=256)
                    patches = patches.reshape(-1, 256, 256, 3) # check

                    # Normalize each patch to [0, 1]
                    patches = patches / 255.0

                    image_data.extend([(patch, img_path) for patch in patches])
        
        # collect masks
        for mask_file in sorted(os.listdir(masks_dir)):
            if mask_file.endswith('.png'):
                mask_path = os.path.join(masks_dir, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_data.append((mask, mask_path))

        return image_data, mask_data

# test 
if __name__ == "__main__":
    # logging configuration
    if not os.path.exists('../docs/logs'):
        os.makedirs('../docs/logs')
    image_log_file = os.path.join('../docs/logs', 'image_shapes.log')
    logging.basicConfig(filename=image_log_file, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dataset_root_folder = '../data'
    dataset_name = "dubai_dataset"

    dataset = SegmentationDataset(dataset_dir=os.path.join(dataset_root_folder, dataset_name))
    print("Total number of tiles:", len(dataset) - 1)

    for tile_id in range(len(dataset) - 1):
        images_data, masks_data = dataset[tile_id]
        
        print("Number of patches:", len(images_data))
        for image, _ in images_data:
            #print("Image shape:", image.shape)
            logging.info("Image shape: %s", image.shape)
