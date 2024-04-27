import os
import cv2
from PIL import Image
import logging
from dataset import SegmentationDataset

def check_images(dataset):
    for tile_id in range(len(dataset) - 1):
        images_data, masks_data = dataset[tile_id]
        for image, _ in images_data:
            print("Image shape:", image.shape)
            logging.info("Image shape: %s", image.shape)
        for mask, _ in masks_data:
            print("Mask shape:", mask.shape)
            logging.info("Mask shape: %s", mask.shape)

def main():
    # logging configuration
    if not os.path.exists('../docs/logs'):
        os.makedirs('../docs/logs')
    image_log_file = os.path.join('../docs/logs', 'image_shapes.log')
    logging.basicConfig(filename=image_log_file, filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # dataset files
    dataset_root_folder = '../data'
    dataset_name = "dubai_dataset"

    # new obj
    dataset = SegmentationDataset(dataset_dir=os.path.join(dataset_root_folder, dataset_name))
    print("Total number of tiles:", len(dataset))

    check_images(dataset)

if __name__ == "__main__":
    main()


