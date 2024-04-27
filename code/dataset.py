import os
import cv2
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
                if image is not None:
                    image_data.append((image, img_path))
        

        # collect masks
        for mask_file in sorted(os.listdir(masks_dir)):
            if mask_file.endswith('.png'):
                mask_path = os.path.join(masks_dir, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_data.append((mask, mask_path))

        return image_data , mask_data

'''
        image_files = sorted(os.listdir(images_dir))
        mask_files = sorted(os.listdir(masks_dir))

        # check if number of images and masks is the same
        assert len(image_files) == len(mask_files)

        data = []
        for img_name, mask_name in zip(image_files, mask_files):
            img_path = os.path.join(images_dir, img_name)
            mask_path = os.path.join(masks_dir, mask_name)

            data.append((img_path, mask_path))

        return data
'''
# test 
if __name__ == "__main__":
    dataset_root_folder = '..data/'
    dataset_name = "dubai_dataset"

    dataset = SegmentationDataset(dataset_dir="../data/dubai_dataset")
    print("Total number of tiles:", len(dataset) - 1)

    for path, subdirs, files in os.walk(os.path.join(dataset_root_folder, dataset_name)):
        dir_name = path.split(os.path.sep)[-1]
        #print(dir_name)
        if dir_name == 'masks': # 'images
            images = os.listdir(path)
            print(path)
            #print(images)
            for i, image_name in enumerate(images):
                if (image_name.endswith('.png')): # '.jpg
                    #print(image_name)
                    a = True
