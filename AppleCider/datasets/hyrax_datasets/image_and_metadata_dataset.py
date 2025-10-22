import os
import numpy as np
import torch

from hyrax.data_sets.data_set_registry import HyraxDataset

EPS = 1e-8  # Small value to prevent division by zero
REAL_CLASSES = [
    "AGN",
    "TDE",
    "SN II",
    "SN IIp",
    "SN Ia",
    "SN IIn",
    "SN Ib",
    "SN Ic",
    "Cataclysmic"
]

class ImageAndMetadataDataset(HyraxDataset):
    def __init__(self, config, data_location):

        self.dataset_config = config['data_set']['ImageAndMetadataDataset']

        self.all_samples = self.dataset_config['all_samples']
        self.augment = self.dataset_config['augment']
        self.classes = self.dataset_config['classes']

        file_names = sorted([f for f in os.listdir(data_location) if f.endswith('.npy')])

        self.raw_files = [np.load(os.path.join(data_location, file_name), allow_pickle=True).item()
                for file_name in file_names]

        self.image_cache = {}

        super().__init__(config)
        # Additional initialization for image and metadata dataset can be added here

    def get_metadata(self, index):
        # Method to retrieve metadata at the specified index
        return self.raw_files[index].get('metadata')

    def get_image(self, index):
        # Method to retrieve image at the specified index
        if index in self.image_cache:
            image = self.image_cache[index]
        else:
            image = self.raw_files[index].get('image')

            if "vit_tower" in self.dataset_config["tags"]:
                i1 = int((63-self.dataset_config["patch_size"][0])/2)
                i2 = int(63 - i1)

            elif not self.dataset_config["cutout_size"]==63:
                i1 = int((63-self.dataset_config["cutout_size"])/2)
                i2 = int(63 - i1)

            else:
                i1 = 0
                i2 = 63
            image = image[:, i1:i2, i1:i2]

            if self.dataset_config['image_norm'] == 'median':
                for c in range(3):
                    channel_median = torch.median(image[c].reshape(-1))
                    image[c] = image[c] - channel_median
                    image[c] = image[c]/(image[c].std()+ EPS)

            elif self.dataset_config['image_norm'] == 'L2':
                norm = torch.norm(image, p=2)  # L2 norm over all elements
                image = image/norm

            self.image_cache[index] = image

        return image


    def get_target(self, index):
        #! I don't know what this does, please confirm correct behavior !!!
        # Method to retrieve target at the specified index
        original_class = self.raw_files[index].get('target')
        target = np.zeros(len(self.classes))

        for idy, category in enumerate(self.classes):
            if original_class in category:
                target[idy] = 1.0

        return target


    def get_real_target(self, index):
        #! I don't know what this does, please confirm correct behavior !!!
        # Method to retrieve real target at the specified index
        original_class = self.raw_files[index].get('target')
        real_target = np.zeros(9)

        for idy, category in enumerate(REAL_CLASSES):
            if original_class == category:
                real_target[idy] = 1.0

        return real_target

    def get_obj_id(self, index):
        # Method to retrieve object ID at the specified index
        return self.raw_files[index].get('obj_id')

    def __len__(self):
        # Return the total number of items in the dataset
        return len(self.raw_files)

    def __getitem__(self, index):
        # Unused, but required by Hyrax to show inheritance from PyTorch Dataset.
        pass
