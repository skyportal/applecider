from operator import index
import os
import numpy as np
import torch

from hyrax.data_sets.data_set_registry import HyraxDataset

from applecider.datasets.oversampler import Oversampler

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

CLASSES = [
    ['SN Ia','SN Ic','SN Ib'],
    ['SN IIP', 'SN IIn','SN II', 'SN IIb'],
    ['Cataclysmic'],
    ['AGN'],
    ['Tidal Disruption Event']
]

class ImageAndMetadataDataset(HyraxDataset, Oversampler):
    def __init__(self, config, data_location):

        self.dataset_config = config['data_set']['ImageAndMetadataDataset']

        self.all_samples = self.dataset_config['all_samples']
        self.augment = self.dataset_config['augment']

        file_names = sorted([f for f in os.listdir(data_location) if f.endswith('.npy')])

        self.raw_files = [np.load(os.path.join(data_location, file_name), allow_pickle=True).item()
                for file_name in file_names]

        self.obj_ids = [file.get('obj_id') for file in self.raw_files]

        self.enable_cache = self.dataset_config['enable_image_cache']
        self.image_cache = {}

        # Look through each data sample, and get it's class index.
        self.class_at_index = np.zeros(len(self.raw_files))
        self.class_counts = np.zeros(len(CLASSES), dtype=np.int64)
        for file_indx, file in enumerate(self.raw_files):
            original_class = file.get('target')

            for idy, category in enumerate(CLASSES):
                if original_class in category:
                    self.class_at_index[file_indx] = idy
                    self.class_counts[idy] += 1
                    continue

        # Produce the counts of each class in the dataset.
        self._calculate_over_sampling_counts(self.dataset_config["class_distribution"])
        self.original_count = len(self.raw_files)

        super().__init__(config)
        # Additional initialization for image and metadata dataset can be added here

    def _get_class_counts(self):
        return self.class_counts

    def _get_class_at_index(self):
        return self.class_at_index

    def get_metadata(self, index):
        # Method to retrieve metadata at the specified index
        index, is_oversampled = self.retrieve_index(index)
        return self.raw_files[index].get('metadata')

    def get_image(self, index):
        # Method to retrieve image at the specified index
        if self.enable_cache and index in self.image_cache:
            image = self.image_cache[index]
        else:
            index, is_oversampled = self.retrieve_index(index)
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

            if self.enable_cache:
                self.image_cache[index] = image

        return image


    def get_target(self, index):
        """The `target` is broad class category of the object.

        Parameters
        ----------
        index : int
            The index of the sample in the dataset.

        Returns
        -------
        np.ndarray
            The one hot target vector for the specified index.
        """
        index_found, is_oversampled = self.retrieve_index(index)
        original_class = self.raw_files[index_found].get('target')
        target = np.zeros(len(CLASSES))

        for idy, category in enumerate(CLASSES):
            if original_class in category:
                target[idy] = 1.0

        return target


    def get_real_target(self, index):
        """The `real_target` is the fine-grained classification of the object.

        Parameters
        ----------
        index : int
            The index of the sample in the dataset.

        Returns
        -------
        np.ndarray
            The one hot real target vector for the specified index.
        """
        index_found, is_oversampled = self.retrieve_index(index)
        original_class = self.raw_files[index_found].get('target')
        real_target = np.zeros(len(REAL_CLASSES))

        for idy, category in enumerate(REAL_CLASSES):
            if original_class == category:
                real_target[idy] = 1.0

        return real_target


    def get_obj_id(self, index):
        # Method to retrieve object ID at the specified index
        index_found, is_oversampled = self.retrieve_index(index)
        return self.raw_files[index_found].get('obj_id')


    def ids(self):
        # Generator to yield all object IDs in the dataset
        for idx in range(len(self)):
            yield self.get_obj_id(idx)

    def __len__(self):
        # Return the total number of items in the dataset
        return self.total_count_with_oversampling


    def __getitem__(self, index):
        # Unused, but required by Hyrax to show inheritance from PyTorch Dataset.
        pass
