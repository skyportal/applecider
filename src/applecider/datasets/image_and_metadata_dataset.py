from operator import index
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

CLASSES = [
    ['SN Ia','SN Ic','SN Ib'],
    ['SN IIP', 'SN IIn','SN II', 'SN IIb'],
    ['Cataclysmic'],
    ['AGN'],
    ['Tidal Disruption Event']
]

class ImageAndMetadataDataset(HyraxDataset):
    def __init__(self, config, data_location):

        self.dataset_config = config['data_set']['ImageAndMetadataDataset']

        self.all_samples = self.dataset_config['all_samples']
        self.augment = self.dataset_config['augment']
        # self.classes = self.dataset_config['classes']

        file_names = sorted([f for f in os.listdir(data_location) if f.endswith('.npy')])

        self.raw_files = [np.load(os.path.join(data_location, file_name), allow_pickle=True).item()
                for file_name in file_names]

        self.obj_ids = [file.get('obj_id') for file in self.raw_files]

        self.enable_cache = self.dataset_config['enable_image_cache']
        self.image_cache = {}

        # Look through each data sample, and get it's class index.
        self.unique_ids_per_class = [set() for _ in range(len(CLASSES))]
        self.class_at_index = np.zeros(len(self.raw_files))
        for file_indx, file in enumerate(self.raw_files):
            original_class = file.get('target')

            for idy, category in enumerate(CLASSES):
                if original_class in category:
                    self.class_at_index[file_indx] = idy
                    self.unique_ids_per_class[idy].add(self.obj_ids[file_indx])

                    continue

        # Produce the counts of each class in the dataset.
        _, self.class_counts = np.unique(self.class_at_index, return_counts=True)
        self.original_count = np.sum(self.class_counts)
        self._calculate_over_sampling_counts()
        self.over_sampled_count = np.sum(self.class_counts + self.additions)



        super().__init__(config)
        # Additional initialization for image and metadata dataset can be added here


    def _calculate_over_sampling_counts(self):
        unnormalized_percentages = np.array(self.dataset_config["class_distribution"])#[0.4, 0.1, 0.1, 0.35, 0.05])
        p_norm = unnormalized_percentages / np.sum(unnormalized_percentages)
        total_current = np.sum(self.class_counts)
        
        req_totals = np.zeros_like(p_norm, dtype=np.int64)
        nonzero_mask = p_norm > 0
        req_totals[nonzero_mask] = np.ceil(self.class_counts[nonzero_mask] / p_norm[nonzero_mask]).astype(np.int64)

        # minimal feasible total is the maximum of those and at least current total
        minimal_total = max(int(req_totals.max()), int(total_current))

        # build integer target counts for minimal_total using floor + allocate remainder by fractional parts
        target_real = p_norm * minimal_total
        target_floor = np.floor(target_real).astype(np.int64)
        remainder = minimal_total - target_floor.sum()
        if remainder > 0:
            residuals = target_real - target_floor
            # indices sorted by residual descending
            order = np.argsort(residuals)[::-1]
            for idx in order[:remainder]:
                target_floor[idx] += 1

        target_counts = target_floor
        self.additions = target_counts - self.class_counts

    def get_over_sampled_class(self, index):
        # Determine which class the given index corresponds to in the over-sampled dataset
        if index < self.original_count:
            return self.raw_files[index], False
        else:
            over_sampled_index = index - self.original_count
            # calculate the cumulative sum of the additions, find the index of the class
            cumulative_additions = np.cumsum(self.additions)
            # find the class index where the over_sampled_index would fit
            class_index = np.searchsorted(cumulative_additions, over_sampled_index, side='left')
            # randomly select an index from self.class_at_index where class matches class_index
            class_indices = np.where(self.class_at_index == class_index)[0]
            random_choice = np.random.choice(class_indices)
            return self.raw_files[random_choice], True


    def get_metadata(self, index):
        # Method to retrieve metadata at the specified index
        file, is_oversampled = self.get_over_sampled_class(index)
        return file.get('metadata')

    def get_image(self, index):
        # Method to retrieve image at the specified index
        if self.enable_cache and index in self.image_cache:
            image = self.image_cache[index]
        else:
            file, is_oversampled = self.get_over_sampled_class(index)
            image = file.get('image')

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
        file, is_oversampled = self.get_over_sampled_class(index)
        original_class = file.get('target')
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
        file, is_oversampled = self.get_over_sampled_class(index)
        original_class = file.get('target')
        real_target = np.zeros(len(REAL_CLASSES))

        for idy, category in enumerate(REAL_CLASSES):
            if original_class == category:
                real_target[idy] = 1.0

        return real_target


    def get_obj_id(self, index):
        # Method to retrieve object ID at the specified index
        file, is_oversampled = self.get_over_sampled_class(index)
        return file.get('obj_id')


    def ids(self):
        # Generator to yield all object IDs in the dataset
        for idx in range(len(self)):
            yield self.get_obj_id(idx)

    def __len__(self):
        # Return the total number of items in the dataset
        return self.over_sampled_count


    def __getitem__(self, index):
        # Unused, but required by Hyrax to show inheritance from PyTorch Dataset.
        pass
