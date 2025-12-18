from torch.utils.data import Dataset
from hyrax.data_sets import HyraxDataset
from pathlib import Path
from typing import Union
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from applecider.datasets.oversampler_mixin import OversamplerMixin
import torch
import pandas as pd


class PhotoEventsDataset(HyraxDataset, Dataset, OversamplerMixin):
    def __init__(self, config: dict, data_location: Union[Path, str] = None, horizon: float = 10.0):
        self.data_location = data_location
        self.filenames = sorted(list(Path(self.data_location).glob('*.npz')))
        #self.manifest_df = pd.read_csv(config["data_set"]["PhotoEventsDataset"]["manifest_path"])
        self.manifest_df = pd.read_csv(config["data_set"]["applecider.datasets.photo_dataset.PhotoEventsDataset"]["manifest_path"])
        self.manifest_df = self.manifest_df.sort_values("obj_id", inplace=False)
        self.object_ids = self.manifest_df["obj_id"].tolist()
        self.horizon = config["data_set"]["PhotoEventsDataset"]["horizon"]
        self.st = np.load(Path(config["data_set"]["PhotoEventsDataset"]["stats_path"]))
        self.use_oversampling = config["data_set"]["PhotoEventsDataset"]["use_oversampling"]

        # Map original subclass IDs to broader classes
        self.taxonomy_mapper = {0: 0,  # SN Ia -> SNI
                                1: 0,  # SN Ib -> SNI
                                2: 0,  # SN Ic -> SNI
                                3: 1,  # SN II -> SNII
                                4: 1,  # SN IIP -> SNII
                                5: 1,  # SN IIn -> SNII
                                6: 1,  # SN IIb -> SNII
                                7: 2,  # Cataclysmic -> CV
                                8: 3,  # AGN -> AGN
                                9: 4,  # Tidal Disruption Event -> TDE
                                }

        ideal_class_distribution = config["data_set"]["PhotoEventsDataset"]["ideal_class_distribution"]
        class_at_index = [self.taxonomy_mapper[label] for label in
                          self.manifest_df.label.tolist()]
        if self.use_oversampling:
            self.prepare_over_sampling(ideal_class_distribution, class_at_index)
        super().__init__(config)

    def __getitem__(self, idx):
        # getting happens via the getter methods below
        pass

    def get_object_id(self, idx):
        """get unique identifier for a specific index"""
        # Find the row in the manifest ids
        old_idx = idx
        if self.use_oversampling:
            idx, is_oversampled = self.retrieve_oversampled_index(idx)
        return self.object_ids[idx]

    def ids(self):
        for idx in range(len(self)):
            yield self.get_object_id(idx)

    def get_label(self, idx):
        """get ID label for a specific index"""
        old_idx = idx
        if self.use_oversampling:
            idx, is_oversampled = self.retrieve_oversampled_index(idx)
        # Find the row in the manifest
        row = self.manifest_df.iloc[idx]
        return self.taxonomy_mapper[row.label]
        #return self.id2broad_id[int(row.label)]

    def get_photometry(self, idx):
        """get photometry tensor for a specific index"""
        #if config["use_oversampling"]
        if self.use_oversampling:
            idx, is_oversampled = self.retrieve_oversampled_index(idx)
        #print(self.manifest_df.iloc[idx]["obj_id"], self.filenames[idx], self.manifest_df.iloc[idx].label)
        #import pdb; pdb.set_trace()
        data = np.load(self.filenames[idx], allow_pickle=True)["data"]
        # TODO: Consider caching data to avoid duplicate loads in each epoch

        # Limit length to <100 for memory constraints
        #data = data[:100]

        # TODO: data augmentation

        # Horizon cut: only keep data up to a certain (relative) time
        data = data[data[:,0] <= self.horizon]


        # Grab features from array slices
        dt = np.log1p(data[:, 0])
        dt_prev = np.log1p(data[:, 1])
        logf = data[:, 3]
        logfe = data[:, 4]
        band = data[:, 2]

        # stack non-band features and one-hot encode band
        vec4 = np.stack([dt, dt_prev, logf, logfe], 1)
        one_hot_encoding = np.eye(3, dtype=np.float32)
        one_hot_band = one_hot_encoding[band.astype(np.int64)]  # (L, 3)

        # Result is a (L, 7) tensor (L = sequence length)
        return np.concatenate([vec4, one_hot_band], 1)  # (L, 7)

    def get_mean(self, idx):
        """get feature means from stats file"""
        return self.st['mean']

    def get_std(self, idx):
        """get feature standard deviations from stats file"""
        return self.st['std']

    def __len__(self):
        if self.use_oversampling:
            return self.total_count_with_oversampling
        else:
            return len(self.filenames)

    @staticmethod
    def collate(batch):
        '''custom collate function for photo events dataset'''
        seqs = []
        labels = []
        for i in batch:
            seqs += [i["data"]["photometry"]]
            if "label" in i["data"]:
                labels += [i["data"]["label"]]

        lengths = [s.shape[0] for s in seqs]
        max_len = max([257, max(lengths)])

        # Create padding arrays: False where there is data, True where there is padding
        padded = []
        for s in seqs:
            pad_width = ((0, max_len - s.shape[0]), (0, 0))
            padded.append(np.pad(s, pad_width, mode='constant', constant_values=0.0))
        pad = np.stack(padded, axis=0)
        pad_mask = np.stack([np.concatenate([np.zeros(l), np.ones(pad.shape[1]-l)]) for l in lengths]).astype(bool)

        # Truncate to a consistent sequence length
        pad = pad[:, :257, :]
        pad_mask = pad_mask[:, :257]

        return {
            "data": {
                "photometry": pad,
                "label": np.array(labels),
                "pad_mask": pad_mask,
                "mean": np.array(batch[0]["data"]["mean"]),
                "std": np.array(batch[0]["data"]["std"]),
            },
        }
