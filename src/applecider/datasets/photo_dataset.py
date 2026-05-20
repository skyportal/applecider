from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from applecider.datasets.oversampler_mixin import OversamplerMixin
from hyrax.datasets import HyraxDataset
from torch.utils.data import Dataset


class PhotoEventsDataset(HyraxDataset, Dataset, OversamplerMixin):
    def __init__(self, config: dict, data_location: Union[Path, str] = None, horizon: float = 10.0):
        self.data_location = data_location
        self.filenames = sorted(list(Path(self.data_location).glob("*.npz")))

        self.photo_config = config["applecider"]["photo_dataset"]

        self.manifest_df = pd.read_csv(self.photo_config["manifest_path"])
        self.manifest_df = self.manifest_df.sort_values("obj_id", inplace=False)
        self.object_ids = self.manifest_df["obj_id"].tolist()
        self.horizon = self.photo_config["horizon"]
        self.st = np.load(Path(self.photo_config["stats_path"]))
        self.use_oversampling = self.photo_config["use_oversampling"]

        # Map original subclass IDs to broader classes
        self.taxonomy_mapper = {
            0: 0,  # SN Ia -> SNI
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

        ideal_class_distribution = self.photo_config["ideal_class_distribution"]
        class_at_index = [self.taxonomy_mapper[label] for label in self.manifest_df.label.tolist()]
        if self.use_oversampling:
            self.prepare_over_sampling(ideal_class_distribution, class_at_index)
        super().__init__(config)

    def get_object_id(self, idx) -> str:
        """get unique identifier for a specific index"""
        # Find the row in the manifest ids
        old_idx = idx
        if self.use_oversampling:
            idx, is_oversampled = self.retrieve_oversampled_index(idx)
        return str(self.object_ids[idx])

    def get_label(self, idx):
        """get ID label for a specific index"""
        old_idx = idx
        if self.use_oversampling:
            idx, is_oversampled = self.retrieve_oversampled_index(idx)
        # Find the row in the manifest
        row = self.manifest_df.iloc[idx]
        return self.taxonomy_mapper[row.label]
        # return self.id2broad_id[int(row.label)]

    def get_photometry(self, idx):
        """get photometry tensor for a specific index"""
        if self.use_oversampling:
            idx, is_oversampled = self.retrieve_oversampled_index(idx)
        # print(self.manifest_df.iloc[idx]["obj_id"], self.filenames[idx], self.manifest_df.iloc[idx].label)
        # import pdb; pdb.set_trace()
        data = np.load(self.filenames[idx], allow_pickle=True)["data"]
        # TODO: Consider caching data to avoid duplicate loads in each epoch

        # Limit length to <100 for memory constraints
        # data = data[:100]

        # TODO: data augmentation

        # Horizon cut: only keep data up to a certain (relative) time
        data = data[data[:, 0] <= self.horizon]

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

    @staticmethod
    def collate_photometry(batch):
        """custom collate function for photometry data"""
        seqs = [i["photometry"] for i in batch]

        lengths = [s.shape[0] for s in seqs]
        max_len = max([257, max(lengths)])

        # Create padding arrays: False where there is data, True where there is padding
        padded = []
        for s in seqs:
            pad_width = ((0, max_len - s.shape[0]), (0, 0))
            padded.append(np.pad(s, pad_width, mode="constant", constant_values=0.0))
        pad = np.stack(padded, axis=0)
        pad_mask = np.stack(
            [np.concatenate([np.zeros(l), np.ones(pad.shape[1] - l)]) for l in lengths]
        ).astype(bool)

        # Truncate to a consistent sequence length
        pad = pad[:, :257, :]
        pad_mask = pad_mask[:, :257]

        return {
            "photometry": pad,
            "pad_mask": pad_mask,
        }

    def get_mean(self, idx):
        """get feature means from stats file"""
        # TODO: Double check this reshaping!!!
        return self.st["mean"].reshape(1, 4)

    def get_std(self, idx):
        """get feature standard deviations from stats file"""
        # TODO: Double check this reshaping!!!
        return self.st["std"].reshape(1, 4)

    def __len__(self):
        if self.use_oversampling:
            return self.total_count_with_oversampling
        else:
            return len(self.filenames)
