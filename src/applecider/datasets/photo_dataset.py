from torch.utils.data import Dataset
from hyrax.data_sets import HyraxDataset
from pathlib import Path
from typing import Union
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd


class PhotoEventsDataset(HyraxDataset, Dataset):
    def __init__(self, config: dict, data_location: Union[Path, str] = None, horizon: float = 10.0):
        self.data_location = data_location
        self.filenames = sorted(list(Path(self.data_location).glob('*.npz')))
        #self.manifest_df = pd.read_csv(config["data_set"]["PhotoEventsDataset"]["manifest_path"])
        self.manifest_df = pd.read_csv(config["data_set"]["applecider.datasets.photo_dataset.PhotoEventsDataset"]["manifest_path"])
        self.object_ids = self.manifest_df["obj_id"].tolist()
        self.horizon = horizon
        self.st = np.load(Path(config["data_set"]["PhotoEventsDataset"]["stats_path"]))

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

        # Do we need to define test vs train data?
        super().__init__(config)

    def __getitem__(self, idx):
        # getting happens via the getter methods below
        pass
    
    def get_object_id(self, idx):
        """get unique identifier for a specific index"""
        # Find the row in the manifest ids
        return self.object_ids[idx]

    def ids(self):
        for idx in range(len(self)):
            yield self.object_ids[idx]

    def get_label(self, idx):
        """get ID label for a specific index"""
        # Find the row in the manifest
        row = self.manifest_df.iloc[idx]
        #import pdb;pdb.set_trace()
        return self.taxonomy_mapper[row.label]
        #return self.id2broad_id[int(row.label)]

    def get_photometry(self, idx):
        """get photometry tensor for a specific index"""
        data = np.load(self.filenames[idx], allow_pickle=True)["data"]
        # TODO: Consider caching data to avoid duplicate loads in each epoch

        # Grab features from array slices
        dt = data[:, 0]
        dt_prev = data[:, 1]
        logf = data[:, 3]
        logfe = data[:, 4]
        band = data[:, 2]

        # stack non-band features and one-hot encode band
        vec4 = np.stack([dt, dt_prev, logf, logfe], 1)
        one_hot_encoding = np.eye(3, dtype=np.float32)
        one_hot_band = one_hot_encoding[band.astype(np.int64)]  # (L, 3)

        # Result is a (L, 7) tensor (L = sequence length)
        return torch.from_numpy(np.concatenate([vec4, one_hot_band], 1))  # (L, 7)

    def get_mean(self, idx):
        """get feature means from stats file"""
        return self.st['mean']

    def get_std(self, idx):
        """get feature standard deviations from stats file"""
        return self.st['std']

    def __len__(self):
        return len(self.filenames)


def collate(batch):
    '''custom collate function for photo events dataset'''

    seqs = []
    labels = []
    for i in batch:
        seqs += [i["data"]["photometry"]]
        labels += [i["data"]["label"]]

    lens = [s.size(0) for s in seqs]
    pad  = pad_sequence(seqs, batch_first=True)              # (B, L, 7)
    mask = torch.stack([torch.cat([torch.zeros(l), torch.ones(pad.size(1)-l)]) for l in lens]).bool()
    # adjust padding mask to account for CLS at idx=0
    pad_mask = torch.cat(
            [torch.zeros(len(batch), 1, device=mask.device, dtype=torch.bool),
             mask], dim=1
        )
    object_ids = [b["object_id"] for b in batch]
    return {"data": {"photometry": pad,
                     "label": torch.tensor(labels),
                     "pad_mask": pad_mask,
                     "mean": torch.tensor(batch[0]["data"]["mean"]),
                     "std": torch.tensor(batch[0]["data"]["std"]),
                     },
            "object_id": object_ids,
            }
