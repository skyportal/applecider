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
        self.manifest_df = pd.read_csv(config["data_set"]["manifest_path"])
        self.horizon = horizon
        self.st = np.load(Path(config["data_set"]["stats_path"]))

        # Taxonomy setup
        broad_classes = ["SNI", "SNII", "CV", "AGN", "TDE"]
        orig2broad = {
                "SN Ia": "SNI", "SN Ib": "SNI", "SN Ic": "SNI",
                "SN II": "SNII", "SN IIP": "SNII", "SN IIn": "SNII", "SN IIb": "SNII",
                "Cataclysmic": "CV", "AGN": "AGN", "Tidal Disruption Event": "TDE",
                }

        broad2id = {c: i for i, c in enumerate(broad_classes)}
        subclass_id2name = [
            "SN Ia", "SN Ib", "SN Ic", "SN II", "SN IIP", "SN IIn", "SN IIb",
            "Cataclysmic", "AGN", "Tidal Disruption Event"
            ]
        self.id2broad_id = {i: broad2id[orig2broad[name]] for i, name in enumerate(subclass_id2name)}

        # Do we need to define test vs train data?
        super().__init__(config)

    def __getitem__(self, idx):
        # load the data from disk
        return self.get_photometry(idx)
    
    def get_id(self,idx):
        """get unique identifier for a specific index"""
        # Find the row in the manifest
        return self.manifest_df.iloc[idx]["obj_id"]


    def get_label(self, idx):
        """get ID label for a specific index"""
        # Find the row in the manifest
        row = self.manifest_df.iloc[idx]

        # Manifest contains a path to another file that should live in data_location
        f_name = row.filepath.split("/")[-1]
        raw = np.load(Path(self.data_location) / f_name, allow_pickle=True)
        arr = raw['data'] if isinstance(raw, np.lib.npyio.NpzFile) else raw
        if self.horizon:
            arr = arr[arr[:, 0] <= self.horizon]
        return self.id2broad_id[int(row.label)]

    def get_photometry(self, idx):
        """get photometry tensor for a specific index"""
        data = np.load(self.filenames[idx], allow_pickle=True)["data"]

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
    return {"data": {"photometry": pad,
                     "label": torch.tensor(labels),
                     "pad_mask": pad_mask,
                     "mean": torch.tensor(batch[0]["data"]["mean"]),
                     "std": torch.tensor(batch[0]["data"]["std"]),
                     }
            }
