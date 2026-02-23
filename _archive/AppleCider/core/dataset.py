import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CiDErDataset(Dataset):
    def __init__(self, config, split="train"):
        super().__init__()

        self.split = split  #  'train', 'val', 'test'
        self.mode = config["mode"]  #  'all', 'ztf', 'photo', 'metadata & images', 'spectra'
        self.preprocessed_path = config["preprocessed_path"]
        self.step = config["step"]
        self.random_seed = config["random_seed"]  # 42, 66, 0, 12, 123

        # for ALERTS
        self.df_train = config["train_csv_path"]
        self.df_val = config["val_csv_path"]
        self.df_test = config["test_csv_path"]

        # for SPECTRA
        self.spec_dir = config["spec_dir"]

        # for PHOTOMETRY
        self.photo_event_path = config["photo_event_path"]
        self.group_labels = config["group_labels"]

        self._split()

        if self.group_labels:
            ## create convenient mapping for label from str to int and from int to str
            ## group -> SN I, SN II, CV, AGN, TDE
            self.id2target = {0: "SN I", 1: "SN II", 2: "Cataclysmic", 3: "AGN", 4: "Tidal Disruption Event"}
            self.target2id = {
                "SN Ia": 0,
                "SN Ic": 0,
                "SN Ib": 0,
                "SN II": 1,
                "SN IIP": 1,
                "SN IIn": 1,
                "SN IIb": 1,
                "Cataclysmic": 2,
                "AGN": 3,
                "Tidal Disruption Event": 4,
            }

        else:
            raise ValueError("Class labels?")

        self.num_classes = len(self.target2id)

    def _split(self):
        """sort train, val,test based on df"""
        if self.split == "train":
            self.df = pd.read_csv(self.df_train)
        elif self.split == "val":
            self.df = pd.read_csv(self.df_val)
        elif self.split == "test":
            self.df = pd.read_csv(self.df_test)

        else:
            raise ValueError("Split must be either train, val, or test.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # obj_id, alert file from df
        obj_id, file = self.df["name"].iloc[index], self.df["file"].iloc[index]
        target = self.target2id[self.df["type"].iloc[index]]  #  get object type, make str label int

        file_path = os.path.join(self.preprocessed_path, file)  #  get object alert file name from df
        sample = np.load(file_path, allow_pickle=True).item()  #  load object alert file: .npy

        try:
            # ☆☆☆☆☆☆ Photometry ☆☆☆☆☆☆
            event_path = os.path.join(self.photo_event_path, self.split)
            photo_event = np.load(os.path.join(event_path, f"{obj_id}.npz"))
            photometry = photo_event["data"] if isinstance(photo_event, np.lib.npyio.NpzFile) else photo_event
            event_mjd = self.df["alert mjd"].iloc[index]

            # new cut to photo event
            photometry = photometry[photometry[:, 0] <= event_mjd]
            # photometry settings
            _BAND_OH = np.eye(3, dtype=np.float32)
            dt = np.log1p(photometry[:, 0])
            dt_prev = np.log1p(photometry[:, 1])
            logf, logfe = photometry[:, 3], photometry[:, 4]
            oh = _BAND_OH[photometry[:, 2].astype(np.int64)]
            vec4 = np.stack([dt, dt_prev, logf, logfe], 1)
            photo_tensor = torch.from_numpy(np.concatenate([vec4, oh], 1))

        except:
            raise ValueError(f"photo event, alert match error! at {file}")

        # ☆☆☆☆☆☆ Metadata ☆☆☆☆☆☆
        metadata = sample["metadata norm"]
        metadata_tensor = torch.tensor(metadata.values)
        metadata_tensor = metadata_tensor.to(torch.float32)

        # ☆☆☆☆☆☆ Images ☆☆☆☆☆☆
        image_tensor = sample["images norm, radial distance"]
        # needs to be converted
        image_tensor = image_tensor.to(torch.float32)

        # ☆☆☆☆☆☆ Spectra ☆☆☆☆☆☆
        obj_spec_file = os.path.join(self.spec_dir, f"{obj_id}.npy")
        flux = np.load(obj_spec_file, allow_pickle=True)
        spectra_tensor = torch.tensor(flux, dtype=torch.float32).unsqueeze(0)

        return photo_tensor, metadata_tensor, image_tensor, spectra_tensor, target
