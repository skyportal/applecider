from torch.utils.data import Dataset
from hyrax.data_sets import HyraxDataset
from pathlib import Path
from typing import Union
import numpy as np


class PhotoEventsDataset(HyraxDataset, Dataset):
    def __init__(self, config: dict, data_location: Union[Path, str] = None):
        self.filenames = sorted(list(Path(data_location).glob('*.npz')))

        # Do we need to define test vs train data?

        super().__init__(config)

    def __getitem__(self, idx):
        # load the data from disk
        return self.get_photometry(idx)

    def get_photometry(self, idx):
        data = np.load(self.filenames[idx], allow_pickle=True)

        # Build a dictionary of the expected photometry fields
        photo_dict = {}
        photo_dict["dt"] = data["data"].T[0]
        photo_dict["dt_prev"] = data["data"].T[1]
        photo_dict["logf"] = data["data"].T[3]
        photo_dict["logfe"] = data["data"].T[4]
        photo_dict["band"] = data["data"].T[2].astype(int)
        #print(photo_dict)
        print(idx)
        return photo_dict

    def __len__(self):
        return len(self.filenames)
