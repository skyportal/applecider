import os

import numpy as np
import torch
from hyrax.data_sets.data_set_registry import HyraxDataset
from torch.utils.data import Dataset


class SpectraData(HyraxDataset, Dataset):
    def __init__(self, config, data_location=None):
        super().__init__(config)

        data_table = torch.load("/Users/maxwest/data/train.pt")

        self.label_strings = {
            'AGN' : 0,
            'Cataclysmic' : 1,
            'SN IIP' : 2,
            'SN IIb': 3,
            'SN IIn' : 4,
            'SN Ia' : 5,
            'SN Ib' : 6,
            'SN Ic' : 7,
            'Tidal Disruption Event' : 8
        }

        self._data = data_table["flux"]
        self._labels = data_table["labels"]
        self._label_idx = [self.label_strings[l] for l in self._labels]
        self._redshifts = data_table["redshifts"]
        self._file_paths = data_table["file_paths"]

        metadata_table = self._read_metadata()
        self.table = metadata_table
        super().__init__(config, metadata_table)

    def ids(self):
        """Return the ids of the data set"""
        return np.arange(len(self._data))

    def shape(self):
        """data shape, including currently enabled columns"""
        cols = len(self.active_columns)
        width, height = self._data[0][0].shape

        return (cols, width, height)

    def get_flux(self, idx):
        flux = self.table["flux"][idx]
        flux = np.expand_dims(flux, 0)
        return flux

    def get_label(self, idx):
        label = self.table["label"][idx]
        return label

    def get_redshift(self, idx):
        redshift = self.table["redshift"][idx]
        return redshift

    def _read_metadata(self):
        """This is a pretend implementation so we don't use the path passed, which you might use
        to find your .csv/.fits/.tsv catalog file and call astropy's Table.read().

        We simply construct a table from our mock data"""
        from astropy.table import Table

        global ras, decs, filenames
        return Table({
            "flux": [f for f in self._data],
            "object_id": self.ids(),
            "label": self._label_idx,
            "redshift": self._redshifts,
            "file_path": self._file_paths,
        })

    def __len__(self):
        return len(self._data)