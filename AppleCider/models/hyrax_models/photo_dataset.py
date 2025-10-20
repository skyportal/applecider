from torch.utils.data import Dataset
from hyrax.data_sets import HyraxDataset
from pathlib import Path
from typing import Union
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch

# =========================
# Global Taxonomy
# =========================
BROAD_CLASSES = ["SNI", "SNII", "CV", "AGN", "TDE"]
ORIG2BROAD = {
    "SN Ia":"SNI","SN Ib":"SNI","SN Ic":"SNI",
    "SN II":"SNII","SN IIP":"SNII","SN IIn":"SNII","SN IIb":"SNII",
    "Cataclysmic":"CV","AGN":"AGN","Tidal Disruption Event":"TDE",
}
NUM_CLASSES = len(BROAD_CLASSES)
BROAD2ID = {c:i for i,c in enumerate(BROAD_CLASSES)}
_SUBCLASS_ID2NAME = [
    "SN Ia","SN Ib","SN Ic","SN II","SN IIP","SN IIn","SN IIb",
    "Cataclysmic","AGN","Tidal Disruption Event"
]
ID2BROAD_ID = {i:BROAD2ID[ORIG2BROAD[name]] for i,name in enumerate(_SUBCLASS_ID2NAME)}


class PhotoEventsDataset(HyraxDataset, Dataset):
    def __init__(self, config: dict, data_location: Union[Path, str] = None):
        self.filenames = sorted(list(Path(data_location).glob('*.npz')))

        # Do we need to define test vs train data?

        super().__init__(config)

    def __getitem__(self, idx):
        # load the data from disk
        return self.get_photometry(idx)
    
    def get_label(self, idx):
        #row = self.df.iloc[idx]
        #raw = np.load(row.filepath)
        #arr = raw['data'] if isinstance(raw, np.lib.npyio.NpzFile) else raw
        #if self.horizon:
        #    arr = arr[arr[:,0] <= self.horizon]
        #return ID2BROAD_ID[int(row.label)]
        return ID2BROAD_ID[0] # Placeholder for testing

    def get_photometry(self, idx):
        data = np.load(self.filenames[idx], allow_pickle=True)["data"]
        # truncate data to two observations for testing
        #data = data[:2,:]

        dt = data[:, 0]
        dt_prev = data[:, 1]
        logf = data[:, 3]
        logfe = data[:, 4]
        band = data[:, 2]

        vec4 = np.stack([dt, dt_prev, logf, logfe], 1)
        one_hot_encoding = np.eye(3, dtype=np.float32)
        one_hot_band = one_hot_encoding[band.astype(np.int64)]  # (L, 3)
        return torch.from_numpy(np.concatenate([vec4, one_hot_band], 1))  # (L, 7)

    def __len__(self):
        return len(self.filenames)


def collate(batch):
    #import pdb;pdb.set_trace()
    mean = 0
    std = 1
    #import pdb;pdb.set_trace()
    seqs = []
    labels = []
    for i in batch:
        #import pdb; pdb.set_trace()
        seqs += [i["data"]["photometry"]]
        labels += [i["data"]["label"]]  

    #seqs, labels = zip(*batch)
    lens = [s.size(0) for s in seqs]
    pad  = pad_sequence(seqs, batch_first=True)              # (B, L, 7)
    mask = torch.stack([torch.cat([torch.zeros(l), torch.ones(pad.size(1)-l)]) for l in lens]).bool()
    cont = (pad[..., :4] - mean) / (std + 1e-8)
    pad_mask = torch.cat(
            [torch.zeros(len(batch), 1, device=mask.device, dtype=torch.bool),
             mask], dim=1
        )
    return {"data": {"photometry": torch.cat([cont, pad[..., 4:]], -1), "label": torch.tensor(labels), "pad_mask": torch.tensor(pad_mask)}}
    #return torch.cat([cont, pad[..., 4:]], -1), torch.tensor(labels), mask  # x, y, pad_mask
