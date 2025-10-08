from hyrax import Hyrax
from hyrax.models import hyrax_model
import numpy as np

import torch
import torch.nn as nn

from ..BaselineCLS import Time2Vec


@hyrax_model
class HyraxBaselineCLS(nn.Module):
    def __init__(self, config, shape):
        super().__init__()

        self.in_proj = nn.Linear(7, config['d_model'])
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, config['d_model']))

        self.time2vec = Time2Vec(config['d_model'])

        enc_layer = nn.TransformerEncoderLayer(
            config['d_model'], config['n_heads'], config['d_model'] * 4,
            config['dropout'], batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, config['n_layers'])
        self.norm = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['num_classes'])

        self.classification = True if config['mode'] == 'photo' else False
        if self.classification:
            self.fc = nn.Linear(config['d_model'], config['num_classes'])

        self.pad_mask = config['pad_mask']

    def forward(self, x):
        """
        x: (B, L, 7)  - the raw event tensor from build_event_tensor
            channels: [ dt, dt_prev, logf, logfe, one-hot-band(3) ]
        pad_mask: (B, L) boolean
        """
        B, L, _ = x.shape

        # project into model dim
        h = self.in_proj(x)                     # (B, L, d_model)
        # extract the *continuous* log1p dt feature
        t = x[..., 0]

        # compute the learned time embedding:
        te = self.time2vec(t)

        # add it:
        h = h + te                              # (B, L, d_model)

        # prepend a learned CLS token:
        tok = self.cls_tok.expand(B, -1, -1)      # (B,1,d_model)
        h = torch.cat([tok, h], dim=1)        # (B, L+1, d_model)

        # adjust padding mask to account for CLS at idx=0
        pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool),
             self.pad_mask], dim=1
        )

        # encode
        z = self.encoder(h, src_key_padding_mask=pad)  # (B, L+1, d_model)

        output = self.norm(z[:, 0])  # (B, d_model )

        if self.classification:
            # classification from the CLS token
            output = self.fc(output)  # (B, num_classes)     

        return output

    def train_step(self, batch):
        """
        This function contains the logic for a single training step. i.e. the
        contents of the inner loop of a ML training process.

        Parameters
        ----------
        batch : tuple
            A tuple containing the two values the loss function

        Returns
        -------
        Current loss value : dict
            Dictionary containing the loss value for the current batch.
        """
        data = batch[0]
        self.optimizer.zero_grad()

        decoded = self.forward(data)
        loss = self.criterion(decoded, data)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    @staticmethod
    def to_tensor(data_dict):
        """
        Converts raw data from a dictionary into a PyTorch tensor suitable for the model.

        Parameters
        ----------
        data_dict : dict
            A dictionary containing raw event data. Expected keys:
            - 'dt': time differences (1D array)
            - 'dt_prev': previous time differences (1D array)
            - 'logf': logarithm of flux (1D array)
            - 'logfe': logarithm of flux error (1D array)
            - 'band': band indices (1D array, integers)

        Returns
     -------
        torch.Tensor
            A tensor of shape (L, 7), where L is the sequence length.
        """
        # The following is all copilot generated
        dt = torch.log1p(torch.tensor(data_dict['dt'], dtype=torch.float32))  # Log1p of time differences
        dt_prev = torch.log1p(torch.tensor(data_dict['dt_prev'], dtype=torch.float32))  # Log1p of previous time differences
        logf = torch.tensor(data_dict['logf'], dtype=torch.float32)  # Logarithm of flux
        logfe = torch.tensor(data_dict['logfe'], dtype=torch.float32)  # Logarithm of flux error
        band = torch.tensor(data_dict['band'], dtype=torch.long)  # Band indices (categorical)

        # One-hot encode the band indices (3 bands assumed)
        band_oh = torch.nn.functional.one_hot(band, num_classes=3).to(torch.float32)

        # Concatenate all features along the last dimension
        return torch.cat([dt.unsqueeze(-1), dt_prev.unsqueeze(-1),
                        logf.unsqueeze(-1), logfe.unsqueeze(-1),
                        band_oh], dim=-1)

# FROM abrown3/AppleCiDEr_Skyportal/modes/photo/photo_t2vec_transformer_classifier.ipynb
# ===============================================================
# Imports & global knobs -------------------------------------
# ===============================================================
import os, math, time
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt, seaborn as sns

from pathlib import Path
from types   import SimpleNamespace
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from sklearn.preprocessing    import label_binarize
from sklearn.metrics          import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, fbeta_score,
    top_k_accuracy_score, roc_curve,
    precision_recall_curve, confusion_matrix,
    classification_report
)
from matplotlib.ticker        import AutoMinorLocator
from matplotlib               import cm as _cm
from collections import Counter

# -------- configuration  -----------------------------------------
CFG = SimpleNamespace(
    # data --------------------------------------------------------
    output_dir     = '/work/hdd/bcrv/ffontinelenunes/data/AppleCider/photo_events',
    stats_file     = 'feature_stats_day100.npz',
    horizon_days   = 50.0, # <- fine-tuning on 50 days
    batch_size     = 256, #64,
    sampler_balance= True,
    num_workers    = 8,
    # model -------------------------------------------------------
    d_model        = 128,
    n_heads        = 8,#4,
    n_layers       = 4,#2,
    dropout        = 0.30,
    max_len        = 257,#300,#256,#128,#128,
    # optimiser ---------------------------------------------------
    lr             = 5e-6,
    weight_decay   = 1e-2,
    # loss & imbalance -------------------------------------------
    focal_gamma    = 2.0,
    # augmentation -------horizon_days----------------------------------------
    cut_time_p     = None, #(.25,.25,.25,.25), #None,  # or (.25,.25,.25,.25)
    p_dropout      = 0.1,
    jitter_scale   = 0.10,
    flux_nu        = 8,    
    # training schedule ------------------------------------------
    epochs         = 150,
    patience       = 30,
    # misc --------------------------------------------------------
    seed           = 42
)

torch.manual_seed(CFG.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ===============================================================
#     Taxonomy  -------------------------------------------------
# ===============================================================
BROAD_CLASSES = ["SNI", "SNII", "Cataclysmic", "AGN", "TDE"]
ORIG2BROAD = {
    "SN Ia":"SNI","SN Ib":"SNI","SN Ic":"SNI",
    "SN II":"SNII","SN IIP":"SNII","SN IIn":"SNII","SN IIb":"SNII",
    "Cataclysmic":"Cataclysmic","AGN":"AGN","Tidal Disruption Event":"TDE",
}
NUM_CLASSES = len(BROAD_CLASSES)
BROAD2ID = {c:i for i,c in enumerate(BROAD_CLASSES)}
_SUBCLASS_ID2NAME = [
    "SN Ia","SN Ib","SN Ic","SN II","SN IIP","SN IIn","SN IIb",
    "Cataclysmic","AGN","Tidal Disruption Event"
]
ID2BROAD_ID = {i:BROAD2ID[ORIG2BROAD[name]] 
               for i,name in enumerate(_SUBCLASS_ID2NAME)}


# ===============================================================
#     Dataset / collate  ----------------------------------------
# ===============================================================
_BAND_OH = np.eye(3, dtype=np.float32)
def build_event_tensor(arr):                    
    dt  = np.log1p(arr[:,0]);  dt_prev = np.log1p(arr[:,1])
    logf, logfe = arr[:,3], arr[:,4]
    oh   = _BAND_OH[arr[:,2].astype(np.int64)]
    vec4 = np.stack([dt, dt_prev, logf, logfe], 1)
    return torch.from_numpy(np.concatenate([vec4, oh], 1))

class PhotoEventDataset(Dataset):
    def __init__(self, manifest_df, horizon=10.0):
        self.df, self.horizon = manifest_df.reset_index(drop=True), horizon
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw = np.load(row.filepath)
        arr = raw['data'] if isinstance(raw, np.lib.npyio.NpzFile) else raw
        if self.horizon:
            arr = arr[arr[:,0] <= self.horizon]
        return build_event_tensor(arr.astype(np.float32)), ID2BROAD_ID[int(row.label)]

def load_stats(path):
    st = np.load(path)
    return (torch.from_numpy(st['mean']),
            torch.from_numpy(st['std']))

def build_collate(mean, std):
    def collate(batch):
        seqs, labels = zip(*batch)
        lens = [s.size(0) for s in seqs]
        pad  = pad_sequence(seqs, batch_first=True)
        mask = torch.stack([
            torch.cat([torch.zeros(l), torch.ones(pad.size(1)-l)])
            for l in lens
        ]).bool()
        cont = (pad[...,:4] - mean) / (std + 1e-8)
        return torch.cat([cont, pad[...,4:]], -1), torch.tensor(labels), mask
    return collate