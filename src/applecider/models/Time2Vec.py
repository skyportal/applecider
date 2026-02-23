"""Time2Vec: map scalar time t -> d_model-dimensional vector.
Authored by Felipe Fontinele Nunes (2025)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda")


def load_stats(path):
    st = np.load(path)
    return (torch.from_numpy(st["mean"]), torch.from_numpy(st["std"]))


def collate(batch):
    photo_seqs, metadata, images, spectra, labels = zip(*batch)

    lens = [s.size(0) for s in photo_seqs]
    pad_seq = pad_sequence(photo_seqs, batch_first=True)
    photo_mask = torch.stack(
        [torch.cat([torch.zeros(l), torch.ones(pad_seq.size(1) - l)]) for l in lens]
    ).bool()

    # yeah yeah i know 🤡 no hardcoding
    mean, std = load_stats(
        "/work/hdd/bcrv/ffontinelenunes/data/AppleCider/photo_events/feature_stats_day100.npz"
    )

    photo_cont = (pad_seq[..., :4] - mean) / (std + 1e-8)

    metadata = torch.stack(metadata)
    spectra = torch.stack(spectra)
    images = torch.stack(images)

    return (
        torch.cat([photo_cont, pad_seq[..., 4:]], -1).to(device),
        photo_mask.to(device),
        metadata,
        images,
        spectra.to(device),
        torch.tensor(labels).to(device),
    )


class Time2Vec(nn.Module):
    """
    Time2Vec: map scalar time t -> d_model-dimensional vector.
    v0 = w0 * t + b0  (linear)
    v[i] = sin(w[i] * t + b[i])  for i=1..d_model-1
    """

    def __init__(self, d_model):
        super().__init__()
        # one linear + (d_model-1) periodic features
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(d_model - 1))
        self.b = nn.Parameter(torch.zeros(d_model - 1))

    def forward(self, t):
        """
        t: (B, L)  - scalar "time since first detection" per event
        returns (B, L, d_model)
        """
        # linear term
        v0 = self.w0 * t + self.b0  # (B, L)
        # periodic terms
        vp = torch.sin(t.unsqueeze(-1) * self.w + self.b)  # (B, L, d_model-1)
        return torch.cat([v0.unsqueeze(-1), vp], dim=-1)  # (B, L, d_model)


# ------------------------------------------------------------------------------


# (NEW) Transformer Encoder + class token
# ------------------------------------------------------------------------------
class BaselineCLS(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, num_classes, dropout, max_len=None):
        super().__init__()
        self.in_proj = nn.Linear(7, d_model)
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, d_model))

        # replace SinCos PE with Time2Vec on the dt channel

        self.time2vec = Time2Vec(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, pad_mask):
        """
        x: (B, L, 7)  - the raw event tensor from build_event_tensor
            channels: [ dt, dt_prev, logf, logfe, one-hot-band(3) ]
        pad_mask: (B, L) boolean
        """
        B, L, _ = x.shape

        # project into model dim
        h = self.in_proj(x)  # (B, L, d_model)
        # extract the *continuous* log1p dt feature
        t = x[..., 0]  # (B, L)

        # compute the learned time embedding:
        te = self.time2vec(t)  # (B, L, d_model)

        # add it:
        h = h + te  # (B, L, d_model)

        # prepend a learned CLS token:
        tok = self.cls_tok.expand(B, -1, -1)  # (B,1,d_model)
        h = torch.cat([tok, h], dim=1)  # (B, L+1, d_model)

        # adjust padding mask to account for CLS at idx=0
        pad = torch.cat([torch.zeros(B, 1, device=pad_mask.device, dtype=torch.bool), pad_mask], dim=1)

        # encode
        z = self.encoder(h, src_key_padding_mask=pad)  # (B, L+1, d_model)

        # classification from the CLS token
        return self.head(self.norm(z[:, 0]))  # (B, num_classes)


# --- MPT heads ----------------------------------------------------------------
class MPTModel(torch.nn.Module):
    def __init__(self, base_enc):
        super().__init__()
        self.encoder = base_enc.encoder
        d = base_enc.in_proj.out_features
        self.head_flux = torch.nn.Linear(d, 1)
        self.head_band = torch.nn.Linear(d, 3)
        self.head_dt = torch.nn.Linear(d, 1)

    def forward(self, z):
        return (
            self.head_flux(z),  # (B, L, 1)
            self.head_band(z),  # (B, L, 3)
            self.head_dt(z),  # (B, L, 1)
        )
