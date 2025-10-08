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