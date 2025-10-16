from hyrax import Hyrax
from hyrax.models import hyrax_model
import numpy as np

import torch
import torch.nn as nn

from ..BaselineCLS import Time2Vec


@hyrax_model
class HyraxBaselineCLS(nn.Module):
    def __init__(self, config, data_sample=None):
        super().__init__()

        self.config = config

        model_config = config["model"]["BaselineCLS"]

        self.in_proj = nn.Linear(7, model_config['d_model'])
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, model_config['d_model']))

        self.time2vec = Time2Vec(model_config['d_model'])

        enc_layer = nn.TransformerEncoderLayer(
            model_config['d_model'], model_config['n_heads'], model_config['d_model'] * 4,
            model_config['dropout'], batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, model_config['n_layers'])
        self.norm = nn.LayerNorm(model_config['d_model'])
        self.head = nn.Linear(model_config['d_model'], model_config['num_classes'])

        self.classification = True if model_config['mode'] == 'photo' else False
        if self.classification:
            self.fc = nn.Linear(model_config['d_model'], model_config['num_classes'])

        self.pad_mask = model_config['pad_mask']

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
        # Assuming reading in a data dictionary from an alert npy file
        data_dict = data_dict["data"]["photometry"]
        dt = data_dict["dt"][:, None]        # (L, 1)
        dt_prev = data_dict["dt_prev"][:, None]     # (L, 1)
        logf = data_dict["logf"][:, None]        # (L, 1)
        logfe = data_dict["logfe"][:, None]      # (L, 1)
        band = data_dict["band"][:, None]                   # (L,)
        
        vec4 = np.stack([dt, dt_prev, logf, logfe], 1)

        one_hot_encoding = np.eye(3, dtype=np.float32)
        one_hot_band = one_hot_encoding[band.astype(np.int64)]  # (L, 3)
        #print(vec4, vec4.shape)
        #return vec4
        return torch.from_numpy(vec4)
        #return torch.from_numpy(np.concatenate([vec4, one_hot_band], 1))  # (L, 7)