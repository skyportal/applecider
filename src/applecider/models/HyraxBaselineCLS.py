from hyrax import Hyrax
from hyrax.models import hyrax_model
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from applecider.models.Time2Vec import Time2Vec


@hyrax_model
class HyraxBaselineCLS(nn.Module):
    def __init__(self, config, data_sample=None):
        super().__init__()

        self.config = config
        self._criterion = FocalLoss

        model_config = config["model"]["BaselineCLS"]
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=model_config["lr"], weight_decay=model_config["weight_decay"])
        # Use AdamW eventually
        #self.optimizer = torch.optim.Adam()

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

    def forward(self, x, pad=None):
        """
        x: (B, L, 7)  - the raw event tensor from build_event_tensor
            channels: [ dt, dt_prev, logf, logfe, one-hot-band(3) ]
        pad_mask: (B, L) boolean
        """
        data = x[0]
        pad = x[2]

        B, L, _ = data.shape

        # project into model dim
        h = self.in_proj(data)                     # (B, L, d_model)
        # extract the *continuous* log1p dt feature
        t = data[..., 0]

        # compute the learned time embedding:
        te = self.time2vec(t)

        # add it:
        hte = h + te                              # (B, L, d_model)
        # prepend a learned CLS token:
        tok = self.cls_tok.expand(B, -1, -1)      # (B,1,d_model)
        hte = torch.cat([tok, hte], dim=1)        # (B, L+1, d_model)

        # encode
        z = self.encoder(hte, src_key_padding_mask=pad)  # (B, L+1, d_model)

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

        labels = batch[1]
        self.optimizer.zero_grad()

        decoded = self.forward(batch)
        loss = self.criterion(decoded, labels)
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
        photo_tensor, label_tensor = data_dict["data"]["photometry"], data_dict["data"]["label"]
        # Use mean and std from data_dict to normalize continuous features
        photo_tensor[..., :4] = (photo_tensor[..., :4] - data_dict["data"]["mean"]) / (data_dict["data"]["std"] + 1e-8)

        if "pad_mask" in data_dict["data"].keys():
            mask_tensor = data_dict["data"]["pad_mask"]
            return (photo_tensor, label_tensor, mask_tensor)

        # Generate all-false padding mask if not provided, useful for infer step
        false_mask = torch.zeros(photo_tensor.size(0), photo_tensor.size(1), dtype=torch.bool)
        return (photo_tensor, label_tensor, false_mask)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, eps: float = 0, reduction: str = 'mean'): #eps=0.1
        super().__init__()

        self.gamma, self.alpha, self.eps, self.reduction = gamma, alpha, eps, reduction
    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        B, C = logits.shape
        logp = F.log_softmax(logits, dim=1); p = logp.exp()
        if self.eps > 0:
            smooth = torch.full_like(logp, fill_value=self.eps/(C-1))
            smooth.scatter_(1, target.unsqueeze(1), 1.0-self.eps)
            y = smooth
        else:
            y = F.one_hot(target, num_classes=C).float()
        focal_weight = (1.0 - p).pow(self.gamma)
        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha.view(1, C)
        loss = -(y * focal_weight * logp).sum(dim=1)
        return loss.mean() if self.reduction=='mean' else loss.sum()