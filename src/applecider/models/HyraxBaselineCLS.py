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
        self.criterion = FocalLoss()

        model_config = config["model"]["BaselineCLS"]
        #self.optimizer = torch.optim.AdamW(self.parameters(), lr=model_config["lr"], weight_decay=model_config["weight_decay"])
        # Use AdamW eventually


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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        """
        x: (B, L, 7)  - the raw event tensor from build_event_tensor
            channels: [ dt, dt_prev, logf, logfe, one-hot-band(3) ]
        pad_mask: (B, L) boolean
        """
        data, pad, _ = x

        B, L, _ = data.shape

        # project into model dim
        h = self.in_proj(data)                     # (B, L, d_model)
        # extract the *continuous* log1p dt feature
        t = data[..., 0]

        # compute the learned time embedding:
        # should be training time2vec, we should double check
        te = self.time2vec(t)

        # add it:
        hte = h + te                              # (B, L, d_model)
        # prepend a learned CLS token:
        tok = self.cls_tok.expand(B, -1, -1)      # (B,1,d_model)
        hte = torch.cat([tok, hte], dim=1)        # (B, L+1, d_model)

        pad_extended = F.pad(pad, (1, 0), value=False)

        # encode
        # In inference this is a convergence point
        z = self.encoder(hte, src_key_padding_mask=pad_extended)  # (B, L+1, d_model)
        output = self.norm(z[:, 0])  # (B, d_model )

        if self.classification:
            # classification from the CLS token
            output = self.fc(output)  # (B, num_classes)
        if self.config["model"]["HyraxBaselineCLS"]["use_probabilities"]:
            output = F.softmax(output, dim=1)
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

        _, _, labels = batch

        decoded = self.forward(batch)
        loss = self.criterion(decoded, labels)
        self.optimizer.zero_grad()
        loss.backward()
        #gradient clipping
        # TODO: make this a config option, potentially a general
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()
        numpy_logits = decoded.detach().cpu().numpy()


        # Additional metrics
        # We wanted epoch-level metrics to assess, but hyrax potentially only allows batch-level metrics here (ask Drew)
        # accuracy, total loss/tot n, any custom metrics
        return {"loss": loss.item(), "num_tdes": np.sum([labels.cpu().numpy() == 4])}

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

        # NOTE: Hyrax will copy this method into a standalone module during
        # training so that it can be used for inference. However, Hyrax cannot
        # copy imports at the top of the file. Since we depend on numpy in this
        # method, we'll import it here to make sure it is present for inference.
        import numpy as np

        if "data" not in data_dict:
            raise ValueError("Data dictionary must contain 'data' key.")

        data = data_dict["data"]
        photo_tensor = data["photometry"]
        label_tensor = np.asarray(data.get("label", []), dtype=np.int64)

        # Use mean and std from data_dict to normalize continuous features
        photo_tensor[..., :4] = (photo_tensor[..., :4] - data["mean"]) / (data["std"] + 1e-8)

        if "pad_mask" in data.keys():
            mask_tensor = data["pad_mask"]
            return (photo_tensor, mask_tensor, label_tensor)

        # Generate all-false padding mask if not provided, useful for infer step
        # The +1 is to account for the CLS token added in the model.
        false_mask = np.zeros((photo_tensor.shape[0], photo_tensor.shape[1]+1), dtype=bool)
        return (photo_tensor, false_mask, label_tensor)

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