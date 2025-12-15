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

        if self.config["model"]["HyraxBaselineCLS"]["pretrained_weights_path"]:
            pretrained_path = self.config["model"]["HyraxBaselineCLS"]["pretrained_weights_path"]
            state_dict = torch.load(pretrained_path)
            self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")

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
        # should be training time2vec, we should double check
        te = self.time2vec(t)

        # add it:
        hte = h + te                              # (B, L, d_model)
        # prepend a learned CLS token:
        tok = self.cls_tok.expand(B, -1, -1)      # (B,1,d_model)
        hte = torch.cat([tok, hte], dim=1)        # (B, L+1, d_model)

        # encode
        # In inference this is a convergence point
        z = self.encoder(hte, src_key_padding_mask=pad)  # (B, L+1, d_model)

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

        labels = batch[1]

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

@hyrax_model
class MPTModel(nn.Module):
    def __init__(self, config, data_sample=None):
        super().__init__()

        self.config = config

        # Pre-trains for BaselineCLS, use the same config parameters and architecture
        model_config = config["model"]["BaselineCLS"]

        enc_layer = nn.TransformerEncoderLayer(
            model_config['d_model'], model_config['n_heads'], model_config['d_model'] * 4,
            model_config['dropout'], batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, model_config['n_layers'])

        self.in_proj = nn.Linear(7, model_config['d_model'])
        self.cls_tok = nn.Parameter(torch.zeros(1, 1, model_config['d_model']))
        self.time2vec = Time2Vec(model_config['d_model'])

        #self.encoder    = base_enc.encoder
        d = self.in_proj.out_features
        self.head_flux = nn.Linear(d, 1)
        self.head_band = nn.Linear(d, 3)
        self.head_dt = nn.Linear(d, 1)

        # TODO: Resolve optimizer setup
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        #opt = torch.optim.AdamW([
        #{'params': enc.encoder.parameters(), 'lr': PT_LR*0.5},
        #{'params': list(mpt.head_flux.parameters())+list(mpt.head_band.parameters())+list(mpt.head_dt.parameters()),
        # 'lr': PT_LR},
        #], weight_decay=1e-2)

    def forward(self, z):
        return self.head_flux(z), self.head_band(z), self.head_dt(z)

    # TODO: This entire train step is roughly sketched in
    def train_step(self, batch):
        data = batch[0]
        pad = batch[2]
        masked_tok = self._mask_batch(data, pad)

        B, L, _ = data.shape

        # project into model dim
        emb = self.in_proj(data)                     # (B, L, d_model)
        # extract the *continuous* log1p dt feature
        t = data[..., 0]

        # compute the learned time embedding:
        te = self.time2vec(t)
        te = F.dropout(te, p=self.config["model"]["BaselineCLS"]["dropout"])

        # add it:
        h_in = emb + te                              # (B, L, d_model)
        # prepend a learned CLS token:
        tok = self.cls_tok.expand(B, -1, -1)      # (B,1,d_model)
        h = torch.cat([tok, h_in], dim=1)        # (B, L+1, d_model)

        # encode
        z_full = self.encoder(h, src_key_padding_mask=pad)  # (B, L+1, d_model)
        h_masked = z_full[:, 1:, :]  # (B, L, d_model)
        f_hat = self.head_flux(h_masked)  # (B, L, 1)
        b_hat = self.head_band(h_masked)  # (B, L, 3)
        dt_hat = self.head_dt(h_masked)   # (B, L, 1)

        #mf = masked_tok.view(-1)
        # Why am I off by one index here? Suggests i'm including the CLS token somehow
        mf = masked_tok[:, 1:].contiguous().view(-1)

        true_f = data[..., 2].view(-1)
        loss_f = F.mse_loss(f_hat.view(-1)[mf], true_f[mf])
        true_b = data[..., 4:7].argmax(-1).view(-1)
        loss_b = F.cross_entropy(b_hat.view(-1, 3)[mf], true_b[mf])
        dt_gt = torch.roll(data[..., 1], -1, dims=1)
        dt_gt[:, -1] = 0.0
        dt_gt = dt_gt.view(-1)
        loss_dt = F.mse_loss(dt_hat[..., 0].view(-1)[mf], dt_gt[mf])

        lambda_f = self.config["model"]["HyraxBaselineCLS"]["lambda_f"]
        lambda_b = self.config["model"]["HyraxBaselineCLS"]["lambda_b"]
        lambda_dt = self.config["model"]["HyraxBaselineCLS"]["lambda_dt"]
        loss = lambda_f * loss_f * lambda_b * loss_b * lambda_dt * loss_dt
        #loss = 5.0 * loss_f + 3.0 * loss_b + 5.0 * loss_dt
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    def _mask_batch(self, x, pad_mask):
        MASK_P = self.config["model"]["HyraxBaselineCLS"]["mask_p"]
        masked = torch.zeros_like(pad_mask)
        B, L, _ = x.shape
        for b in range(B):
            valid = (~pad_mask[b]).nonzero(as_tuple=True)[0]
            k = max(int(len(valid)*MASK_P), 3)
            num_each = k // 3; extras = k - 3*num_each
            bands    = x[b, :, 4:7].argmax(-1)
            idxs     = []
            for band in [0,1,2]:
                valid_b = valid[bands[valid]==band]
                if len(valid_b)>0:
                    take = min(len(valid_b), num_each)
                    perm = torch.randperm(len(valid_b))[:take]
                    idxs.append(valid_b[perm])
            if extras>0:
                remaining = torch.cat(idxs) if len(idxs)>0 else torch.tensor([], device=valid.device, dtype=valid.dtype)
                pool = valid[~torch.isin(valid, remaining)]
                if len(pool)>0:
                    perm = torch.randperm(len(pool))[:extras]
                    idxs.append(pool[perm])
            idx = torch.cat(idxs) if len(idxs)>0 else torch.tensor([], device=valid.device, dtype=valid.dtype)
            if len(idx)>0:
                x[b, idx, 2:7] = 0.0
                masked[b, idx] = True
        return masked

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