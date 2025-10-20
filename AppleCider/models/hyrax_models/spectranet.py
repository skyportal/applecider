import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyrax.models import hyrax_model

class SpectraNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes,
                use_ln=True, do_pool=False):
        super().__init__()
        self.do_pool = do_pool
        self.use_ln = use_ln
        self.k = len(kernel_sizes)
        norm_channels = out_channels * self.k

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.norm = (
            nn.LayerNorm(norm_channels)
            if use_ln else nn.BatchNorm1d(norm_channels)
        )

        if do_pool:
            self.total_pooled_channels = norm_channels
            self.downsample = nn.Conv1d(norm_channels, out_channels, kernel_size=1)
            self.pool_max = nn.MaxPool1d(4)
    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.convs], dim=1)

        if self.use_ln:
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            x = self.norm(x)

        x = F.gelu(x)

        if self.do_pool:
            x = self.downsample(x)
            x = self.pool_max(x)
        return x

def make_stage(in_c, out_c, depth, kernel_sizes, use_ln=True, do_pool=True):
    k = len(kernel_sizes)
    blocks = []
    for i in range(depth):
        blocks.append(SpectraNetBlock(
            in_channels=in_c if i == 0 else out_c * k,
            out_channels=out_c,
            kernel_sizes=kernel_sizes,
            use_ln=use_ln,
            do_pool=(do_pool if i == depth - 1 else False),
        ))
    return nn.Sequential(*blocks), k

@hyrax_model
class SpectraNet(nn.Module):
    def __init__(self, config=None, data_sample=None):
        super().__init__()
        
        self.config = config

        self.redshift = config["model"]["SpectraNet"]["redshift"]

        #! Should this be hard coded?
        self.kernel_sizes_per_stage = [
            [3, 61, 1021],
            [3, 31, 251],
            [3, 15, 61],
            [3, 11, 31],
            [3, 7, 13]
        ]

        self.depths = config["model"]["SpectraNet"]["depths"]

        if len(self.depths) != len(self.kernel_sizes_per_stage):
            raise ValueError("Length of depths must match number of stages")

        #! Should this be part of config ?
        use_ln_stages = [True, True, True, True, True]
        channels = [1, 64, 128, 256, 512, 1024]

        #! This could be a for loop probably
        self.stage1, k1 = make_stage(channels[0], channels[1], self.depths[0], self.kernel_sizes_per_stage[0], use_ln=use_ln_stages[0])
        self.stage2, k2 = make_stage(channels[1], channels[2], self.depths[1], self.kernel_sizes_per_stage[1], use_ln=use_ln_stages[1])
        self.stage3, k3 = make_stage(channels[2], channels[3], self.depths[2], self.kernel_sizes_per_stage[2], use_ln=use_ln_stages[2])
        self.stage4, k4 = make_stage(channels[3], channels[4], self.depths[3], self.kernel_sizes_per_stage[3], use_ln=use_ln_stages[3])
        self.stage5, k5 = make_stage(channels[4], channels[5], self.depths[4], self.kernel_sizes_per_stage[4], use_ln=use_ln_stages[4], do_pool=False)
        self.ks = [k1, k2, k3, k4, k5]

        #! This has a hard coded calculation in the other model
        #! Should be part of config ?
        length = 16
        self.flat_dim = 3072

        if self.redshift:
            self.regressor = nn.Sequential(
                nn.Linear(self.flat_dim, 384),
                nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.5),
                nn.Linear(384, 1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.flat_dim, 384),
                nn.LayerNorm(384), nn.GELU(), nn.Dropout(0.5),
                nn.Linear(384, config["model"]["SpectraNet"]["class_order"])
            )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)  # [B, C, L]

        x_max = F.adaptive_max_pool1d(x,1).squeeze(-1)  # [B, C]

        x_fused = torch.cat([x_max], dim=1)  # [B, 3C]

        if self.redshift:
            raise NotImplementedError("not implemented yet.")
            # return self.regressor(x_fused).squeeze(1)
        else:
            return self.classifier(x_fused)

    def train_step(self, batch):
        """Simple placeholder train step for Hyrax integration
        based on my kbmod-ml version lol
        """
        # spectranet uses the `train_one_epoch` function in utils.py
        inputs, labels = batch

        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    @staticmethod
    def to_tensor(data_dict):

        return (torch.tensor(data_dict["data"]["flux"]), torch.tensor(data_dict["data"]["label"]))



