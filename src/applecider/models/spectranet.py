import torch
import torch.nn as nn
import torch.nn.functional as F

from hyrax.models import hyrax_model


class SpectraNetBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_sizes, use_ln=True, do_pool=False
    ):
        """Basic layer block for SpectraNet. Typically generated from parameters
        through the `make_stage` function and as part of the `SpectraNet` initialization.
        """
        super().__init__()
        self.do_pool = do_pool
        self.use_ln = use_ln
        self.k = len(kernel_sizes)
        norm_channels = out_channels * self.k

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.norm = (
            nn.LayerNorm(norm_channels) if use_ln else nn.BatchNorm1d(norm_channels)
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


def make_stage(in_channel, out_channel, depth, kernel_sizes, use_ln=True, do_pool=True):
    """Function to create one layer of the SpectraNet model.

    Parameters
    ----------
    in_channel : `int`
        the number of input channels.
    out_channel : `int`
        the number of output channels.
    depth : `int`
        depth of the layer, i.e. how many
        times it's repeated sequentially.
    kernel_sizes : `list` of `list` of `int`s
        the kernel sizes convolution.
    use_ln : `bool`
        whether to use layer normalization for
        this layer. Default `True`.
    do_pool : `bool`
        whether to do pooling with this layer.
        typically disabled only for the final layer.
        Default `True`.

    Returns
    -------
    An `nn.Sequential` of `SpectraNetBlock`s and list of the number of kernels per layer.
    """
    k = len(kernel_sizes)
    blocks = []
    for i in range(depth):
        blocks.append(
            SpectraNetBlock(
                in_channels=in_channel if i == 0 else out_channel * k,
                out_channels=out_channel,
                kernel_sizes=kernel_sizes,
                use_ln=use_ln,
                do_pool=(do_pool if i == depth - 1 else False),
            )
        )
    return nn.Sequential(*blocks), k


@hyrax_model
class SpectraNet(nn.Module):
    """Model for training spectra data.
    Can be configured to output either a transient
    classification or give a redshift score prediction
    (see the "AppleCider/default_config.toml" file).
    """

    def __init__(self, config=None, data_sample=None):
        super().__init__()

        self.config = config

        spectranet_config = config["model"]["SpectraNet"]

        self.redshift = spectranet_config["redshift"]
        kernel_sizes_per_stage = spectranet_config["kernel_sizes_per_stage"]
        depths = spectranet_config["depths"]
        use_ln_stages = spectranet_config["use_ln_stages"]
        channels = spectranet_config["channels"]
        flat_dim = spectranet_config["flat_dim"]
        class_order = spectranet_config["class_order"]

        if (
            len(depths)
            != len(use_ln_stages)
            != len(channels)
            != len(kernel_sizes_per_stage)
        ):
            raise ValueError(
                "depths, use_ln_stages, channels, and kernel_sizes_per_stage must be the same length."
            )

        # create each layer
        self.stages = []
        self.ks = []
        for i in range(len(depths)):
            in_channel = 1 if i == 0 else channels[i - 1]
            out_channel = channels[i]
            depth = depths[i]
            kernel_size = kernel_sizes_per_stage[i]
            use_ln_stage = use_ln_stages[i]
            do_pool = False if i == len(depths) - 1 else True

            stage, k = make_stage(
                in_channel=in_channel,
                out_channel=out_channel,
                depth=depth,
                kernel_sizes=kernel_size,
                use_ln=use_ln_stage,
                do_pool=do_pool,
            )

            self.stages.append(stage)
            self.ks.append(k)

        # create the sequence for all layers
        self.all_stages = nn.Sequential(*self.stages)

        if self.redshift:
            self.regressor = nn.Sequential(
                nn.Linear(flat_dim, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(384, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(flat_dim, 384),
                nn.LayerNorm(384),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(384, class_order),
            )

    def forward(self, batch):
        # batch is a tuple of (data, labels, redshifts). When training, labels
        # and redshifts are present, during inference they are empty tensors.
        x, _, _ = batch

        x = self.all_stages(x)

        x_max = F.adaptive_max_pool1d(x, 1).squeeze(-1)  # [B, C]

        x_fused = torch.cat([x_max], dim=1)  # [B, 3C]

        if self.redshift:
            return self.regressor(x_fused).squeeze(1)
        else:
            return self.classifier(x_fused)

    def train_step(self, batch):
        # spectranet uses the `train_one_epoch` function in utils.py
        _, labels, redshifts = batch

        self.optimizer.zero_grad()
        outputs = self(batch)
        if self.redshift:
            loss = self.criterion(outputs, redshifts)
        else:
            loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    @staticmethod
    def to_tensor(data_dict):
        """This method will receive a dictionary of data and should convert it
        to the relevant numpy arrays needed for either training or inference."""

        # NOTE: Hyrax will copy this method into a standalone module during
        # training so that it can be used for inference. However, Hyrax cannot
        # copy imports at the top of the file. Since we depend on numpy in this
        # method, we'll import it here to make sure it is present for inference.
        import numpy as np

        if "data" not in data_dict:
            raise ValueError("Data dictionary must have a 'data' key.")

        data = data_dict["data"]
        return (
            np.asarray(data.get("flux", []), dtype=np.float32),
            # Return empty arrays even if the labels/redshifts are not present
            np.asarray(data.get("label", []), dtype=np.int16),
            np.asarray(data.get("redshift", []), dtype=np.float32),
        )
