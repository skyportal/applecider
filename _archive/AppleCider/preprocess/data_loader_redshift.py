"""SpectraNet model definition.
Authored by Maojie Xu, Argyro Sasli, and Alexandra Junell (2025)
"""
import numpy as np
import torch
from torch.utils.data import DataLoader


class SpectraPTDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path, redshift_key="redshift_errors", apply_mask=False):
        data = torch.load(pt_path)
        self.flux = data["flux"]  # shape: [N, 1, 4096]
        self.labels = data[redshift_key]  # shape: [N]
        self.apply_mask = apply_mask

        assert self.flux.shape[0] == len(self.labels), "Mismatch in flux and redshift length"

    def __len__(self):
        return self.flux.shape[0]

    def __getitem__(self, idx):
        x = self.flux[idx]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, 4096]

        if self.apply_mask:
            x_masked, _ = self._apply_random_mask_with_mask(x.clone())
        else:
            x_masked = x.clone()

        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x_masked, y

    def _apply_random_mask_with_mask(self, x):
        """
        Randomly zero out left and right sides (up to 25% each).
        """
        seq_len = x.shape[-1]
        mask = torch.ones_like(x, dtype=torch.bool)

        max_crop = seq_len // 4
        left_cut = np.random.randint(0, max_crop + 1)
        right_cut = seq_len - np.random.randint(0, max_crop + 1)

        x[:, :left_cut] = 0.0
        x[:, right_cut:] = 0.0
        mask[:, :left_cut] = 0
        mask[:, right_cut:] = 0

        return x, mask


def create_data_loaders(config):
    train_dataset = SpectraPTDataset(
        pt_path=config["train_dir"],
        redshift_key="redshift_errors",
        apply_mask=True,
    )
    val_dataset = SpectraPTDataset(
        pt_path=config["val_dir"],
        redshift_key="redshift_errors",
        apply_mask=False,
    )
    test_dataset = SpectraPTDataset(
        pt_path=config["test_dir"],
        redshift_key="redshift_errors",
        apply_mask=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.fusion_input = data["x"]
        self.labels = data["y"]
        assert self.fusion_input.shape[0] == len(self.labels)

    def __len__(self):
        return self.fusion_input.shape[0]

    def __getitem__(self, idx):
        x = self.fusion_input[idx]  # [10, 2]
        y = self.labels[idx].float().unsqueeze(0)
        return x, y


def create_fusion_loaders(config):
    train_dataset = FusionDataset(config["train_dir"])
    val_dataset = FusionDataset(config["val_dir"])
    test_dataset = FusionDataset(config["test_dir"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
