"""SpectraNet model definition.
Authored by Maojie Xu, Argyro Sasli, and Alexandra Junell (2025)
"""
import gc
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim
from torch.amp import autocast
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_config(config, trial=None):
    logger = logging.getLogger(__name__)
    trial_id = getattr(trial, "number", "Manual")
    config_str = " | ".join([f"{k}={v}" for k, v in config.items()])
    logger.info(f"[Trial {trial_id}] {config_str}")


def train_one_epoch_regression(model, loader, optimizer, device, scaler, loss_fn, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    total_samples = 0
    progress_bar = tqdm(loader, desc="Training")

    for x, y in progress_bar:
        x, y = x.to(device), y.to(device).float()
        batch_size = x.size(0)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            pred = model(x)
            loss = loss_fn(pred, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        progress_bar.set_postfix(loss=loss.item())

    return total_loss / total_samples


def validate_regression(model, loader, device, loss_fn, plot=True):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation"):
            x, y = x.to(device), y.to(device).float()
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * x.size(0)

            preds.append(pred.cpu())
            targets.append(y.cpu())

    preds = torch.cat(preds).numpy().flatten()
    targets = torch.cat(targets).numpy().flatten()

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))

    dz = (preds - targets) / (1 + targets)
    bias = np.mean(dz)
    sigma_nmad = 1.48 * np.median(np.abs(dz))
    outlier_rate = np.mean(np.abs(dz) > 0.15)

    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(targets, preds, s=5, alpha=0.5, label="Predictions")
        z_min, z_max = min(targets), max(targets)
        plt.plot([z_min, z_max], [z_min, z_max], "r--", label="y=x")

        # Outlier 阈值 ±0.15(1+z)
        plt.plot([z_min, z_max], [z_min, z_max + 0.15 * (1 + z_max)], "g--", lw=1)
        plt.plot([z_min, z_max], [z_min, z_max - 0.15 * (1 + z_max)], "g--", lw=1)

        plt.xlabel("True redshift")
        plt.ylabel("Predicted redshift")
        plt.title("Predicted vs. True Redshift")
        plt.legend()
        plt.show()

    return {
        "loss": total_loss / len(loader.dataset),
        "mse": mse,
        "mae": mae,
        "bias": bias,
        "sigma_nmad": sigma_nmad,
        "outlier_rate": outlier_rate,
    }


def build_optimizer(model, config):
    return torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )


def build_scheduler(optimizer, config):
    warmup_epochs = config.get("warmup_epochs", 5)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=config["start_factor"],
        end_factor=config["end_factor"],
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config["T_0"], T_mult=config["T_mult"], eta_min=config["eta_min"]
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    return scheduler


def early_stopping(no_improve_epochs, patience):
    return no_improve_epochs >= patience


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
