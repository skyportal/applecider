"""SpectraNet model definition.
Authored by Maojie Xu, Argyro Sasli, and Alexandra Junell (2025)
"""
import gc
import logging
import os
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.amp import autocast
from tqdm import tqdm


# 1. EMA
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# 2. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.full_like(targets, self.alpha).to(inputs.device)
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
            else:
                raise TypeError("alpha must be float or torch.Tensor")
            ce_loss = ce_loss * alpha_t

        loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# 4. Utility Functions
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


# 5. Training and Validation


def train_one_epoch(
    model, loader, criterion, optimizer, device, scaler, max_grad_norm=1.0, ema=None, use_redshift=False
):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(loader, desc="Training")

    for batch in progress_bar:
        if use_redshift:
            sequence, redshift, target = batch
            redshift = redshift.to(device).unsqueeze(1)  # shape: [B, 1]
        else:
            sequence, target = batch
            redshift = None

        sequence, target = sequence.to(device), target.to(device)
        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            outputs = model(sequence, redshift) if use_redshift else model(sequence)
            loss = criterion(outputs, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema:
            ema.update()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += preds.eq(target).sum().item()
        total += target.size(0)
        progress_bar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    return running_loss / len(loader), 100.0 * correct / total


def validate(
    model, loader, criterion, device, class_names, epoch, project_root, use_redshift=False, epoch_type="val"
):
    model.eval()
    macro_auc = None
    per_class_auc_dict = {}

    total, correct, correct_top3, running_loss = 0, 0, 0, 0.0
    all_preds, all_targets, all_probs = [], [], []
    progress_bar = tqdm(loader, desc="Validation")

    with torch.no_grad():
        for batch in progress_bar:
            if use_redshift:
                sequence, redshift, target = batch
                redshift = redshift.to(device).unsqueeze(1)
            else:
                sequence, target = batch
                redshift = None

            sequence, target = sequence.to(device), target.to(device)

            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                output = model(sequence, redshift) if use_redshift else model(sequence)
                loss = criterion(output, target)
                probs = torch.softmax(output, dim=1)

            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            _, top3 = output.topk(3, dim=1)
            correct_top3 += (top3 == target.view(-1, 1)).any(dim=1).sum().item()
            total += target.size(0)
            running_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            progress_bar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    val_loss = running_loss / len(loader)
    val_acc = 100.0 * correct / total
    val_top3_acc = 100.0 * correct_top3 / total
    val_f1 = f1_score(all_targets, all_preds, average="macro")

    all_targets = np.array(all_targets)
    all_probs = np.concatenate(all_probs)
    class_top3_acc = compute_class_top3_acc(all_targets, all_probs, len(class_names))

    composite_score = calculate_composite_score(val_acc / 100, val_top3_acc / 100, val_f1, minority_acc=None)

    plot_confusion_matrix_double(all_targets, all_preds, class_names)

    # === 测试集绘制 AUC-ROC ===
    if epoch_type == "test":
        y_true = np.array(all_targets)
        y_prob = all_probs
        num_classes = len(class_names)

        try:
            macro_auc = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr")
            print(f"\n[TEST] Macro ROC-AUC: {macro_auc:.4f}")

            # 普通论文风格参数（机器学习/计算机视觉常用）
            mpl.rcParams.update(
                {
                    "font.family": "sans-serif",
                    "mathtext.fontset": "stixsans",
                    "font.size": 9,
                    "axes.labelsize": 10,
                    "axes.titlesize": 10,
                    "legend.fontsize": 4,
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9,
                    "lines.linewidth": 0.6,
                    "axes.linewidth": 1.0,
                    "xtick.direction": "in",
                    "ytick.direction": "in",
                    "xtick.top": True,
                    "ytick.right": True,
                    "legend.frameon": True,
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                }
            )

            fig, ax = plt.subplots(figsize=(4, 3))
            colors = (
                plt.cm.tab10(np.linspace(0, 1, num_classes))
                if num_classes <= 10
                else plt.cm.tab20(np.linspace(0, 1, num_classes))
            )

            for i in range(num_classes):
                binary_true = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(binary_true, y_prob[:, i])
                roc_auc_val = auc(fpr, tpr)
                per_class_auc_dict[class_names[i]] = roc_auc_val
                ax.plot(fpr, tpr, color=colors[i], label=f"{class_names[i]} (AUC={roc_auc_val:.2f})")

            ax.plot([0, 1], [0, 1], "k--", lw=1.0)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right", handlelength=1.5)
            ax.minorticks_on()
            ax.grid(True, linestyle="--", alpha=0.5)

            fig.tight_layout()
            save_dir = os.path.join(project_root, "record")
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, "roc_curve_test.pdf"), bbox_inches="tight")
            fig.savefig(os.path.join(save_dir, "roc_curve_test.png"), dpi=300, bbox_inches="tight")
            plt.show()
            print(f"[TEST] ROC curve saved to: {save_dir}")

        except ValueError as e:
            print(f"[TEST] ROC-AUC curve generation failed: {e}")
            macro_auc = None
            per_class_auc_dict = {}

    return (
        val_loss,
        val_acc,
        val_f1,
        val_top3_acc,
        class_top3_acc,
        composite_score,
        macro_auc,
        per_class_auc_dict,
    )


# 6. Optimizer and Scheduler


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


# 7. Extra Utils


def compute_class_top3_acc(all_targets, all_probs, num_classes):
    class_top3_acc = []
    for i in range(num_classes):
        indices = np.where(all_targets == i)[0]
        if len(indices) > 0:
            probs = all_probs[indices]
            top3 = np.argsort(probs, axis=1)[:, -3:]
            acc = np.any(top3 == i, axis=1).mean()
            class_top3_acc.append(acc)
        else:
            class_top3_acc.append(0.0)
    return class_top3_acc


def safe_classification_report(all_targets, all_preds, class_names):
    try:
        return classification_report(
            all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0
        )
    except:
        return {cls: {"f1-score": 0.0} for cls in class_names}


def save_confusion_matrix_if_needed(all_targets, all_preds, class_names, project_root, epoch):
    save_dir = os.path.join(project_root, "record")
    os.makedirs(save_dir, exist_ok=True)
    suffix = (
        f"val_epoch_{epoch + 1:03d}"
        if isinstance(epoch, int)
        else "test_comparison"
        if epoch == "test"
        else str(epoch)
    )
    save_path = os.path.join(save_dir, f"confusion_matrix_{suffix}.png")
    plot_confusion_matrix_double(
        all_targets, all_preds, class_names, save_path=save_path, dataset_type=suffix
    )


def plot_confusion_matrix_double(y_true, y_pred, class_names, save_path=None, dataset_type=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples", xticklabels=class_names, yticklabels=class_names, ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix - Absolute Values")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_normalized * 100,
        annot=True,
        fmt=".0f",
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix - %")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    title = "Confusion Matrix Comparison"
    if dataset_type:
        title = f"{title} {dataset_type}"

    plt.suptitle(title, fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def get_cb_weights(train_df, class_names, beta=0.9999):
    label_counts = train_df["label"].value_counts().to_dict()
    class_counts = [label_counts.get(c, 1) for c in class_names]
    effective_nums = [(1 - beta**n) / (1 - beta) for n in class_counts]
    weights = [1.0 / en for en in effective_nums]
    weights = [w / sum(weights) for w in weights]
    return torch.tensor(weights, dtype=torch.float)


def calculate_composite_score(acc, top3_acc, f1, minority_acc, weights=(0.4, 0.3, 0.3, 0)):
    return (
        weights[0] * acc
        + weights[1] * top3_acc
        + weights[2] * f1
        + weights[3] * (minority_acc if minority_acc is not None else 0.0)
    )


def early_stopping(no_improve_epochs, patience):
    return no_improve_epochs >= patience


def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
