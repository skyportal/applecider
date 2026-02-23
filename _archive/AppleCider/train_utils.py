import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# our custom utils
import preprocess.utils as utils
import torch
import torch.nn as nn
import torch.optim
from sklearn.metrics import auc, f1_score, roc_auc_score, roc_curve
from torch.amp import autocast
from tqdm import tqdm


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


# for validation and testing
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
    class_top3_acc = utils.compute_class_top3_acc(all_targets, all_probs, len(class_names))

    composite_score = utils.calculate_composite_score(
        val_acc / 100, val_top3_acc / 100, val_f1, minority_acc=None
    )

    utils.plot_confusion_matrix_double(all_targets, all_preds, class_names)

    # === AUC-ROC ===
    if epoch_type == "test":
        y_true = np.array(all_targets)
        y_prob = all_probs
        num_classes = len(class_names)

        try:
            macro_auc = roc_auc_score(y_true, y_prob, average="macro", multi_class="ovr")
            print(f"\n[TEST] Macro ROC-AUC: {macro_auc:.4f}")

            #
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


# Optimizer and Scheduler
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
