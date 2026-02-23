import os
from datetime import datetime

import numpy as np
import optuna
import seaborn as sns
import torch
import wandb
from AppleCider.util.early_stopping import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        warmup_scheduler,
        criterion,
        criterion_val,
        device,
        config,
        trial=None,
    ):
        self.model = model  ## AppleCider, ZwickyCider, Informer, MetaModel, BTSModel, GalSpecNet
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.criterion = criterion
        self.criterion_val = criterion_val
        self.device = device
        self.trial = trial  ## if using optuna

        self.mode = config["mode"]  ## 'all', 'ztf', 'photo', 'meta', 'image', 'spectra'
        self.save_weights = config["save_weights"]  ## True, False
        self.weights_path = config["weights_path"]
        self.use_wandb = config["use_wandb"]  ## True, False
        self.test_run_id = config["test_run_id"]  ## True, False (if using test set)

        self.early_stopping = EarlyStopping(patience=config["early_stopping_patience"])
        self.warmup_epochs = config["warmup_epochs"]
        self.clip_grad = config["clip_grad"]
        self.clip_value = config["clip_value"]

        self.group_labels = config["group_labels"]  ## True, False: labels -> SN I, SN II, CV, AGN, TDE
        self.group_labels_SN = config["group_labels_SN"]  ## True, False: labels -> SN, CV, AGN, TDE

        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0

        if self.use_wandb:
            self.run_id = config["run_id"]
        if self.test_run_id:  ## True, False (if using test set)
            self.run_id = config["run_id"]

    def store_weights(self, epoch):
        if self.use_wandb:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.weights_path,
                    f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{epoch}-{self.run_id}.pth',
                ),
            )
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.weights_path,
                    f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-best-{self.run_id}.pth',
                ),
            )

        else:
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{epoch}.pth'
                ),
            )
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-best.pth'
                ),
            )

    def zero_stats(self):
        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0

    def update_stats(self, loss, logits, labels):
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probabilities, dim=1)
        correct_predictions = (predicted_labels == labels).sum().item()

        self.total_correct_predictions += correct_predictions
        self.total_predictions += labels.size(0)
        self.total_loss.append(loss.item())

    def calculate_stats(self):
        return sum(self.total_loss) / len(
            self.total_loss
        ), self.total_correct_predictions / self.total_predictions

    def get_logits(self, photometry, photometry_mask, images, metadata, spectra):
        if self.mode == "photo":
            logits = self.model(photometry, photometry_mask)
        elif self.mode == "spectra":
            logits = self.model(spectra)
        elif self.mode == "meta":
            logits = self.model(metadata)
        elif self.mode == "image":
            logits = self.model(images)
        elif self.mode == "ztf":
            logits = self.model(photometry, photometry_mask, images, metadata)
        else:  # all 4 modalities
            logits = self.model(photometry, photometry_mask, images, metadata, spectra)

        return logits

    def step(self, photometry, photometry_mask, images, metadata, spectra, labels):
        """Perform a training step for the classification model"""
        logits = self.get_logits(photometry, photometry_mask, images, metadata, spectra)

        loss = self.criterion(logits, labels)

        self.update_stats(loss, logits, labels)

        return loss

    def step_val(self, photometry, photometry_mask, images, metadata, spectra, labels):
        """Perform a training step for the classification model"""
        logits = self.get_logits(photometry, photometry_mask, images, metadata, spectra)

        loss = self.criterion_val(logits, labels)

        self.update_stats(loss, logits, labels)

        return loss

    def get_gradient_norm(self):
        total_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm**0.5

    def train_epoch(self, train_dataloader):
        self.model.train()
        self.zero_stats()

        for photometry, photometry_mask, images, metadata, spectra, labels in tqdm(
            train_dataloader, total=len(train_dataloader), desc="Train", colour="#9ACD32", leave=True
        ):
            photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
            images, metadata, spectra = (
                images.to(self.device),
                metadata.to(self.device),
                spectra.to(self.device),
            )
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            loss = self.step(photometry, photometry_mask, images, metadata, spectra, labels)

            if self.use_wandb:
                wandb.log({"step_loss": loss.item()})

            loss.backward()

            if self.use_wandb:
                grad_norm = self.get_gradient_norm()
                wandb.log({"grad_norm": grad_norm})

            self.optimizer.step()

        loss, acc = self.calculate_stats()

        return loss, acc

    def val_epoch(self, val_dataloader):
        self.model.eval()
        self.zero_stats()

        with torch.no_grad():
            for photometry, photometry_mask, images, metadata, spectra, labels in tqdm(
                val_dataloader, total=len(val_dataloader), desc="Validation", colour="#9ACD32", leave=True
            ):
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                images, metadata, spectra = (
                    images.to(self.device),
                    metadata.to(self.device),
                    spectra.to(self.device),
                )
                labels = labels.to(self.device)

                if self.mode == "clip":
                    self.step_val_clip(photometry, photometry_mask, spectra, metadata)
                else:
                    self.step_val(photometry, photometry_mask, images, metadata, spectra, labels)

        loss, acc = self.calculate_stats()

        return loss, acc

    def train(self, train_dataloader, val_dataloader, epochs):
        best_val_loss = np.inf
        best_val_acc = 0

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_dataloader)
            val_loss, val_acc = self.val_epoch(val_dataloader)

            best_val_loss = min(val_loss, best_val_loss)

            if self.trial:
                self.trial.report(val_loss, epoch)

                if self.trial.should_prune():
                    print("Prune")
                    wandb.finish()
                    raise optuna.exceptions.TrialPruned()

            if self.warmup_scheduler and epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
                current_lr = self.warmup_scheduler.get_last_lr()[0]
            else:
                self.scheduler.step(val_loss)
                current_lr = self.scheduler.get_last_lr()[0]

            if self.use_wandb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "learning_rate": current_lr,
                        "epoch": epoch,
                    }
                )

            if best_val_acc < val_acc:
                best_val_acc = val_acc

                if self.use_wandb:
                    wandb.log({"best_val_acc": best_val_acc})

                if self.save_weights:
                    self.store_weights(epoch)

            print(
                f"Epoch {epoch}: Train Loss {round(train_loss, 4)} \t Val Loss {round(val_loss, 4)} \t \
                    Train Acc {round(train_acc, 4)} \t Val Acc {round(val_acc, 4)}"
            )

            if self.early_stopping.step(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        return best_val_loss

    def evaluate(self, val_dataloader, id2target):
        self.model.eval()

        all_true_labels = []
        all_predicted_labels = []

        for photometry, photometry_mask, images, metadata, spectra, labels in tqdm(
            val_dataloader, total=len(val_dataloader), desc="Validation", colour="#9ACD32", leave=True
        ):
            with torch.no_grad():
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                images, metadata, spectra = (
                    images.to(self.device),
                    metadata.to(self.device),
                    spectra.to(self.device),
                )

                logits = self.get_logits(photometry, photometry_mask, images, metadata, spectra)
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                _, predicted_labels = torch.max(probabilities, dim=1)

                all_true_labels.extend(labels.numpy())
                all_predicted_labels.extend(predicted_labels.cpu().numpy())

        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        conf_matrix_percent = 100 * conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

        labels = [id2target[i] for i in range(len(conf_matrix))]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))

        ## Plot absolute values confusion matrix
        sns.heatmap(
            conf_matrix, annot=True, fmt="d", cmap="BuPu", xticklabels=labels, yticklabels=labels, ax=axes[0]
        )
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("True")
        axes[0].set_title("Confusion Matrix - Absolute Values")

        ## Plot percentage values confusion matrix
        sns.heatmap(
            conf_matrix_percent,
            annot=True,
            fmt=".0f",
            cmap="BuPu",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[1],
        )
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel(" ")
        axes[1].set_title("Confusion Matrix - %")

        if self.use_wandb:
            ## Save confusion matrix img
            wandb.log({"conf_matrix": wandb.Image(fig)})

            ## Save absolute confusion matrix as table
            artifact_matrix = wandb.Artifact("confusion-matrix", type="table")
            if self.group_labels:
                matrix_table = wandb.Table(columns=["SN I", "SN II", "CV", "AGN", "TDE"], data=conf_matrix)
            elif self.group_labels_SN:
                matrix_table = wandb.Table(columns=["SN", "CV", "AGN", "TDE"], data=conf_matrix)
            else:
                matrix_table = wandb.Table(
                    columns=[
                        "SN Ia",
                        "SN Ic",
                        "SN Ib",
                        "SN II",
                        "SN IIP",
                        "SN IIn",
                        "SN IIb",
                        "CV",
                        "AGN",
                        "TDE",
                    ],
                    data=conf_matrix,
                )

            artifact_matrix.add(matrix_table, "absolute-confusion-matrix")
            wandb.log_artifact(artifact_matrix)

        return conf_matrix
