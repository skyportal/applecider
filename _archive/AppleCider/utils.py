import os
import gc
import random
import logging

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

# some utility functions/helpers
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_config(config, trial=None):
    logger = logging.getLogger(__name__)
    trial_id = getattr(trial, 'number', 'Manual')
    config_str = ' | '.join([f"{k}={v}" for k, v in config.items()])
    logger.info(f"[Trial {trial_id}] {config_str}")

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# some metrics functions
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
        return classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    except:
        return {cls: {'f1-score': 0.0} for cls in class_names}

def save_confusion_matrix_if_needed(all_targets, all_preds, class_names, project_root, epoch):
    save_dir = os.path.join(project_root, "record")
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"val_epoch_{epoch + 1:03d}" if isinstance(epoch, int) else "test_comparison" if epoch == "test" else str(epoch)
    save_path = os.path.join(save_dir, f"confusion_matrix_{suffix}.png")
    plot_confusion_matrix_double(all_targets, all_preds, class_names, save_path=save_path, dataset_type=suffix)

def plot_confusion_matrix_double(y_true, y_pred, class_names, save_path=None, dataset_type=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix - Absolute Values')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_normalized * 100, annot=True, fmt='.0f', cmap='Purples', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix - %')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    title = 'Confusion Matrix Comparison'
    if dataset_type:
        title = f'{title} {dataset_type}'

    plt.suptitle(title, fontsize=18)
    plt.tight_layout()

    if save_path :
        plt.savefig(save_path)

    plt.show()

# multi seed run for better statistics
def run_multi_seed(config, seed_list, main=None, build_model=None):
    test_scores = []
    test_metrics_list = []
    macro_auc_list = []
    per_class_auc_lists = {}

    for i, seed in enumerate(seed_list):
        print(f"\n=== Running seed {seed} ({i+1}/{len(seed_list)}) ===")
        config['seed'] = seed
        test_score, test_metrics = main(trial=None, config=config, build_model=build_model)

        test_scores.append(test_score)
        test_metrics_list.append(test_metrics)

        # Collect Macro AUC and Per-Class AUCs
        per_class_auc_dict = test_metrics.get("per_class_auc_dict", {})
        macro_auc = np.mean(list(per_class_auc_dict.values()))
        macro_auc_list.append(macro_auc)

        for cls_name, auc_val in per_class_auc_dict.items():
            if cls_name not in per_class_auc_lists:
                per_class_auc_lists[cls_name] = []
            per_class_auc_lists[cls_name].append(auc_val)

    # Aggregate metrics
    mean_score = np.mean(test_scores)
    std_score = np.std(test_scores)

    acc_list = [m["acc"] for m in test_metrics_list]
    top3_acc_list = [m["test_top3_acc"] for m in test_metrics_list]
    f1_list = [m["f1"] for m in test_metrics_list]

    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)

    mean_top3_acc = np.mean(top3_acc_list)
    std_top3_acc = np.std(top3_acc_list)

    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)

    mean_macro_auc = np.mean(macro_auc_list)
    std_macro_auc = np.std(macro_auc_list)

    # Pack results
    result_summary = {
        "seed_list": seed_list,
        "test_scores": test_scores,
        "test_metrics_list": test_metrics_list,
        "acc_list": acc_list,
        "top3_acc_list": top3_acc_list,
        "f1_list": f1_list,
        "macro_auc_list": macro_auc_list,
        "per_class_auc_lists": per_class_auc_lists,
        "mean_score": mean_score,
        "std_score": std_score,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_top3_acc": mean_top3_acc,
        "std_top3_acc": std_top3_acc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_macro_auc": mean_macro_auc,
        "std_macro_auc": std_macro_auc,
    }

    print("\n=== Multi-Seed Test Results ===")
    for i, seed in enumerate(seed_list):
        print(f"Seed {seed}: Composite Score = {test_scores[i]:.4f}, "
              f"Acc = {acc_list[i]:.2f}%, Top-3 Acc = {top3_acc_list[i]:.2f}%, "
              f"F1 = {f1_list[i]:.4f}, Macro AUC = {macro_auc_list[i]:.4f}")

    print("\n=== Multi-Seed Mean / Std ===")
    print(f"Composite Score: Mean = {mean_score:.4f}, Std = {std_score:.4f}")
    print(f"Accuracy:        Mean = {mean_acc:.2f}%, Std = {std_acc:.2f}%")
    print(f"Top-3 Accuracy:  Mean = {mean_top3_acc:.2f}%, Std = {std_top3_acc:.2f}%")
    print(f"Macro F1:        Mean = {mean_f1:.4f}, Std = {std_f1:.4f}")
    print(f"Macro AUC:       Mean = {mean_macro_auc:.4f}, Std = {std_macro_auc:.4f}")

    print("\n=== Per-Class AUC (Mean ± Std) ===")
    for cls_name, aucs in per_class_auc_lists.items():
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print(f"{cls_name:20s}: Mean = {mean_auc:.4f}, Std = {std_auc:.4f}")

    return result_summary

# some training helpers
def get_cb_weights(train_df, class_names, beta=0.9999):
    label_counts = train_df['label'].value_counts().to_dict()
    class_counts = [label_counts.get(c, 1) for c in class_names]
    effective_nums = [(1 - beta ** n) / (1 - beta) for n in class_counts]
    weights = [1.0 / en for en in effective_nums]
    weights = [w / sum(weights) for w in weights]
    return torch.tensor(weights, dtype=torch.float)

def calculate_composite_score(acc, top3_acc, f1, minority_acc, weights=(0.4, 0.3, 0.3, 0)):
    return (
        weights[0] * acc +
        weights[1] * top3_acc +
        weights[2] * f1 +
        weights[3] * (minority_acc if minority_acc is not None else 0.0)
    )

def early_stopping(no_improve_epochs, patience):
    return no_improve_epochs >= patience

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()