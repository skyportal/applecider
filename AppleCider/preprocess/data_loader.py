"""SpectraNet model definition.
Authored by Maojie Xu, Argyro Sasli, and Alexandra Junell (2025)
"""
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms

class SpectraPTDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path, redshift_key='labels', class_order=None, apply_random_mask=False):
        data = torch.load(pt_path)
        self.flux = data['flux']
        self.labels = data[redshift_key]

        assert self.flux.shape[0] == len(self.labels), "Mismatch in flux and redshift length"

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_order)} if class_order else None
        self.apply_random_mask = apply_random_mask

        if self.class_to_idx and isinstance(self.labels[0], str):
            self.labels = [self.class_to_idx[label] for label in self.labels]

    def __len__(self):
        return self.flux.shape[0]

    def _apply_random_mask_with_mask(self, x):
        mask = torch.ones_like(x, dtype=torch.bool)
        
        seq_len = x.shape[-1]
        max_crop = seq_len // 4  # 25% of input length

        left_cut = np.random.randint(0, max_crop + 1)
        right_cut = seq_len - np.random.randint(0, max_crop + 1)

        x = x.clone()
        mask[:, :left_cut] = 0
        mask[:, right_cut:] = 0
        x[~mask] = 0.0

        return x, mask

    def __getitem__(self, idx):
        x = self.flux[idx]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, 4096]
        
        if self.apply_random_mask:
            x, mask = self._apply_random_mask_with_mask(x)
        else:
            mask = torch.ones_like(x, dtype=torch.bool)

        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y
def create_data_loaders(config):
    class_order = config['class_order'] 

    train_dataset = SpectraPTDataset(
        pt_path=config['train_dir'],
        redshift_key='labels',
        class_order=class_order,
        apply_random_mask=True  
    )
    val_dataset = SpectraPTDataset(
        pt_path=config['val_dir'],
        redshift_key='labels',
        class_order=class_order,
        apply_random_mask=False
    )
    test_dataset = SpectraPTDataset(
        pt_path=config['test_dir'],
        redshift_key='labels',
        class_order=class_order,
        apply_random_mask=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader, test_loader, class_order


class FusionClassificationDataset(Dataset):
    def __init__(self, pt_file, class_order=None, label_to_index=None):
        data = torch.load(pt_file)
        self.x = data['x']  # [N, 10, 12]
        raw_labels = data['y']  # List[str] or List[int]

        if class_order is not None:
            self.label_to_index = {name: i for i, name in enumerate(class_order)}
        elif label_to_index is not None:
            self.label_to_index = label_to_index
        elif isinstance(raw_labels[0], str):
            class_names = sorted(set(raw_labels))
            self.label_to_index = {name: i for i, name in enumerate(class_names)}
        else:
            self.label_to_index = None

        if self.label_to_index and isinstance(raw_labels[0], str):
            self.y = torch.tensor([self.label_to_index[label] for label in raw_labels], dtype=torch.long)
        else:
            self.y = torch.tensor(raw_labels, dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],        # shape: [10, 12]
            'y': self.y[idx]         # int label
        }
        
def create_fusion_loaders_classification(config):
    class_order = config.get("class_order", None)

    train_set = FusionClassificationDataset(
        pt_file=config['train_dir'],
        class_order=class_order
    )
    val_set = FusionClassificationDataset(
        pt_file=config['val_dir'],
        label_to_index=train_set.label_to_index
    )
    test_set = FusionClassificationDataset(
        pt_file=config['test_dir'],
        label_to_index=train_set.label_to_index
    )

    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True,  num_workers=config['num_workers'], pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    return train_loader, val_loader, test_loader,class_order
