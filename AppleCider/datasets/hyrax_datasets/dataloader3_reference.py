import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import unittest
import tempfile
import shutil
from collections import defaultdict
import random
import math
import inspect
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from tqdm import tqdm
# import pickle
# from cnn import Image_net
from datetime import datetime as t


# import numpy as np
# from astropy.stats import sigma_clipped_stats
# from photutils.psf import EPSFBuilder
# from photutils.detection import find_peaks
# from photutils import FWHM, centroid_2dg, centroid_com

        

class AstroDataset(Dataset):

    def __init__(self, raw_files, config,all_samples = False, augment=True, num_workers=4):
        tnow = t.now()
        eps = 1e-8
        

        self.raw_files = raw_files
        # print(f"time taken loading files: {t.now()-tnow}")
        
        self.augment = augment
        self.classes = config['classes'].values()
        self.eps = 1e-8  # Small value to prevent division by zero
        self.augmentation_transform = transform_image()
        self.all_samples = all_samples
        self.config = config
        



        real_classes = [
            "AGN",
            "TDE",
            "SN II",
            "SN IIp",
            "SN Ia",
            "SN IIn",
            "SN Ib",
            "SN Ic",
            "Cataclysmic"
        ]
        # model = astroMiNN


        # Build object index mapping

        if self.all_samples:
            self.obj_info = []
            self.obj_id_to_indices = defaultdict(list)
            self.samples = []
        else:
            self.obj_id_to_samples = defaultdict(list)

        self.objects = []
        self.real_classes = []
        self.target_counts = torch.zeros(len(config['classes'].keys()), dtype=torch.float32)





        for sample in tqdm(self.raw_files): 
            # Load sample and handle obj_id
            if self.all_samples and config['alert'] > 0 and sample['alerte'] > config['alert']:
                continue
            obj_id = sample.get('obj_id')
            # print(sample)
            # exit()

            original_class = sample['target']
            target = np.zeros(len(self.classes))
            real_target = np.zeros(9)
            for idy, category in enumerate(self.classes):
                if original_class in category:
                    target[idy] = 1.0
            target = torch.tensor(target, dtype=torch.float32)
            for idy, category in enumerate(real_classes):
                if original_class == category:
                    real_target[idy] = 1.0

            # print(sample['metadata'])



            if not obj_id in self.objects:
                self.objects.append(obj_id)
                self.real_classes.append(real_target)
                if not self.all_samples:
                    self.target_counts += target
            if self.all_samples:
                self.target_counts += target

                    

            image = sample['image']
            # if config['patch_size']==5:
            #     config['cutout_size'] = 60

            if "vit_tower" in config["tags"]:
                i1 = int((63-config["patch_size"][0])/2)
                i2 = int(63 - i1)

            elif not config["cutout_size"]==63:
                i1 = int((63-config["cutout_size"])/2)
                i2 = int(63 - i1)

            else:
                i1 = 0
                i2 = 63
            image = sample['image'][:, i1:i2, i1:i2]

            if config['image_norm'] == 'median':
                for c in range(3):
                    channel_median = torch.median(image[c].reshape(-1))
                    image[c] = image[c] - channel_median
                    image[c] = image[c]/(image[c].std()+ eps)

            elif config['image_norm'] == 'L2':
                norm = torch.norm(image, p=2)  # L2 norm over all elements
                image = image/norm
            
            sample["metadata"] = torch.nan_to_num(torch.tensor(sample["metadata"], dtype=torch.float32), nan=-999.0)



            sample = {
            'metadata': sample['metadata'],
            'image': image,
            'target': target,
            'real_target': real_target,
            'obj_id': obj_id,
            'alert': sample['alerte']
        }
            if self.all_samples:
                self.obj_id_to_indices[obj_id].append(len(self.samples))
                self.samples.append(sample)

            else:
                self.obj_id_to_samples[obj_id].append(sample)


    def _load_sample(self, idx):
        """Load sample"""
        file_path = os.path.join(self.npy_dir, self.file_names[idx])
        return np.load(file_path, allow_pickle=True).item()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.all_samples:
            sample = self.samples[idx]
        

        else:
            objectid = self.objects[idx]
            # print(objectid)
            samples = sorted(self.obj_id_to_samples[objectid], key=lambda x: x["alert"])[:self.config["alert"]]
            # print([sample['obj_id'] for sample in samples])
            # exit()
            if int(self.config["alert"]) == 0:
                sample = random.choice(samples)
            else:
                if self.config["alert"] > len(samples):
                    x = len(samples)
                elif self.config["alert"] < -len(samples):
                    x = -len(samples)
                else:
                    x = self.config["alert"]
                sample = samples[int(x)-1 if x>0 else int(x)]
        target = sample['target']
        image = sample['image']

        if self.augment:
            image = self.augmentation_transform(image)

        return {
            'metadata': sample["metadata"],
            'image': image,
            'target': target,
            'real_target': sample['real_target'],
            'obj_id': sample['obj_id']
        }





def transform_image():
    return transforms.Compose([
        # Geometric transforms first
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.Lambda(lambda x: torch.rot90(x, k=1, dims=(-2, -1))),  # 90°
            transforms.Lambda(lambda x: torch.rot90(x, k=3, dims=(-2, -1))),  # 270°
        ], p=0.75),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.0005, 0.0005)),
        ], p=1),
    ])






def get_dataset(dir_path:str, config:dict, batch_size=32, seed=33, 
                            num_workers=12):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    file_names = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy')])
    # if config["debug"]:
    #         file_names = sorted([f for f in os.listdir(dir_path)[:5000] if f.endswith('.npy')])
    # else:
    #     file_names = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy')])

    raw_files = [np.load(os.path.join(dir_path, path), allow_pickle=True).item()
            for path in tqdm(file_names)]
            
    return raw_files
    


def make_dataloaders(dataset, config:dict, batch_size=32, seed=33, 
                            num_workers=12):

    dataset = AstroDataset(dataset, config, all_samples = config["use_all_samples"], augment=True, num_workers=num_workers)

    
    # First split: train vs (val + test) at OBJECT level
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    train_obj_indexes, val_test_obj_indexes = next(sss.split(np.zeros(len(dataset.objects)), dataset.real_classes))
    
    

    # Second split: val vs test from the remaining 30%
    val_test_classes = [dataset.real_classes[index] for index in val_test_obj_indexes]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_obj_indexes, test_obj_indexes = next(sss.split(np.zeros(len(val_test_obj_indexes)), val_test_classes))

    

    if config["use_all_samples"]:
        train_obj_ids = [dataset.objects[i] for i in train_obj_indexes]
        val_test_obj_ids = [dataset.objects[i] for i in val_test_obj_indexes]
        val_obj_ids = [val_test_obj_ids[i] for i in val_obj_indexes]
        test_obj_ids = [val_test_obj_ids[i] for i in test_obj_indexes]

        train_indices = []
        for obj_id in train_obj_ids:
            train_indices.extend(dataset.obj_id_to_indices[obj_id])
        
        val_indices = []
        for obj_id in val_obj_ids:
            val_indices.extend(dataset.obj_id_to_indices[obj_id])
        
        test_indices = []
        for obj_id in test_obj_ids:
            test_indices.extend(dataset.obj_id_to_indices[obj_id])
        
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        test_subset = Subset(dataset, test_indices)

    else:
        train_subset = Subset(dataset, train_obj_indexes)
        val_subset = Subset(dataset, val_obj_indexes)
        test_subset = Subset(dataset, test_obj_indexes)

    # Create dataloaders
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers//2, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size*4, shuffle=False, num_workers=num_workers//4, pin_memory=True
    )
    test_loader = DataLoader(
        test_subset, batch_size=batch_size*4, shuffle=False, num_workers=num_workers//4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, dataset.target_counts