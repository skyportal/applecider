import os
import sys ; sys.path.insert(0, '/projects/bcrv/abrown3/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import random

from tqdm.auto import tqdm

import torch
import math
from datetime import datetime

import optuna
from optuna.exceptions import DuplicatedStudyError

#from pathlib import Path
from types   import SimpleNamespace
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence

#from sklearn.preprocessing    import label_binarize
from sklearn.metrics          import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, fbeta_score,
    top_k_accuracy_score, roc_curve,
    precision_recall_curve, confusion_matrix,
    classification_report
)
from matplotlib.ticker        import AutoMinorLocator
from matplotlib               import cm as _cm
from collections import Counter

import torch.nn as nn
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, LinearLR
import torch.nn.functional as F 
from torchvision import transforms

from AppleCiDEr_Skyportal.AppleCider.core.dataset import CiderDataset
from AppleCiDEr_Skyportal.AppleCider.models.Time2Vec import collate
from AppleCiDEr_Skyportal.AppleCider.models.XastroMiNN import ResidualTowerBlock, SplitHeadConvNeXt


config = {
        'project': 'AppleCiDEr',
        'config_from': None,
        'random_seed': 42,  # 42, 66, 0, 12, 123
        'use_wandb': True,
        'save_weights': True,
        'weights_path': f'/projects/bcrv/abrown3/AppleCiDEr_Skyportal/cider_weights/multi-{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        'use_pretrain': None,
        'freeze': False,

        'mode':'all',        # 'photo', 'spectra', 'metdata & images', 'all', 'ztf'
        ## Data General
        'step': 'type',
        'classes': ['SN I','SN II', 'Cataclysmic', 'AGN', 'Tidal Disruption Event'],
        'group_labels': True,
        'num_classes': 5,
        "gpu":1,
        
        ## Data General
        'preprocessed_path': '/work/nvme/bcrv/abrown3/preprocessed_data/data_multi/day10',
        # for ALERTS 
        'train_csv_path':'/projects/bcrv/abrown3/AppleCiDEr_csv/AppleCider_Train_vetted_7-6.csv',
        'val_csv_path':'/projects/bcrv/abrown3/AppleCiDEr_csv/AppleCider_Val_vetted_7-6.csv',
        'test_csv_path':'/projects/bcrv/abrown3/AppleCiDEr_csv/AppleCider_Test_vetted_7-6.csv',
         # for SPECTRA
        'spec_dir': '/work/nvme/bcrv/mxu11',
        #'class_weights': False,
        #'generate_train_val_files': False,
        
        ## personal tag for weights files in local folder 
        'custom_weight_path': False,
        'custom_weight_name': '-',
        
        ## only for the wandb users.....
        'use_notes_tags': True,
        'wandb_tags': ['Delta!', 'hidden dim', 'group labels'],
        'wandb_notes': f'hidden dim',

        ## Photometry Model 🍏🍏
        'photo_event_path': '/work/hdd/bcrv/ffontinelenunes/data/AppleCider/photo_events',
        
        'output_dir':'/work/hdd/bcrv/ffontinelenunes/data/AppleCider/photo_events',
        'stats_file':'/work/hdd/bcrv/ffontinelenunes/data/AppleCider/photo_events/feature_stats_day100.npz',
        'horizon_days' :  10.0, # <- fine-tuning on 50 days
        #'batch_size'   :  256, #64,
        'sampler_balance': False,
        'num_workers'    : 0,
        # model stuff
        'p_d_model': 128,
        'p_n_heads': 8,       #4,
        'p_n_layers': 4,    #2,
        'p_dropout': 0.30,
        'max_len': 257,     #300,#256,#128,#128,
        #'lr' : 5e-6,
        # 'weight_decay' : 1e-2,
        # 'focal_gamma':2.0,
        #'cut_time_p':None, #(.25,.25,.25,.25), #None,  # or (.25,.25,.25,.25)
        #'p_dropout':0.1,
        #'jitter_scale':0.10,
        #'flux_nu':8,    
        # training schedule
        #'epochs' :150,
        #'patience' :30,
        # misc 
        #'seed':42,
        #'NUM_CLASSES':5,

        ## Spectra Model 🍏🍏🍏🍏🍏
    
        #"learning_rate",
        #"weight_decay": 1e-5,
        #"ema_decay": 0.995,
        #"focal_loss_gamma": 2.0,
        #"warmup_epochs": 10,
        "T_0": 5,
        "T_mult": 1,
        "eta_min": 1e-5,
        "start_factor": 1e-6,
        "end_factor": 1.0,
        #"num_workers": 0,
        #"sampling": False,
        #"epochs": 100,
        #"optimizer": "AdamW",
        #"patience": 14,
        "train_dir": '/work/nvme/bcrv/mxu11',  "val_dir": '/work/nvme/bcrv/mxu11', "test_dir":'/work/nvme/bcrv/mxu11',

        

         ## Metadata & Image Model 🍏🍏
        "num_experts":4,
        "towers_hidden_dims":16,
        "towers_outdims":4,
        "embedding":"False",
        
        "fusion_hidden_dims":128,
        "fusion_router_dims":128,
        "fusion_outdims":4,
        
        "cnn_lr":2,
        "cnn_decay":5e-2,
        "psf_lr":0.5,
        "psf_decay":5e-2,
        "mag_lr": 2,
        "mag_decay": 0.0,
        
        "lc_lr": 2,
        "lc_decay": 0.05,
        "spatial_lr":2,
        "spatial_decay":0.0,
        
        "coord_lr": 0.5,
        "coord_decay": 0.0,
        
        "nst1_lr":2,
        "nst1_decay":0.0,
        "nst2_lr":2,
        "nst2_decay":0.0,
        
        "fusion_lr":1,
        "fusion_decay":1e-2,
        "fusion_beta1": 0.9,
        "fusion_beta2": 0.999,
        
        "router_decay":0.0,
        "router_lr":1.5,
        "router_beta1":0.9,
        "router_beta2":0.999,
        "router_lr_2":1,
        "router_beta1_2":0.95,
        "router_beta2_2":0.99,
        
        "classifier_decay": 0.0,
        "classifier_lr":2.44,
        "classifier_beta1":0.95,
        "classifier_beta2":0.99,
        
        "beta1":0.9,
        "beta2":0.999,
        "eps":5e-10,
        
        "sched_pat":5,
        "sched_factor":0.4,
        
        "min_lr":6e-10,
        
        "weight_exp":1,
        "gamma":2.5,
        
        "criterion": "cross_entropy",
        "scheduler": "cosine_annealing", 
        "t_max": 6,
        "max_norm":5,
    
        ## MultiModal Model 🍏🍏
        'hidden_dim': 64,
        'fusion': 'avg',  # 'avg', 'concat'

        ## Training
        'batch_size': 32,
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01,
        'epochs': 5,
        'early_stopping_patience': 4,
        'scheduler': 'ReduceLROnPlateau',  # 'ExponentialLR', 'ReduceLROnPlateau'
        'gamma': 0.9,  # for ExponentialLR scheduler
        'factor': 0.3,  # for ReduceLROnPlateau scheduler
        'patience': 3,  # for ReduceLROnPlateau scheduler
        'warmup': False,
        'warmup_epochs': 10,
    }


def get_config(trial):
    config = {
        'project': 'AppleCiDEr',
        'config_from': None,
        'random_seed': 42,  # 42, 66, 0, 12, 123
        'use_wandb': True,
        'save_weights': True,
        'weights_path': f'/projects/bcrv/abrown3/AppleCiDEr_Skyportal/cider_weights/multi-{datetime.now().strftime("%Y-%m-%d-%H-%M")}',
        'use_pretrain': None,
        'freeze': False,        
        
        'step': 'type',
        'classes': ['SN I','SN II', 'Cataclysmic', 'AGN', 'Tidal Disruption Event'],
        'group_labels': True,
        'num_classes': 5,
        #'max_samples': 5500,
        
        ## Data General
        'preprocessed_path': '/work/nvme/bcrv/abrown3/preprocessed_data/data_multi/day10',
        # for ALERTS 
        'train_csv_path':'/projects/bcrv/abrown3/AppleCiDEr_csv/AppleCider_Train_vetted_7-6.csv',
        'val_csv_path':'/projects/bcrv/abrown3/AppleCiDEr_csv/AppleCider_Val_vetted_7-6.csv',
        'test_csv_path':'/projects/bcrv/abrown3/AppleCiDEr_csv/AppleCider_Test_vetted_7-6.csv',
         # for SPECTRA
        'spec_dir': '/work/nvme/bcrv/mxu11',
        #'class_weights': False,
        #'generate_train_val_files': False,
        
        ## personal tag for weights files in local folder 
        'custom_weight_path': False,
        'custom_weight_name': '-',
        
        ## only for the wandb users.....
        'use_notes_tags': True,
        'wandb_tags': ['Delta!', 'hidden dims version', 'group labels'],
        'wandb_notes': f' added back hidden dim config',

        ## Photometry Model 🍏🍏
        'photo_event_path': '/work/hdd/bcrv/ffontinelenunes/data/AppleCider/photo_events',
        
        'output_dir':'/work/hdd/bcrv/ffontinelenunes/data/AppleCider/photo_events',
        'stats_file':'/work/hdd/bcrv/ffontinelenunes/data/AppleCider/photo_events/feature_stats_day100.npz',
        'horizon_days' :  10.0, # <- fine-tuning on 50 days
        #'batch_size'   :  256, #64,
        'sampler_balance': False,
        'num_workers'    : 0,
        # model stuff
        'p_d_model': 128,
        'p_n_heads': 8,       #4,
        'p_n_layers': 4,    #2,
        'p_dropout': 0.30,
        'max_len': 257,     #300,#256,#128,#128,
        #'lr' : 5e-6,
        # 'weight_decay' : 1e-2,
        # 'focal_gamma':2.0,
        #'cut_time_p':None, #(.25,.25,.25,.25), #None,  # or (.25,.25,.25,.25)
        #'p_dropout':0.1,
        #'jitter_scale':0.10,
        #'flux_nu':8,    
        # training schedule
        #'epochs' :150,
        #'patience' :30,
        # misc 
        #'seed':42,
        #'NUM_CLASSES':5,

        ## Spectra Model 🍏🍏
        #"learning_rate",
        #"weight_decay": 1e-5,
        #"ema_decay": 0.995,
        #"focal_loss_gamma": 2.0,
        #"warmup_epochs": 10,
        "T_0": 5,
        "T_mult": 1,
        "eta_min": 1e-5,
        "start_factor": 1e-6,
        "end_factor": 1.0,
        #"num_workers": 0,
        #"sampling": False,
        #"epochs": 100,
        #"optimizer": "AdamW",
        #"patience": 14,
        "train_dir": '/work/nvme/bcrv/mxu11',  "val_dir": '/work/nvme/bcrv/mxu11', "test_dir":'/work/nvme/bcrv/mxu11',  
        
        #'s_dropout': 0.2,
        #'s_conv_channels': [1, 64, 64, 32, 32],
        #'s_kernel_size': 3,
        #'s_mp_kernel_size': 4,

        ## Metadata & Image Model 🍏🍏
        "num_experts":4,
        "towers_hidden_dims":16,
        "towers_outdims":4,
        "embedding":"False",
        
        "fusion_hidden_dims":128,
        "fusion_router_dims":128,
        "fusion_outdims":4,
        
        "cnn_lr":2,
        "cnn_decay":5e-2,
        "psf_lr":0.5,
        "psf_decay":5e-2,
        "mag_lr": 2,
        "mag_decay": 0.0,
        
        "lc_lr": 2,
        "lc_decay": 0.05,
        "spatial_lr":2,
        "spatial_decay":0.0,
        
        "coord_lr": 0.5,
        "coord_decay": 0.0,
        
        "nst1_lr":2,
        "nst1_decay":0.0,
        "nst2_lr":2,
        "nst2_decay":0.0,
        
        "fusion_lr":1,
        "fusion_decay":1e-2,
        "fusion_beta1": 0.9,
        "fusion_beta2": 0.999,
        
        "router_decay":0.0,
        "router_lr":1.5,
        "router_beta1":0.9,
        "router_beta2":0.999,
        "router_lr_2":1,
        "router_beta1_2":0.95,
        "router_beta2_2":0.99,
        
        "classifier_decay": 0.0,
        "classifier_lr":2.44,
        "classifier_beta1":0.95,
        "classifier_beta2":0.99,
        
        "beta1":0.9,
        "beta2":0.999,
        "eps":5e-10,
        
        "sched_pat":5,
        "sched_factor":0.4,
        
        "min_lr":6e-10,
        
        "weight_exp":1,
        "gamma":2.5,
        
        "criterion": "cross_entropy",
        "scheduler": "cosine_annealing", 
        "t_max": 6,
        "max_norm":5,
    
        ## MultiModal Model 🍏🍏
        'hidden_dim': 64,
        'fusion': 'avg',  # 'avg', 'concat'

        ## Training
        'batch_size': 32,
        'lr': 0.001,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01,
        'epochs': 5,
        'early_stopping_patience': 4,
        'scheduler': 'ReduceLROnPlateau',  # 'ExponentialLR', 'ReduceLROnPlateau'
        'gamma': 0.9,  # for ExponentialLR scheduler
        'factor': 0.3,  # for ReduceLROnPlateau scheduler
        'patience': 3,  # for ReduceLROnPlateau scheduler
        'warmup': False,
        'warmup_epochs': 10,
    }

    if STUDY_NAME.startswith('photo'):
        config['mode'] = 'photo'
        config['epochs'] = 50
    elif STUDY_NAME.startswith('spectra'):
        config['mode'] = 'spectra'
        config['epochs'] = 50
    elif STUDY_NAME.startswith('meta'):
        config['mode'] = 'meta'
        config['epochs'] = 50
    elif STUDY_NAME.startswith('image'):
        config['mode'] = 'image'
        config['epochs'] = 50
        
    elif STUDY_NAME.startswith('ztf'):
        config['mode'] = 'ztf'
        config['epochs'] = 50
        
    elif STUDY_NAME.startswith('all'):
        config['mode'] = 'all'
        config['epochs'] = 50
    
    else:
        raise NotImplementedError(f"Unknown study name {STUDY_NAME}")

    if config['mode'] in ('photo', 'ztf', 'all'):
        config['p_dropout'] = trial.suggest_float('p_dropout', 0.0, 0.4)


    config['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    config['factor'] = trial.suggest_float('factor', 0.1, 1.0)
    config['beta1'] = trial.suggest_float('beta1', 0.7, 0.99)
    config['weight_decay'] = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)

    config['study_name'] = STUDY_NAME

    return config



device = torch.device("cuda")

class XastroMiNN(nn.Module):
    """
    Image and Metadata transient classifier
    """
    
    def __init__(self, num_classes=5, num_mlp_experts=4, towers_hidden_dims = 16,
                 towers_outdims = 32,
                 fusion_hidden_dims = 128,
                 fusion_router_dims = 128,
                 fusion_outdims = 32, config=None
                 ):
        super().__init__()
        
        #self.classification = True if config['mode'] == 'metdata & images' else False
        
        self.has_image = True  # Flag for image availability
        self.num_classes = num_classes
        self.towers_hidden_dims = towers_hidden_dims
        self.towers_outdims = towers_outdims

        self.fusion_hidden_dims = fusion_hidden_dims  # was 1024
        self.fusion_router_dims = fusion_router_dims # was 256
        self.fusion_outdims = fusion_outdims


        # ===== Metadata Processing Towers ===== 
        # Each tower processes specific metadata features
        # PSF quality features tower 
        self.psf_tower = ResidualTowerBlock(2, self.towers_hidden_dims, towers_outdims)
    
        # Magnitude features tower 
        self.mag_tower = ResidualTowerBlock(7, self.towers_hidden_dims*2, towers_outdims)
        
        # LC features tower
        self.lc_tower = ResidualTowerBlock(12, self.towers_hidden_dims*3, towers_outdims)

        # Spatial features tower (distpsnr1, distpsnr2, nmtchps)
        self.spatial_tower = ResidualTowerBlock(3, self.towers_hidden_dims, towers_outdims)

        # Nearest source features tower 1 (sgscore1, distpsnr1)
        self.nst1_tower = ResidualTowerBlock(2, self.towers_hidden_dims, fusion_outdims)
        # self.mega_tower = ResidualTowerBlock(7, 64, self.towers_outdims)
        # Nearest source features tower 2 (sgscore2, distpsnr2)
        self.nst2_tower = ResidualTowerBlock(2, self.towers_hidden_dims, fusion_outdims)

        self.coord_tower = ResidualTowerBlock(2, self.towers_hidden_dims, fusion_outdims)

        self.mega_tower = ResidualTowerBlock(19, 128, towers_outdims)

        # ===== Image Processing =====
        # self.image_tower = timm.create_model(
        #     'vit_tiny_patch16_224',
        #     pretrained=True,
        #     num_classes=32,
        #     img_size=49,
        #     in_chans=4
        # )
        
        self.image_tower = SplitHeadConvNeXt(
                    pretrained=False,       # or False if training from scratch
                    in_chans=4,             # Critical: override default 3-channel input
                    outdims=towers_outdims  # Your task's number of classes
                ).to(device)

        fusion_dims = 6*towers_outdims + 3*fusion_outdims 
        # ===== Modality Fusion MoE ===== 
        # Combines features from all towers (4 metadata + image)
        self.fusion_experts = nn.ModuleList([
            ResidualTowerBlock(fusion_dims, fusion_hidden_dims, 5)
            for _ in range(num_mlp_experts)
        ])

        num_experts=num_mlp_experts 
        self.fusion_router = nn.Sequential(
            nn.Linear( fusion_dims, fusion_dims//2),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dims//2, num_experts),
            # nn.Softmax(dim=-1)  # each expert gets independent [0,1] weight
            nn.Sigmoid()
        )

    def forward(self, metadata, image):
        '''    Processes input metadata and optional image data through specialized towers,
            combines features using a Mixture of Experts (MoE) approach, and returns
            classification logits with expert weighting information.

            Parameters:
            -----------
            metadata : torch.Tensor
                Input metadata tensor of shape (batch_size, num_metadata_features).
                Expected to contain 24 metadata features (indices 0-23).
            image : torch.Tensor, optional
                Input image tensor of shape (batch_size, channels, height, width).
                If None, image features will be zero-initialized.

            Returns:
            --------
            dict
                Dictionary containing:
                - 'logits': torch.Tensor
                    Output classification logits of shape (batch_size, num_classes)
                - 'expert_weights': torch.Tensor
                    Raw fusion weights for all experts
                - 'fusion_weights': torch.Tensor
                    Same as expert_weights (maintained for compatibility)'''

        # Process all metadata features through respective towers
        psf_feats = self.psf_tower(metadata[:, [5,14]])  # PSF features
        lc_feats = self.lc_tower(metadata[:, [6, 9, 10, 13, 15, 17, 18, 19, 20, 21, 22, 23]])
        mag_feats = self.mag_tower(metadata[:, [6, 9, 10, 13, 15, 17, 18]]) 

        spatial_feats = self.spatial_tower(metadata[:, [2,3,4]])  # Spatial features
        nsta = self.nst1_tower(metadata[:, [0,2]])  # Nearest source A features
        nstb = self.nst2_tower(metadata[:, [1,3]])  # Nearest source B features
        coord_feats = self.coord_tower(metadata[:, [7,8]])
        megatower = self.mega_tower(metadata[:, [0,1,2,3,4,5,6,7,8,9,10,11,12, 13, 14,15, 16, 17, 18]])
        
        # Process image if available (zeros otherwise)
        image_feats = self.image_tower(image) if image is not None else torch.zeros_like(nsta)

        # Concatenate all features for fusion
        all_feats = torch.cat([nsta, nstb, spatial_feats, psf_feats, mag_feats, coord_feats, megatower, image_feats, lc_feats], dim=1)

        # Fusion MoE - combine features from all modalities
        fusion_weights = self.fusion_router(all_feats)

        moe_output = torch.zeros(metadata.size(0), 5, device='cuda') # ,device=metadata.to(device))

        topk_weights, topk_indices = torch.topk(fusion_weights, k=2, dim=-1)  # [B, k]

        # Process only through selected experts
        for expert_idx, expert in enumerate(self.fusion_experts):
            # Mask for samples where this expert is in top-k
            expert_mask = (topk_indices == expert_idx).any(dim=-1)  # [B]   # 'ResidualTowerBlock'
            
            if expert_mask.any():
                # Get weights for this expert [M] where M=sum(expert_mask)
                weights = topk_weights[expert_mask, (topk_indices[expert_mask] == expert_idx).nonzero()[:, 1]]
                # Compute expert output only for relevant samples
                expert_out = expert(all_feats[expert_mask])  # [M, num_classes]
                # Weighted contribution
                moe_output[expert_mask] += weights.unsqueeze(-1) * expert_out

        return moe_output

    
def build_spec_model(config):
    class SpectraNetBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_sizes,
                     use_skip=True, use_ln=True, do_pool=False):
            super().__init__()
            self.use_skip = use_skip
            self.do_pool = do_pool
            self.use_ln = use_ln
            self.kernel_sizes = kernel_sizes
            self.k = len(kernel_sizes)
            

            self.convs = nn.ModuleList([
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ])
            self.norm = (
                nn.LayerNorm(out_channels * self.k)
                if use_ln else nn.BatchNorm1d(out_channels * self.k)
            )

            if use_skip:
                self.proj = nn.Conv1d(in_channels, out_channels * self.k, kernel_size=1)

            if do_pool:
                self.pool_max = nn.MaxPool1d(4)
                self.pool_avg = nn.AvgPool1d(4)
                self.pool_min = nn.MaxPool1d(4)

        def forward(self, x):
            residual = self.proj(x) if self.use_skip else None
            x_conv = [conv(x) for conv in self.convs]
            x = torch.cat(x_conv, dim=1)

            if self.use_ln:
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.norm(x)

            if self.use_skip:
                x = residual + x

            x = F.gelu(x)

            if self.do_pool:
                x_max = self.pool_max(x)
                x_avg = self.pool_avg(x)
                x_min = -self.pool_min(-x)
                x = torch.cat([x_max, x_avg, x_min], dim=1)

            return x

    class SpectraClassification(nn.Module):
        def __init__(self, depths=[1, 1, 1, 1, 1]):
            super().__init__()
            self.kernel_sizes_per_stage = [
                [3, 61, 1021 ],  
                [3, 31, 251],            
                [3, 15,	61],                  
                [3,11, 31],                         
                [3,7,13]                      
            ]
            assert len(depths) == len(self.kernel_sizes_per_stage)
            self.classification = True if config['mode'] == 'spectra' else False


            use_ln_stages = [False, False, False, False, True]

            channels = [1, 16, 32, 64, 128, 256]

            def make_stage(in_c, out_c, depth, kernel_sizes, use_ln=True, do_pool=True):
                k = len(kernel_sizes)
                blocks = []
                for i in range(depth):
                    blocks.append(SpectraNetBlock(
                        in_channels=in_c if i == 0 else out_c * k,
                        out_channels=out_c,
                        kernel_sizes=kernel_sizes,
                        use_skip=True,
                        use_ln=use_ln,
                        do_pool=(do_pool if i == depth - 1 else False)
                    ))
                return nn.Sequential(*blocks), k

            self.stage1, k1 = make_stage(channels[0], channels[1], depths[0], self.kernel_sizes_per_stage[0], use_ln=use_ln_stages[0])
            self.stage2, k2 = make_stage(channels[1]*k1*3, channels[2], depths[1], self.kernel_sizes_per_stage[1], use_ln=use_ln_stages[1])
            self.stage3, k3 = make_stage(channels[2]*k2*3, channels[3], depths[2], self.kernel_sizes_per_stage[2], use_ln=use_ln_stages[2])
            self.stage4, k4 = make_stage(channels[3]*k3*3, channels[4], depths[3], self.kernel_sizes_per_stage[3], use_ln=use_ln_stages[3])
            self.stage5, k5 = make_stage(channels[4]*k4*3, channels[5], depths[4], self.kernel_sizes_per_stage[4], use_ln=use_ln_stages[4], do_pool=False)

            self.ks = [k1, k2, k3, k4, k5]
            length = 4096
            for _ in range(4): length //= 4
            self.flat_dim = channels[5] * self.ks[-1] * length

            
            self.class_model = nn.Sequential(
                nn.Linear(self.flat_dim, 2048),
                nn.LayerNorm(2048), nn.GELU(), nn.Dropout(0.5),
                nn.Linear(2048, 256),
                nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3)) #, 
            #    nn.Linear(256, len(config['classes']))
            #)
            
            if self.classification:
                self.fc = nn.Linear(256, len(config['classes']))

        def forward(self, x):
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
            z = x.reshape(x.size(0), -1)
            
            output = self.class_model(z)
            
            if self.classification:
                output = self.fc(output)
          
            return output
    
    
    return SpectraClassification()



# ===============================================================
#    Model  -----------------------------------------------------
# ===============================================================

# ------------------------------------------------------------------------------
# (NEW) Time2Vec positional encoding
# ------------------------------------------------------------------------------
class Time2Vec(nn.Module):
    """
    Time2Vec: map scalar time t -> d_model-dimensional vector.
    v0 = w0 * t + b0  (linear)
    v[i] = sin(w[i] * t + b[i])  for i=1..d_model-1
    """
    def __init__(self, d_model):
        super().__init__()
        # one linear + (d_model-1) periodic features
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w  = nn.Parameter(torch.randn(d_model-1))
        self.b  = nn.Parameter(torch.zeros(d_model-1))

    def forward(self, t):
        """
        t: (B, L)  - scalar "time since first detection" per event
        returns (B, L, d_model)
        """
        # linear term
        v0 = self.w0 * t + self.b0                          # (B, L)
        # periodic terms
        vp = torch.sin(t.unsqueeze(-1) * self.w + self.b)   # (B, L, d_model-1)
        return torch.cat([v0.unsqueeze(-1), vp], dim=-1)    # (B, L, d_model)


class BaselineCLS(nn.Module):
    def __init__(self, d_model, n_heads, n_layers,
                 num_classes, dropout, max_len=None, mode=None):
        super().__init__()
        self.in_proj  = nn.Linear(7, d_model)
        self.cls_tok  = nn.Parameter(torch.zeros(1,1,d_model))
        
        # replace SinCos PE with Time2Vec on the dt channel
        
        self.time2vec = Time2Vec(d_model).to(torch.device("cuda")) # # changed from original, req to work 
        enc_layer      = nn.TransformerEncoderLayer(
                            d_model, n_heads, d_model*4,
                            dropout, batch_first=True)
        self.encoder  = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm     = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, num_classes)
        
        self.classification = True if mode == 'photo' else False
        if self.classification:
            self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, pad_mask):
        """
        x: (B, L, 7)  - the raw event tensor from build_event_tensor
            channels: [ dt, dt_prev, logf, logfe, one-hot-band(3) ]
        pad_mask: (B, L) boolean
        """
        B,L,_ = x.shape

        # project into model dim
        h = self.in_proj(x)                     # (B, L, d_model)
        # extract the *continuous* log1p dt feature
        t = x[..., 0]                           # (B, L)

        # compute the learned time embedding:
        te = self.time2vec(t)                  # (B, L, d_model)

        # add it:
        h = h + te                              # (B, L, d_model)

        # prepend a learned CLS token:
        tok = self.cls_tok.expand(B,-1,-1)      # (B,1,d_model)
        h   = torch.cat([tok, h], dim=1)        # (B, L+1, d_model)
        
        # adjust padding mask to account for CLS at idx=0
        pad = torch.cat(
            [torch.zeros(B,1, device=torch.device("cuda"), dtype=torch.bool),
             pad_mask], dim=1
        ) # changed from original, req to work 

        # encode
        z = self.encoder(h, src_key_padding_mask=pad)  # (B, L+1, d_model)
        
        output = self.norm(z[:,0]) # (B, d_model )
        
        if self.classification:
            # classification from the CLS token
            output = self.fc(output) # (B, num_classes)     
            
        return output
                
        
class AppleCider(nn.Module):
    
    """
    AppleCider
    """
    
    def __init__(self, config):
        super(AppleCider, self).__init__()

        self.classification = True if config['mode'] == 'all' else False

        self.photometry_encoder = BaselineCLS(d_model=config['p_d_model'],
                               n_heads=config['p_n_heads'],n_layers=config['p_n_layers'],
                               num_classes=config['num_classes'],
                               dropout=config['p_dropout'], max_len=config['max_len']).to(device)
        self.spectra_encoder = build_spec_model(config).to(device) # output logits
        self.img_metadata_encoder = XastroMiNN(config)

        self.photometry_proj = nn.Linear(config['p_d_model'], config['hidden_dim'])  # 5)
        self.spectra_proj = nn.Linear(256, config['hidden_dim']) #5)
        self.img_metadata_proj = nn.Linear(5, config['hidden_dim']) #5)
        
        if self.classification:
            self.fusion = config['fusion']
            in_features = config['hidden_dim'] * 3 if self.fusion == 'concat' else config['hidden_dim']
            self.fc = nn.Linear(in_features, config['num_classes'])
        
    def get_embeddings(self, photometry, photometry_mask, metadata, images, spectra):
        p_emb = self.photometry_proj(self.photometry_encoder(photometry, photometry_mask))
        s_emb = self.spectra_proj(self.spectra_encoder(spectra))
        im_emb = self.img_metadata_proj(self.img_metadata_encoder(metadata, images))
        
        ## normalize features
        p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)
        im_emb = im_emb / im_emb.norm(dim=-1, keepdim=True)
        s_emb = s_emb / s_emb.norm(dim=-1, keepdim=True)
        
        return p_emb, im_emb, s_emb

    def forward(self, photometry, photometry_mask, metadata, images, spectra):
        p_emb, im_emb, s_emb = self.get_embeddings(photometry, photometry_mask, metadata, images, spectra)
        
        if self.classification:

            if self.fusion == 'concat':
                emb = torch.cat((p_emb, im_emb, s_emb), dim=1)
            elif self.fusion == 'avg':
                emb = (p_emb + im_emb +  s_emb) / 3
            else:
                raise NotImplementedError
            
            logits = self.fc(emb)

            return logits
        else:
            raise NotImplementedError


            
            
model = AppleCider(config)
model.to(device)


class EarlyStopping:
    def __init__(self, patience=15):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def step(self, metric):
        if self.best_score is None:
            self.best_score = metric
            self.counter = 1
        else:
            if metric < self.best_score:
                self.best_score = metric
                self.counter = 0
            else:
                self.counter += 1
        return self.counter >= self.patience

    
class Trainer:
    def __init__(self, model, optimizer, scheduler, warmup_scheduler, criterion, criterion_val, device, config, trial=None):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.criterion = criterion
        self.criterion_val = criterion_val
        self.device = torch.device("cuda")
        self.trial = trial

        self.mode = config['mode']
        self.save_weights = config['save_weights']
        self.weights_path = config['weights_path']
        self.use_wandb = config['use_wandb']
        self.early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
        self.warmup_epochs = config['warmup_epochs']

        self.total_loss = [] ; self.total_correct_predictions = 0 ; self.total_predictions = 0
        
        self.custom_weight_path = config['custom_weight_path'] #; self.custom_weight_name = config['custom_weight_name']
        
        if self.use_wandb:
            self.run_id = config['run_id']
  
    def store_weights(self, epoch):
        
        if self.use_wandb:
            torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{epoch}-{self.run_id}.pth'))
            torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-best-{self.run_id}.pth'))
        else:
            if self.custom_weight_path:
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{epoch}-{self.custom_weight_name}.pth'))
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-best-{self.custom_weight_name}.pth'))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-{epoch}-.pth'))
                torch.save(self.model.state_dict(), os.path.join(self.weights_path, f'weights-{datetime.now().strftime("%Y-%m-%d-%H-%M")}-best.pth'))

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
        return sum(self.total_loss) / len(self.total_loss), self.total_correct_predictions / self.total_predictions

    def get_logits(self, photometry, photometry_mask, metadata, images, spectra):
        
        if self.mode == 'photo':
            logits = self.model(photometry, photometry_mask)
        else:  # all 4 modalities
            logits = self.model(photometry, photometry_mask, metadata, images,  spectra)

        return logits


    def step(self, photometry, photometry_mask, metadata, images, spectra, labels):
        """Perform a training step for the classification model"""
        logits = self.get_logits(photometry, photometry_mask, metadata, images, spectra)    
        
        loss = self.criterion(logits, labels)

        self.update_stats(loss, logits, labels)

        return loss
    
    def step_val(self, photometry, photometry_mask, metadata, images, spectra, labels):
        """Perform a training step for the classification model"""
        logits = self.get_logits(photometry, photometry_mask, metadata, images, spectra)    
        
        loss = self.criterion_val(logits, labels)
        
        self.update_stats(loss, logits, labels)

        return loss

    def get_gradient_norm(self):
        total_norm = 0.0

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5

    def train_epoch(self, train_dataloader):
        self.model.train()
        self.zero_stats()
        
        for photometry, photometry_mask, metadata, images, spectra, labels in tqdm(train_dataloader, total=len(train_dataloader), desc='Train', colour='#9ACD32',leave=True):
            photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
            metadata, images = metadata.to(self.device), images.to(self.device)
            spectra = spectra.to(self.device)
            labels = labels.to(self.device)    
            
            self.optimizer.zero_grad()

            
            loss = self.step(photometry, photometry_mask, metadata, images, spectra, labels)

            if self.use_wandb:
                wandb.log({'step_loss': loss.item()})

            loss.backward()

            if self.use_wandb:
                grad_norm = self.get_gradient_norm()
                wandb.log({'grad_norm': grad_norm})

            self.optimizer.step()

        loss, acc = self.calculate_stats()

        return loss, acc
   

    def val_epoch(self, val_dataloader):
        self.model.eval()
        self.zero_stats()

        with torch.no_grad():
            for photometry, photometry_mask, metadata, images, spectra, labels in tqdm(val_dataloader, total=len(val_dataloader), desc='Validation', colour='#9ACD32', leave=True):
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                
                metadata, images = metadata.to(self.device), images.to(self.device)
                spectra = spectra.to(self.device)
                labels = labels.to(self.device)

                self.step_val(photometry, photometry_mask, metadata, images, spectra, labels)

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
                    print('Prune')
                    wandb.finish()
                    raise optuna.exceptions.TrialPruned()

            if self.warmup_scheduler and epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
                current_lr = self.warmup_scheduler.get_last_lr()[0]
            else:
                self.scheduler.step(val_loss)
                current_lr = self.scheduler.get_last_lr()[0]

            if self.use_wandb:
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc,
                           'val_acc': val_acc,'learning_rate': current_lr, 'epoch': epoch})

            if best_val_acc < val_acc:
                best_val_acc = val_acc

                if self.use_wandb:
                    wandb.log({'best_val_acc': best_val_acc})

                if self.save_weights:
                    self.store_weights(epoch)

            print(f'Epoch {epoch}: Train Loss {round(train_loss, 4)} \t Val Loss {round(val_loss, 4)} \t \
                    Train Acc {round(train_acc, 4)} \t Val Acc {round(val_acc, 4)}')

            if self.early_stopping.step(val_loss):
                print(f'Early stopping at epoch {epoch}')
                break

        return best_val_loss

    def evaluate(self, val_dataloader, id2target):
        self.model.eval()

        all_true_labels = []
        all_predicted_labels = []
        
        for photometry, photometry_mask, metadata, images, spectra, labels in tqdm(val_dataloader, total=len(val_dataloader), desc='validation', colour='#9ACD32',leave=True):
            with torch.no_grad():
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                spectra = spectra.to(self.device)
                metadata, images = metadata.to(self.device), images.to(self.device)

                logits = self.get_logits(photometry, photometry_mask, metadata, images, spectra)

                probabilities = torch.nn.functional.softmax(logits, dim=1)
                _, predicted_labels = torch.max(probabilities, dim=1)

                #all_true_labels.extend(labels.numpy())
                all_true_labels.extend(labels.cpu().numpy()) 
                all_predicted_labels.extend(predicted_labels.cpu().numpy())
                # all_predicted_labels.extend(predicted_labels.cpu().numpy())

        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        conf_matrix_percent = 100 * conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

        labels = [id2target[i] for i in range(len(conf_matrix))]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
        
        ## Plot absolute values confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='BuPu', xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title('Confusion Matrix - Absolute Values')

        ## Plot percentage values confusion matrix
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.0f', cmap='BuPu', xticklabels=labels, yticklabels=labels,ax=axes[1])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel(' ')
        axes[1].set_title('Confusion Matrix - %')

        if self.use_wandb:
            ## Save both confusion matrix images
            matrix_table = wandb.Table(columns=["SN I","SN II", "CV", "AGN", "TDE"], data=conf_matrix)
            wandb.log({'conf_matrix': wandb.Image(fig)})
            ## Save absolute confusion matrix as table
            artifact_matrix = wandb.Artifact("confusion-matrix", type="table")
            artifact_matrix.add(matrix_table, "absolute-confusion-matrix")
            wandb.log_artifact(artifact_matrix)
         
        return conf_matrix
    
    def evaluate_alert(self, alert_dataloader, id2target):
        self.model.eval()
    
        all_true_labels = []
        all_predicted_labels = []
        
        
        for photometry, photometry_mask, metadata, images, spectra, labels in tqdm(alert_dataloader, total=len(alert_dataloader), desc='alert evaluation', colour='#9ACD32',leave=True):
            with torch.no_grad():
                
                photometry, photometry_mask = photometry.to(self.device), photometry_mask.to(self.device)
                metadata, images = metadata.to(self.device), images.to(self.device)
                spectra = spectra.to(self.device)

                logits = self.get_logits(photometry, photometry_mask, metadata, images, spectra)
                
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                _, predicted_labels = torch.max(probabilities, dim=1)
                
                all_true_labels.extend(labels.cpu().numpy()) 
                all_predicted_labels.extend(predicted_labels.cpu().numpy())

                
        #print("all_predicted_label",all_predicted_label)
        #print("all_predicted_label",all_predicted_label)
        
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        conf_matrix_percent = 100 * conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

        labels = [id2target[i] for i in range(len(conf_matrix))]
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
        
        ## Plot absolute values confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='BuPu', xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title('Confusion Matrix - Absolute Values')

        ## Plot percentage values confusion matrix
        sns.heatmap(conf_matrix_percent, annot=True, fmt='.0f', cmap='BuPu', xticklabels=labels, yticklabels=labels,ax=axes[1])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title('Confusion Matrix - %')
        
        

        if self.use_wandb:

            
            ## Save both confusion matrix images
            wandb.log({'conf_matrix': wandb.Image(fig)})
            
            matrix_table = wandb.Table(columns=["SN I","SN II", "CV", "AGN", "TDE"], data=conf_matrix)
            
            ## Save absolute confusion matrix as table
            artifact_matrix = wandb.Artifact("confusion-matrix", type="table")
            artifact_matrix.add(matrix_table, "absolute-confusion-matrix")
            wandb.log_artifact(artifact_matrix)
         
        return conf_matrix
    
    
    
def run(config, trial, add_notes=None, tags_list=None):
    
    train_dataset = CiderDataset(config, split='train')
    val_dataset = CiderDataset(config, split='val')

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, collate_fn=collate)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate)

    device = torch.device("cuda") # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using', device)

    model = AppleCider(config)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    scheduler, warmup_scheduler = get_schedulers(config, optimizer)
    
    #if config['class_weights']:
    
        #if os.path.isfile(config['class_weights_path']):
        #    with open(config['class_weights_path'], 'rb') as file:
        #        weights = pickle.load(file)
        #    weight_sorted = dict(sorted(weights.items(), key=lambda item: item))
        #    weight_sorted_list = list(weight_sorted.values())
        #    weight_tensor = torch.tensor(weight_sorted_list, dtype=torch.float32)
        #    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        #    criterion_val = torch.nn.CrossEntropyLoss()
        #    
        #else:
        #    raise ValueError(f'class weights=True, but no class weight file at config[class_weights_path] exists')
        
    #else:
    criterion = torch.nn.CrossEntropyLoss()
    criterion_val = torch.nn.CrossEntropyLoss()
    
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, warmup_scheduler=warmup_scheduler, criterion=criterion, criterion_val = criterion_val, device=device, config=config, trial=trial)
    best_val_loss = trainer.train(train_dataloader, val_dataloader, epochs=config['epochs'])

    if config['mode'] != 'clip':
        trainer.evaluate(val_dataloader, id2target=val_dataset.id2target)

    return best_val_loss


def get_schedulers(config, optimizer):
    if config['scheduler'] == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=config['gamma'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'])
    else:
        raise NotImplementedError(f"Scheduler {config['scheduler']} not implemented")

    if config['warmup']:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1, total_iters=config['warmup_epochs'])
    else:
        warmup_scheduler = None

    return scheduler, warmup_scheduler


def set_random_seeds(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic 
    
    
import wandb
wandb.login(key='2441c7228ac0ac87db25beba7ace3e6dbad8ec21')

WANDB_NOTEBOOK_NAME = 'CIDER.ipynb'


def objective(trial):
    config = get_config(trial)
    set_random_seeds(config['random_seed'])

    if config['use_wandb']:
        wandb_run = wandb.init(project='AppleCiDEr',
                               config=config, group=STUDY_NAME,
                               reinit=True,
                               notes=config['wandb_notes'],
                               tags=config['wandb_tags'])
        config.update(wandb.config)                           # i dont think this proper updates config file lmao
        config['run_id'] = wandb_run.id
        config['weights_path'] += f'-{wandb_run.id}'
        wandb_run.config['weights_path'] = config['weights_path']   # try to update config file? again
        print(config['weights_path'])
        wandb_run.config['run_id'] = wandb_run.id

    if config['save_weights']:
        os.makedirs(config['weights_path'], exist_ok=True)
        #update config file? 

    best_val_loss = run(config, trial)
    wandb.finish()

    return best_val_loss

if __name__ == '__main__':
    STUDY_NAME = 'all'
    
    try:
        study = optuna.create_study(study_name=STUDY_NAME, direction='minimize', pruner=optuna.pruners.NopPruner())
        print(f"Study '{STUDY_NAME}' created.")
    except DuplicatedStudyError:
        study = optuna.load_study(study_name=STUDY_NAME, pruner=optuna.pruners.NopPruner())
        print(f"Study '{STUDY_NAME}' loaded.")

    study.optimize(objective, n_trials=10)