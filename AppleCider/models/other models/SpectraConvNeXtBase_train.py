import os
import gc
import sys
import math
from math import sqrt
import numpy as np

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectraConvNeXtBase(nn.Module):
    
    def __init__(self, config):
        super(SpectraConvNeXtBase, self).__init__()
            
        self.classification = True if config['mode'] == 'spectra' else False
        
        self.dropout_rate = config['s_dropout']
        self.s_dim = config['s_dim']
        
        self.backbone = timm.create_model('convnext_base', pretrained=True, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, self.s_dim),
            
            nn.BatchNorm1d(self.s_dim),
            
            nn.ReLU(inplace=True),
            
            nn.Dropout(self.dropout_rate),
            
            nn.Linear(self.s_dim, 256),
            
            nn.BatchNorm1d(256),
            
            nn.ReLU(inplace=True),
            
            nn.Dropout(self.dropout_rate),
            
            #nn.Linear(256, num_classes)
        )
        
        if self.classification:
            self.fc = nn.Linear(256, config['num_classes'])
            

    def forward(self, x):
        x = self.backbone(x)
        output = self.classifier(x)
            
        if self.classification:
            output = self.fc(output)
            
        return output





