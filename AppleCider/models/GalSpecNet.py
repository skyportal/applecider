import math
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class GalSpecNet(nn.Module):
    
    """
    GalSpecNet
    Paper link: https://doi.org/10.1093/mnras/stad2913
    """
    
    def __init__(self, config):
        super(GalSpecNet, self).__init__()

        self.classification = True if config['mode'] == 'spectra' else False
        self.dropout_rate = config['s_dropout']
        self.conv_channels = config['s_conv_channels']
        self.kernel_size = config['s_kernel_size']
        self.mp_kernel_size = config['s_mp_kernel_size']

        self.layers = nn.ModuleList([])

        for i in range(len(self.conv_channels) - 1):
            self.layers.append(
                nn.Conv1d(self.conv_channels[i], self.conv_channels[i + 1], kernel_size=self.kernel_size)
            )
            self.layers.append(nn.ReLU())

            if i < len(self.conv_channels) - 2:  # Add MaxPool after each Conv-ReLU pair except the last
                self.layers.append(nn.MaxPool1d(kernel_size=self.mp_kernel_size))

        self.dropout = nn.Dropout(self.dropout_rate)

        if self.classification:
            self.fc = nn.Linear(1632, config['num_classes'])

    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)

        if self.classification:
            x = self.fc(x)

        return x