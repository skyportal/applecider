import torch
import torch.nn as nn
import torch.nn.functional as



class MetaModel(nn.Module):
    
    """
    Metadata model from AstroM3
    Paper link: https://arxiv.org/abs/2411.08842
    """""
    
    
    def __init__(self, config):
        super(MetaModel, self).__init__()

        self.classification = True if config['mode'] == 'meta' else False
        self.input_dim = len(config['meta_cols'])
        self.hidden_dim = config['m_hidden_dim']
        self.dropout = config['m_dropout']

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        if self.classification:
            self.fc = nn.Linear(self.hidden_dim, config['num_classes'])

    def forward(self, x):
        
        x = self.model(x)

        if self.classification:
            x = self.fc(x)

        return x
