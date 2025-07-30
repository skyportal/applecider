import torch
import numpy as np
from AppleCider.models.AstroMiNN import AstroMiNN
from AppleCider.models.SpectraNet import build_spec_model
from AppleCider.models.BaselineCLS import BaselineCLS

class AppleCider(nn.Module):
    
    """
    🍏 AppleCider: Photometry (BaselineCLS) + Images & Metadata (astroMiNN) + Spectra (SpectraNetBlock)
    
    Paper I: arxiv.org/abs/2507.16088v2 🍏
    """
    
    def __init__(self, config):
        super(AppleCider, self).__init__()

        self.classification = True if config['mode'] == 'all' else False

        self.photometry_encoder = BaselineCLS(d_model=config['p_d_model'],
                               n_heads=config['p_n_heads'],n_layers=config['p_n_layers'],
                               num_classes=config['num_classes'],
                               dropout=config['p_dropout'], max_len=config['max_len']).to(device)
        self.spectra_encoder = build_spec_model(config).to(device) # output logits
        self.img_metadata_encoder = astroMiNN(config)
        
        self.photometry_proj = nn.Linear(config['p_d_model'], 5)
        self.spectra_proj = nn.Linear(256, 5)
        
        if self.classification:
            self.fusion = config['fusion']
            in_features = 5 * 3 if self.fusion == 'concat' else 5
            self.fc = nn.Linear(in_features, config['num_classes'])
        
    def get_embeddings(self, photometry, photometry_mask, metadata, images, spectra):

        p_emb = self.photometry_proj(self.photometry_encoder(photometry, photometry_mask))
        s_emb = self.spectra_proj(self.spectra_encoder(spectra))
        im_emb = self.img_metadata_encoder(metadata, images)
        
        ## normalize features
        p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)
        s_emb = s_emb / s_emb.norm(dim=-1, keepdim=True)
        im_emb = im_emb / im_emb.norm(dim=-1, keepdim=True)
        

        return p_emb, im_emb, s_emb


    def forward(self, photometry, photometry_mask, metadata, images, spectra):

        p_emb, im_emb, s_emb = self.get_embeddings(photometry, photometry_mask, metadata, images, spectra)
        
        if self.classification:

            if self.fusion == 'concat':
                emb = torch.cat((p_emb, s_emb, im_emb), dim=1)
            elif self.fusion == 'avg':
                emb = (p_emb + s_emb + im_emb) / 3
            else:
                raise NotImplementedError
            
            logits = self.fc(emb)

            return logits
        else:
            raise NotImplementedError
