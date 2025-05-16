import torch
import torch.nn as nn



def get_multimodal_model(config):
    if config['mode'] == 'ztf':
        model = ZwickyCider(config)   
    elif config['mode'] == 'all':
        model = AppleCider(config)    
    elif config['mode'] == 'acai':
        #model = Acai(config)
        print('Not implemented! Come back Later')
    else:
        raise ValueError(f'Please select a multimodal model! Try again!')
    return model


def photo_model(config):
    """ photometry model from config """
    if config['photometry_model'] == 'Informer':
        from AppleCider.models.Informer import Informer
        model = Informer(config)
        
    elif config['photometry_model'] == 'TransformerClassifier':
        from AppleCider.models.TransformerClassifier import TransformerClassifier
        model = TransformerClassifier(config)
        
    else:
        raise ValueError(f'Please select a photometry model! \nex: Informer, TransformerClassifier')
    return model

def image_model(config):
    """ image model from config """
    if config['image_model'] == 'BTSModel':
        from AppleCider.models.BTSModel import BTSModel
        model = BTSModel(config)
        
    else:
        raise ValueError(f'Please select an image model! \nex: BTSModel')
    return model

def meta_model(config):
    """ metadata model from config """
    
    if config['meta_model'] == 'MetaModel':
        from AppleCider.models.MetaModel import MetaModel
        model = MetaModel(config)
        
    else:
        raise ValueError(f'Please select a metadata model! \nex: MetaModel')
    return model

def spectra_model(config):
    """ spectra model from config """
    if config['spectra_model'] == 'GalSpecNet':
        from AppleCider.models.GalSpecNet import GalSpecNet
        model = GalSpecNet(config)
        
    elif config['spectra_model'] == 'SpectraConvNeXtBase':
        from AppleCider.models.SpectraConvNeXtBase import SpectraConvNeXtBase
        model = SpectraConvNeXtBase(config)
        
    elif config['spectra_model'] == 'SpectraEfficientNetV2L':
        from AppleCider.models.SpectraEfficientNetV2L import SpectraEfficientNetV2L
        model = SpectraEfficientNetV2L(config)
        
    ## TODO: add other spec models
    # EfficientNetV2L
    # ViTBase
    else:
        raise ValueError(f'Please select a spectra model! \nex: GalSpecNet, SpectraConvNeXtBase, SpectraEfficientNetV2L.')
    return model



class ZwickyCider(nn.Module):
    """
    ZwickyCider = photometry model, image model, metadata model
    """
    
    def __init__(self, config):
        super(ZwickyCider, self).__init__()

        self.classification = True if config['mode'] == 'ztf' else False
        
        self.photometry_encoder = photo_model(config) 
        self.image_encoder = image_model(config)      
        self.metadata_encoder = meta_model(config)    
        
        self.photometry_proj = nn.Linear( config['seq_len'] * config['p_d_model'], config['hidden_dim'])
        self.image_proj = nn.Linear(784, config['hidden_dim'])
        self.metadata_proj = nn.Linear(config['m_hidden_dim'], config['hidden_dim'])

        if self.classification:
            self.fusion = config['fusion']
            in_features = config['hidden_dim'] * 3 if self.fusion == 'concat' else config['hidden_dim']
            self.fc = nn.Linear(in_features, config['num_classes'])

    def get_embeddings(self, photometry, photometry_mask, images, metadata):
    
        p_emb = self.photometry_proj(self.photometry_encoder(photometry, photometry_mask))
        i_emb = self.image_proj(self.image_encoder(images))
        m_emb = self.metadata_proj(self.metadata_encoder(metadata))

        # normalize features
        p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)
        i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
        m_emb = m_emb / m_emb.norm(dim=-1, keepdim=True)
        
        return p_emb, i_emb, m_emb

    def forward(self, photometry, photometry_mask, images, metadata):
        
        p_emb, i_emb, m_emb = self.get_embeddings(photometry, photometry_mask, images, metadata)

        if self.classification:

            if self.fusion == 'concat':
                emb = torch.cat((p_emb, i_emb, m_emb), dim=1)
            elif self.fusion == 'avg':
                emb = (p_emb + i_emb + m_emb) / 3
            else:
                raise NotImplementedError

            logits = self.fc(emb)

            return logits

            
class AppleCider(nn.Module):
    
    """
    AppleCider = photometry model, 
                 image model,
                 metadata model,
                 spectra model
    """
    
    def __init__(self, config):
        super(AppleCider, self).__init__()

        self.classification = True if config['mode'] == 'all' else False

        self.photometry_encoder = photo_model(config) 
        self.image_encoder = image_model(config)      
        self.metadata_encoder = meta_model(config)    
        self.spectra_encoder = spectra_model(config)  

        self.photometry_proj = nn.Linear( config['seq_len'] * config['p_d_model'], config['hidden_dim'])
        self.metadata_proj = nn.Linear(config['m_hidden_dim'], config['hidden_dim'])
        self.image_proj = nn.Linear(784, config['hidden_dim']) 
        self.spectra_proj = nn.Linear(1632, config['hidden_dim'])

        if self.classification:
            self.fusion = config['fusion']
            in_features = config['hidden_dim'] * 4 if self.fusion == 'concat' else config['hidden_dim']
            self.fc = nn.Linear(in_features, config['num_classes'])

    def get_embeddings(self, photometry, photometry_mask, images, metadata, spectra):
    
        p_emb = self.photometry_proj(self.photometry_encoder(photometry, photometry_mask))
        i_emb = self.image_proj(self.image_encoder(images))
        m_emb = self.metadata_proj(self.metadata_encoder(metadata))
        s_emb = self.spectra_proj(self.spectra_encoder(spectra))
        
        # normalize features
        p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)
        i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
        m_emb = m_emb / m_emb.norm(dim=-1, keepdim=True)
        s_emb = s_emb / s_emb.norm(dim=-1, keepdim=True)
        
        return p_emb, i_emb, m_emb, s_emb

    def forward(self, photometry, photometry_mask, images, metadata, spectra):
        
        p_emb, i_emb, m_emb, s_emb = self.get_embeddings(photometry, photometry_mask, images, metadata, spectra)

        if self.classification:

            if self.fusion == 'concat':
                emb = torch.cat((p_emb, i_emb, m_emb, s_emb), dim=1)
            elif self.fusion == 'avg':
                emb = (p_emb + i_emb + m_emb + s_emb) / 4
            else:
                raise NotImplementedError

            logits = self.fc(emb)

            return logits
        
        
        
#class Acai(nn.Module):
#    """
#    Acai
#    """
#    
#    def __init__(self, config):
#        super(ZwickyCider, self).__init__()
#
#        self.classification = True if config['mode'] == 'ztf' else False
#        
#        self.photometry_encoder = photo_model(config)
#        self.im_encoder = acai_model(config)
#        self.spectra_encoder = spectra_model(config)
#        
#        self.photometry_proj = nn.Linear( config['seq_len'] * config['p_d_model'], config['hidden_dim'])
#        self.im_proj = nn.Linear(784, config['hidden_dim'])
#        self.spectra_proj = nn.Linear(1632, config['hidden_dim'])
#
#
#        if self.classification:
#            self.fusion = config['fusion']
#            in_features = config['hidden_dim'] * 3 if self.fusion == 'concat' else config['hidden_dim']
#            self.fc = nn.Linear(in_features, config['num_classes'])
#
#    def get_embeddings(self, photometry, photometry_mask, im, spectra):
#    
#        p_emb = self.photometry_proj(self.photometry_encoder(photometry, photometry_mask))
#        im_emb = self.image_proj(self.image_encoder(images, metadata))
#        s_emb = self.spectra_proj(self.spectra_encoder(spectra))
#
#        # normalize features
#        p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)
#        im_emb = im_emb / im_emb.norm(dim=-1, keepdim=True)
#        
#        return p_emb, im_emb, m_emb
#
#    def forward(self, photometry, photometry_mask, images, metadata):
#        
#        p_emb, i_emb, m_emb = self.get_embeddings(photometry, photometry_mask, images, metadata)
#
#        if self.classification:
#
#            if self.fusion == 'concat':
#                emb = torch.cat((p_emb, i_emb, m_emb), dim=1)
#            elif self.fusion == 'avg':
#                emb = (p_emb + i_emb + m_emb) / 3
#            else:
#                raise NotImplementedError
#
#            logits = self.fc(emb)
#
#            return logits
