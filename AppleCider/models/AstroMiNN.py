import torch
import torch.nn as nn



class AstroMiNN(nn.Module):
    
    """
    Image and Metadata transient classifier
    Paper: arxiv.org/abs/2507.16088v2
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

        spatial_feats = self.spatial_tower(metadata[:, [2,3,4]])   # Spatial features
        nsta = self.nst1_tower(metadata[:, [0,2]])                 # Nearest source A features
        nstb = self.nst2_tower(metadata[:, [1,3]])                 # Nearest source B features
        coord_feats = self.coord_tower(metadata[:, [7,8]])
        megatower = self.mega_tower(metadata[:, [0,1,2,3,4,5,6,7,8,9,10,11,12, 13, 14,15, 16, 17, 18]])
        
        # Process image if available (zeros otherwise)
        image_feats = self.image_tower(image) if image is not None else torch.zeros_like(nsta)

        # Concatenate all features for fusion
        all_feats = torch.cat([nsta, nstb, spatial_feats, psf_feats, mag_feats, coord_feats, megatower, image_feats, lc_feats], dim=1)

        # Fusion MoE - combine features from all modalities
        fusion_weights = self.fusion_router(all_feats)

        moe_output = torch.zeros(metadata.size(0), 5, device='cuda')

        topk_weights, topk_indices = torch.topk(fusion_weights, k=2, dim=-1)  # [B, k]

        # Process only through selected experts
        for expert_idx, expert in enumerate(self.fusion_experts):
            # Mask for samples where this expert is in top-k
            expert_mask = (topk_indices == expert_idx).any(dim=-1)  # [B]  # 'ResidualTowerBlock'
            
            if expert_mask.any():
                # Get weights for this expert [M] where M=sum(expert_mask)
                weights = topk_weights[expert_mask, (topk_indices[expert_mask] == expert_idx).nonzero()[:, 1]]
                # Compute expert output only for relevant samples
                expert_out = expert(all_feats[expert_mask])  # [M, num_classes]
                # Weighted contribution
                moe_output[expert_mask] += weights.unsqueeze(-1) * expert_out

        return moe_output
