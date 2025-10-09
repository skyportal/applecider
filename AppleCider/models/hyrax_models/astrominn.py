import numpy as np
import timm
import torch
import torch.nn as nn

from hyrax.models import hyrax_model

class SplitHeadConvNeXt(nn.Module):
    def __init__(self, pretrained=False, in_chans=4, outdims=4):
        super().__init__()
        # Load base model (disable default head)
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0 # Disable original classifier
        )

        # Get feature dimension (e.g., 768 for 'convnext_tiny')
        features = self.backbone.num_features

        # Define two separate heads
        self.head_main = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(features),
            nn.Linear(features, features//2), # 4-dimensional output
            nn.ReLU(),
            nn.Dropout(.4),
            nn.Linear(features//2, features),
            nn.Linear( features, outdims)
        )

        self.head_aux = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, outdims),
            nn.Tanh()
        ) # 1-dimensional output

    def forward(self, x):
        features = self.backbone(x) # (batch_size, features)
        out = self.head_main(features) # (batch_size, 4)
        activation = self.head_aux(features) # (batch_size, 1)
        return out * activation


class ResidualTowerBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.start_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )

        self.main_path = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(.25),
            nn.Linear(hidden_dim, output_dim))

        self.activation = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(.25),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

        self.skip_path = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        first_half = self.start_path(x)
        gating = self.activation(first_half)
        out = self.main_path(first_half) * gating + self.skip_path(x)

        return out


@hyrax_model
class AstroMiNN(nn.Module):
    """
    Image and Metadata transient classifier for use with Hyrax
    """
    def __init__(self, config=None, data_sample=None):
        super().__init__()

        self.config = config
        ac = self.config['model']['AstroMiNN']

        self.has_image = True #! Flag for image availability, is this important???
        self.num_classes = ac['num_classes']
        self.num_mlp_experts = ac['num_mlp_experts']
        self.towers_hidden_dims = ac['towers_hidden_dims']
        self.towers_outdims = ac['towers_outdims']

        self.fusion_hidden_dims = ac['fusion_hidden_dims'] # was 1024
        self.fusion_router_dims = ac['fusion_router_dims'] # was 256
        self.fusion_outdims = ac['fusion_outdims']

        # ===== Metadata Processing Towers =====
        #! All of these seem extremely fragile - we should define the input_size
        #! at runtime using the data_sample as the guide.
        # Each tower processes specific metadata features
        # PSF quality features tower
        self.psf_tower = ResidualTowerBlock(2, self.towers_hidden_dims, self.towers_outdims)

        # Magnitude features tower
        self.mag_tower = ResidualTowerBlock(7, self.towers_hidden_dims*2, self.towers_outdims)

        # LC features tower
        self.lc_tower = ResidualTowerBlock(12, self.towers_hidden_dims*3, self.towers_outdims)

        # Spatial features tower (distpsnr1, distpsnr2, nmtchps)
        self.spatial_tower = ResidualTowerBlock(3, self.towers_hidden_dims, self.towers_outdims)

        # Nearest source features tower 1 (sgscore1, distpsnr1)
        self.nst1_tower = ResidualTowerBlock(2, self.towers_hidden_dims, self.fusion_outdims)

        # Nearest source features tower 2 (sgscore2, distpsnr2)
        self.nst2_tower = ResidualTowerBlock(2, self.towers_hidden_dims, self.fusion_outdims)

        self.coord_tower = ResidualTowerBlock(2, self.towers_hidden_dims, self.fusion_outdims)

        self.mega_tower = ResidualTowerBlock(19, 128, self.towers_outdims)

        # ===== Image Processing =====
        #! Hardcoded '4' should probably dynamically extracted from `data_sample`???
        self.image_tower = SplitHeadConvNeXt(
            pretrained=False, # False if training from scratch
            in_chans=4, #! Critical: override default 3-channel input
            outdims=self.towers_outdims # Your task's number of classes
        )

        #! These are odd hard coded values: 6 and 3???
        fusion_dims = 6*self.towers_outdims + 3*self.fusion_outdims

        # ===== Modality Fusion MoE =====
        # Combines features from all towers (4 metadata + image)
        #! I'm assuming that the hardcoded '5' is the num_classes defined in the config?
        self.fusion_experts = nn.ModuleList([
            ResidualTowerBlock(fusion_dims, self.fusion_hidden_dims, 5)
            for _ in range(self.num_mlp_experts)
        ])

        self.fusion_router = nn.Sequential(
            nn.Linear( fusion_dims, fusion_dims//2),
            nn.Tanh(),
            nn.Dropout(0.3), #! This seems like it should be a config parameter too???
            nn.Linear(fusion_dims//2, self.num_mlp_experts),
            nn.Sigmoid()
        )

        # Define some training metrics
        self.total_loss = []
        self.total_correct_predictions = 0
        self.total_predictions = 0

    def forward(self, metadata, image):
        """Processes input metadata and optional image data through specialized towers,
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
                Same as expert_weights (maintained for compatibility)
        """

        # Process all metadata features through respective towers
        nsta = self.nst1_tower(metadata[:, [0,2]]) # Nearest source A features
        nstb = self.nst2_tower(metadata[:, [1,3]]) # Nearest source B features
        spatial_feats = self.spatial_tower(metadata[:, [2,3,4]]) # Spatial features
        psf_feats = self.psf_tower(metadata[:, [5,14]])  # PSF features
        mag_feats = self.mag_tower(metadata[:, [6, 9, 10, 13, 15, 17, 18]])
        coord_feats = self.coord_tower(metadata[:, [7,8]])
        #! Why not metadata[:, :19] ???
        megatower = self.mega_tower(metadata[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]])
        # Process image if available (zeros otherwise)
        image_feats = self.image_tower(image) if image is not None else torch.zeros_like(nsta)
        lc_feats = self.lc_tower(metadata[:, [6, 9, 10, 13, 15, 17, 18, 19, 20, 21, 22, 23]])

        # Concatenate all features for fusion
        all_feats = torch.cat([nsta, nstb, spatial_feats, psf_feats, mag_feats, coord_feats, megatower, image_feats, lc_feats], dim=1)

        # Fusion MoE - combine features from all modalities
        fusion_weights = self.fusion_router(all_feats)

        #! Is the hardcoded '5' here the num_classes from the config???
        moe_output = torch.zeros(metadata.size(0), 5)

        topk_weights, topk_indices = torch.topk(fusion_weights, k=2, dim=-1)  # [B, k]

        # Process only through selected experts
        for expert_idx, expert in enumerate(self.fusion_experts):
            # Mask for samples where this expert is in top-k
            expert_mask = (topk_indices == expert_idx).any(dim=-1) # [B]   # 'ResidualTowerBlock'

            if expert_mask.any():
                # Get weights for this expert [M] where M=sum(expert_mask)
                weights = topk_weights[expert_mask, (topk_indices[expert_mask] == expert_idx).nonzero()[:, 1]]
                # Compute expert output only for relevant samples
                expert_out = expert(all_feats[expert_mask]) # [M, num_classes]
                # Weighted contribution
                moe_output[expert_mask] += weights.unsqueeze(-1) * expert_out

        return moe_output

    def _update_stats(self, loss, logits, labels):
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probabilities, dim=1)
        correct_predictions = (predicted_labels == labels).sum().item()

        self.total_correct_predictions += correct_predictions
        self.total_predictions += labels.size(0)
        self.total_loss.append(loss.item())

    def _calculate_stats(self):
        return sum(self.total_loss) / len(self.total_loss), self.total_correct_predictions / self.total_predictions

    def train_step(self, batch):
        """This method has been created based on the logic found in the file
        `.../AppleCider/core/trainer.py`.
        Based on that file, the optimizer and criterion functions appear to be
        configurable, but it's not clear which one are actually used.
        """

        # This is a placeholder until I implement the to_tensor method.
        print(batch)
        metadata, images, labels = batch

        self.optimizer.zero_grad()

        logits = self.forward(metadata, images)

        loss = self.criterion(logits, labels)

        self._update_stats(loss, logits, labels)

        loss.backward()

        self.optimizer.step()

        # in trainer.py this is calculated per epoch, here we calculate it per batch
        loss, acc = self._calculate_stats()

        return {'loss': loss, 'acc': acc}


    @staticmethod
    def to_tensor(data_dict):
        """Place holder for use with Hyrax. This method will receive a dictionary
        of data and should convert it to the relevant tensors needed for either
        training or inference."""
        data = data_dict['data']

        columns = [chr(i) for i in range(ord('a'), ord('z')+1)]

        # horizontally concatenate metadata columns
        metadata = torch.tensor(np.hstack([data[c] for c in columns]))
        images = torch.tensor(data['image']).float()
        labels = torch.tensor(data['label']).long()
        return (metadata, images, labels)
