import numpy as np
import os

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

import math
import random
import math

import timm

from matplotlib.gridspec import GridSpec

from sklearn.metrics import precision_recall_curve, auc

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import auc as sklearn_auc
import random
from datetime import datetime as t

import copy



CLASSES = [['SN Ia','SN Ic','SN Ib'],[ 'SN IIP', 'SN IIn','SN II', 'SN IIb'], ['Cataclysmic'], ['AGN'], ['Tidal Disruption Event']]

# SHOW_CLASSES =['AGN', 'Tidal Disruption Event', 'SN Ia', 'SN Ic', 'SN IIP', 'SN IIn','SN II','SN Ib', 'Cataclysmic']
# SHOW_CLASSES =['AGN', 'Tidal Disruption Event', 'SN Ia', 'other SN', 'Cataclysmic']
# SHOW_CLASSES = ['Nuclear', 'SN', 'Cataclysmic']
SHOW_CLASSES =['SN I', 'SN II', 'Cataclysmic', 'AGN','Tidal Disruption Event']


device = torch.device("cuda")

# GPU Selection Logic
def select_gpu(gpu=None,min_free_memory_gb=22):
    if not gpu==None:
        print(f'cuda:{gpu}')
        return f'cuda:{gpu}'
    if not torch.cuda.is_available():
        return 'cpu'
    
    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)
        if free >= min_free_memory_gb:
            torch.cuda.set_device(i)
            print(f'cuda:{i}')
            return f'cuda:{i}'
    
    print("No suitable GPU found, using CPU")
    return 'cpu'


def calculate_val_loss(loader, model, criterion, DEVICE):
    """Calculate validation loss using the same criterion as training.
    
    Args:
        loader: DataLoader for validation data
        model: Model to evaluate
        criterion: Loss function (same as used in training)
        DEVICE: Device to run calculations on
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            metadata = batch['metadata'].to(DEVICE)
            image = batch['image'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            
            outputs = model(metadata, image)
            loss = criterion(outputs['logits'], target)
            val_loss += loss.item()
    
    return val_loss / len(loader)

def get_class_counts(dataloader):
    num_classes = len(CLASSES)  # Always use coarse classes
    counts = torch.zeros(num_classes, dtype=torch.long)
    
    for batch in dataloader:
        targets = batch['target']
        # For one-hot encoded targets (assuming your targets are one-hot)
        if targets.dim() == 2:
            counts += targets.sum(dim=0).long()
        # For class index targets (alternative format)
        else:
            class_indices = targets.argmax(dim=1) if targets.dim() == 2 else targets
            counts += torch.bincount(class_indices, minlength=num_classes)
    # print(counts)
    
    return counts



class CNN_tower(nn.Module):
    def __init__(self, output_dims=512, img_size=49):
        super().__init__()
        self.output_dims = output_dims
        self.img_size = img_size
        
        # --- Three Independent Backbones ---
        def make_backbone():
            return nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3,  padding='same'),
                nn.ReLU()
            )
        
        self.backbone_ch0 = make_backbone()  # Galaxy 1
        
        # Positional embeddings (only used for ch0 and ch2)
        self.pos_embedding = PositionEmbeddingSine(128, img_size//4)
        self.coord_conv = nn.Conv2d(128, 128, kernel_size=1)
        
        # Attention (only for ch0 and ch2)
        self.attention = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Projection for channel 1 (no position/attention)
        self.proj_ch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128)
        )
        
        # Combined projection (ch0+ch2 features + offsets + ch1 features)
        self.proj = nn.Sequential(
            nn.Linear(128*3, 256),  # +2 for dx/dy between ch0 and ch2
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, output_dims)
        )
        
        self.fusion_router_2 = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # [16, 63, 63]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [16, 31, 31]
            
            # Conv Block 2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [32, 31, 31]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [32, 15, 15]
            
            # Conv Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [64, 15, 15]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [64, 7, 7]
            
            # Final layers
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        device = x.device
        
        # Process each channel with its own backbone
        feats_ch0 = self.backbone_ch0(x[:, 0:1])  # (B, 128, H', W')
        feats_ch1 = self.backbone_ch0(x[:, 1:2])  # (B, 128, H', W')
        feats_ch2 = self.backbone_ch0(x[:, 2:3])  # (B, 128, H', W')
        
        # --- Positional logic for ch0 and ch2 only ---
        pos_encoding = self.pos_embedding(B, H//4, W//4, device)
        pos_features = self.coord_conv(pos_encoding)
        
        # Attended features for ch0 and ch2
        feats_ch0 = feats_ch0 + pos_features
        feats_ch2 = feats_ch2 + pos_features
        
        attn_ch0 = self.attention(feats_ch0)
        attn_ch0 = attn_ch0 / (attn_ch0.sum(dim=(2,3), keepdim=True) + 1e-8)
        attn_ch2 = self.attention(feats_ch2)
        attn_ch2 = attn_ch2 / (attn_ch2.sum(dim=(2,3), keepdim=True) + 1e-8)
        
        
        # Centroids for ch0 and ch2
        def get_centroid(attn):
            B, _, H_, W_ = attn.shape
            grid_x = torch.linspace(-1, 1, W_).to(device).view(1, 1, 1, W_).expand(B, -1, H_, -1)
            grid_y = torch.linspace(-1, 1, H_).to(device).view(1, 1, H_, 1).expand(B, -1, -1, W_)
            cx = (attn * grid_x).sum(dim=(2,3))  # (B, 1)
            cy = (attn * grid_y).sum(dim=(2,3))  # (B, 1)
            return torch.cat([cx, cy], dim=1)  # (B, 2)
        
        centroid_ch0 = get_centroid(attn_ch0)
        centroid_ch2 = get_centroid(attn_ch2)
        
        # Relative offset (ch0 to ch2)
        dx = centroid_ch0[:, 0] - centroid_ch2[:, 0]
        dy = centroid_ch0[:, 1] - centroid_ch2[:, 1]
        offsets = torch.stack([dx, dy], dim=1)

        # With:
        distance = torch.sqrt((centroid_ch0[:,0] - centroid_ch2[:,0])**2 + 
                            (centroid_ch0[:,1] - centroid_ch2[:,1])**2)
        angle = torch.atan2(dy, dx)  # Optional (if angle matters)
        offsets = distance.unsqueeze(1)  # (B, 1) instead of (B, 2)
        
        # Channel 1: Simple pooled features (no position/attention)
        pooled_ch1 = self.proj_ch1(feats_ch1)  # (B, 128)
        
        # Combine features: ch0/ch2 centroids + offsets + ch1 features
        feats_ch0 = feats_ch0.sum(dim=(2,3))  # (B, 128)
        feats_ch2 = feats_ch2.sum(dim=(2,3))  # (B, 128)
        combined = torch.cat([feats_ch0, feats_ch2, pooled_ch1], dim=1)  # (B, 128*3 + 2)
        
        output = self.proj(combined)
        return output




class PositionEmbeddingSine(nn.Module):
    """
    This is a bit of a frankenversion of the position embedding from "Attention Is All You Need"
    and "End-to-End Object Detection with Transformers" (DETR).
    It combines sine/cosine positional encodings with learned features.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        # Learned components
        self.learned_embedding = nn.Sequential(
            nn.Conv2d(2, num_pos_feats//2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(num_pos_feats//2, num_pos_feats, kernel_size=1)
        )
        
        # Fourier feature mapping
        self.fourier_proj = nn.Linear(4, num_pos_feats//2)
        # self.fourier_proj = nn.Linear(4, num_pos_feats//2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.uniform_(self.fourier_proj.weight, 0.0, 1.0)
        nn.init.zeros_(self.fourier_proj.bias)

    def forward(self, batch_size, height, width, device):
        if self.normalize:
            y_embed = torch.linspace(-1, 1, height, device=device)
            x_embed = torch.linspace(-1, 1, width, device=device)
        else:
            y_embed = torch.arange(height, device=device).float()
            x_embed = torch.arange(width, device=device).float()

        if self.normalize:
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale

        # Basic coordinate grid
        dim_t = torch.arange(self.num_pos_feats // 2, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.num_pos_feats // 2))

        pos_x = x_embed[:, None] / dim_t  # (width, num_pos_feats//2)
        pos_y = y_embed[:, None] / dim_t  # (height, num_pos_feats//2)
        
        # Sine/cosine positional encodings
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # (width, num_pos_feats//2)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)  # (height, num_pos_feats//2)
        
        # Create 2D positional encodings
        pos_x = pos_x.unsqueeze(0).expand(height, -1, -1)  # (height, width, num_pos_feats//2)
        pos_y = pos_y.unsqueeze(1).expand(-1, width, -1)    # (height, width, num_pos_feats//2)
        pos = torch.cat([pos_y, pos_x], dim=-1)            # (height, width, num_pos_feats)
        
        # Permute to channel-first format and add batch dimension
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, C, H, W)
        
        # Create basic coordinate maps for learned embedding
        if self.normalize:
            y_coords = torch.linspace(-1, 1, height, device=device)
            x_coords = torch.linspace(-1, 1, width, device=device)
        else:
            y_coords = torch.arange(height, device=device).float() / height
            x_coords = torch.arange(width, device=device).float() / width
            
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coord_map = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, 2, H, W)
        
        # Learned embedding component
        learned_pos = self.learned_embedding(coord_map)  # (B, C, H, W)
        
        # Fourier feature mapping
        fourier_input = torch.cat([
            torch.sin(coord_map * 2 * math.pi),
            torch.cos(coord_map * 2 * math.pi)
        ], dim=1)  # (B, 4, H, W)
        fourier_input = fourier_input.permute(0, 2, 3, 1)  # (B, H, W, 4)
        fourier_features = self.fourier_proj(fourier_input)  # (B, H, W, C/2)
        fourier_features = fourier_features.permute(0, 3, 1, 2)  # (B, C/2, H, W)
        
        # Combine all positional information
        full_pos = torch.cat([
            pos,
            learned_pos,
            fourier_features
        ], dim=1)  # (B, 2.5*C, H, W)
        
        # Project down to original dimension
        return full_pos[:, :self.num_pos_feats]  # (B, C, H, W)
    
    
class SplitHeadConvNeXt(nn.Module):
    # original: def __init__(self, pretrained=False, in_chans=4, outdims=4):
    def __init__(self, pretrained=False, in_chans=4, outdims=4):
        super().__init__()
        # Load base model (disable default head)
        self.backbone = timm.create_model(
            'convnext_tiny',
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0  # Disable original classifier
        )
        
        # Get feature dimension (e.g., 768 for 'convnext_tiny')
        features = self.backbone.num_features
        
        # Define two separate heads
        self.head_main = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(features),
            nn.Linear(features, features//2),    # 4-dimensional output
            nn.ReLU(),
            nn.Dropout(.4),
            nn.Linear(features//2, features),
            nn.Linear( features, outdims)
        )
        
        self.head_aux = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, outdims),
            nn.Tanh()
            
            )     # 1-dimensional output

    def forward(self, x):
        features = self.backbone(x)  # (batch_size, features)
        out = self.head_main(features)  # (batch_size, 4)
        activation = self.head_aux(features)    # (batch_size, 1)
        return out*activation

    
    
    
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#c702b6', '#2ca02c','#1f77b4', '#ff7f0e', '#2ca02c', '#c702b6', '#2ca02c']

def get_model_predictions(loader, model, device):
    """Get model predictions and targets for coarse classification"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        # model.fusion_router.set_testing_mode(True)
        for batch in loader:
            metadata = batch['metadata'].to(device)
            image = batch['image'].to(device)
            
            outputs = model(metadata, image)
            
            # Get probabilities from logits using 
            probs = torch.softmax(outputs['logits'], dim=-1).cpu().numpy()
            targets = batch['target'].cpu().numpy()
            
            all_probs.append(probs)
            all_targets.append(targets)
    
    return np.concatenate(all_probs), np.concatenate(all_targets)


def plot_pr_curves(ax, probs, targets):
    """Plot precision-recall curves with correct color mapping"""
    colors = plt.cm.tab10.colors  # Use a built-in qualitative colormap
    
    for class_idx, class_name in enumerate(CLASSES):
        precision, recall, _ = precision_recall_curve(
            targets[:, class_idx],
            probs[:, class_idx]
        )
        ax.plot(recall, precision, 
               color=colors[class_idx % len(colors)],  # Auto-scale to class count
               lw=2, 
               label=f'{class_name} (AUC={auc(recall, precision):.2f}')  # Show AUC in legend
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='center left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    
    
def plot_confusion_matrix(ax, targets, y_pred):
    """Plot normalized confusion matrix on given axis"""
    # Convert from one-hot to class indices if needed
    if targets.ndim == 2:
        targets = np.argmax(targets, axis=1)
    cm = confusion_matrix(targets, y_pred, labels=range(len(CLASSES)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_title("Normalized Confusion Matrix")
    tick_marks = np.arange(len(CLASSES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, f"{cm_normalized[i, j]:.2f}",
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black")
    
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')



    
def plot_combined_results(loader, model, DEVICE, seed=None, best_model_path=None):
    """Main function to generate combined plot
    Args:
        loader: Data loader for the model
        model: Either a model object or a filepath to a saved .pth file
        seed: Optional seed for filename differentiation
    """
    # Handle case where model is a filepath to a .pth file
    if isinstance(model, str) and model.endswith('.pth'):
        # Load the model from state dict
        model_obj = torch.load(model, weights_only = False)
        if isinstance(model_obj, torch.nn.Module):
            # If the .pth file contains the full model
            model = model_obj
        else:
            # If the .pth file contains just the state dict
            # Note: You'll need to know the model architecture to properly load the state dict
            # You might want to add model_class parameter to handle this case
            raise ValueError("State dict loading requires model architecture information")
    
    # Get predictions
    probs, targets = get_model_predictions(loader, model, DEVICE)
    
    # For confusion matrix, we need class predictions
    y_pred = np.argmax(probs, axis=1)
    
    # Create figure
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    
    # Plot PR curves
    ax1 = plt.subplot(gs[0])
    plot_pr_curves(ax1, probs, targets)
    
    # Plot confusion matrix
    ax2 = plt.subplot(gs[1])
    plot_confusion_matrix(ax2, targets, y_pred)
    
    # Save and show
    plt.tight_layout()
    model_name = model.__class__.__name__ if hasattr(model, '__class__') else model.split('/')[-1].replace('.pth', '')
    filename = f'/projects/bcrv/abrown3/jobs/models_img_meta/{model_name}_{seed}_{wandb.run.id}.png' if seed else f'/projects/bcrv/abrown3/jobs/models_img_meta/{model_name}_{wandb.run.id}.png'
    
    wandb.log({"pr, confusion matrix":wandb.Image(fig)})
    #wandb.log({'plot':wandb.Image(plt)
    #    })
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    # Print metrics
    pr_auc_mean, pr_aucs = print_metrics(loader, model, DEVICE)
    print(f'\nSaved plots under {filename}\n')
    return pr_auc_mean, pr_aucs, plt
    
    
    
def plot_combined_results(loader, model, DEVICE, seed=None, best_model_path=None):
    """Main function to generate combined plot
    Args:
        loader: Data loader for the model
        model: Either a model object or a filepath to a saved .pth file
        seed: Optional seed for filename differentiation
    """
    # Handle case where model is a filepath to a .pth file
    if isinstance(model, str) and model.endswith('.pth'):
        # Load the model from state dict
        model_obj = torch.load(model, weights_only = False)
        if isinstance(model_obj, torch.nn.Module):
            # If the .pth file contains the full model
            model = model_obj
        else:
            # If the .pth file contains just the state dict
            # Note: You'll need to know the model architecture to properly load the state dict
            # You might want to add model_class parameter to handle this case
            raise ValueError("State dict loading requires model architecture information")
    
    # Get predictions
    probs, targets = get_model_predictions(loader, model, DEVICE)
    
    # For confusion matrix, we need class predictions
    y_pred = np.argmax(probs, axis=1)
    
    # Create figure
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    
    # Plot PR curves
    ax1 = plt.subplot(gs[0])
    plot_pr_curves(ax1, probs, targets)
    
    # Plot confusion matrix
    ax2 = plt.subplot(gs[1])
    plot_confusion_matrix(ax2, targets, y_pred)
    
    # Save and show
    plt.tight_layout()
    model_name = model.__class__.__name__ if hasattr(model, '__class__') else model.split('/')[-1].replace('.pth', '')
    filename = f'/projects/bcrv/abrown3/jobs/models_img_meta/{model_name}_{seed}_{wandb.run.id}.png' if seed else f'/projects/bcrv/abrown3/jobs/models_img_meta/{model_name}_{wandb.run.id}.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print metrics
    pr_auc_mean, pr_aucs = print_metrics(loader, model, DEVICE)
    print(f'\nSaved plots under {filename}\n')
    return pr_auc_mean, pr_aucs, plt



def calculate_pr_auc_print_metrics(loader, model, device):
    model.eval()
    all_targets = []
    all_probs = []
    
    # For expert load trackings
    fusion_expert_loads = []
    classification_expert_loads = []
    import torch.nn.functional as F

    with torch.no_grad():
        for batch in loader:
            metadata = batch['metadata'].to(device)
            image = batch['image'].to(device)

            target = batch['target'].to(device)
            
            outputs = model(metadata, image=image)
            
            # Store expert weights
            try:
                fusion_expert_loads.append(outputs['fusion_weights'].cpu())
                classification_expert_loads.append(outputs['expert_weights'].cpu())
            except:
                None
            probs = F.softmax(outputs['logits'], dim=1)
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())
    
    # Calculate PR-AUCs
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()


    # class_fractions = np.average(class_counts.numpy()) / class_counts.numpy() 
    
    pr_aucs = []
    weighted_pr_aucs=[]
    for class_idx in range(len(CLASSES)):  # CLASSES = list of class names/indices
        precision, recall, _ = precision_recall_curve(
            all_targets[:, class_idx],  # Binary targets for this class
            all_probs[:, class_idx]     # Predicted probabilities for this class
        )
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        
        # Weight by class fraction
        # weighted_pr_auc = pr_auc / class_fractions[class_idx]
        # weighted_pr_aucs.append(weighted_pr_auc)


    
    # Calculate expert load statistics
    try:
        fusion_loads = torch.cat(fusion_expert_loads).mean(dim=0).numpy()
        classification_loads = torch.cat(classification_expert_loads).mean(dim=0).numpy()
    
        return np.mean(pr_aucs), pr_aucs, fusion_loads, classification_loads
    except:
        return np.mean(pr_aucs), pr_aucs, [0,0,0], [0,0,0]


def print_metrics(loader, model,  DEVICE):
    """Print evaluation metrics"""
    # Print class distribution
    # print("\nTest set class distribution:")
    #class_counts = {i: 0 for i, name in enumerate(CLASSES)}
    #
    #for batch in loader:
    #    targets = batch['target']
    #    if targets.dim() == 2:  # one-hot
    #        batch_indices = torch.argmax(targets, dim=1)
    #    else:  # class indices
    #        batch_indices = targets
    #  
    #  # for idx in range(len(CLASSES)):
    #  #     class_counts[idx] += (batch_indices == idx).sum().item()
    #
    #for idx, name in enumerate(CLASSES):
    #    print(f"{name}: {class_counts[idx]}")

    # PR-AUC
    pr_auc_mean, pr_aucs, fusion_loads, classification_loads = calculate_pr_auc_print_metrics(loader, model, DEVICE)
    print(f'fusion loads:{fusion_loads}')
    # print(f'classification loads: {classification_loads}')
    print(f"\nMean PR-AUC: {pr_auc_mean:.3f}")
    
    for name, auc in zip(CLASSES, pr_aucs):
        print(f"{name}: {auc:.3f}")
    return pr_auc_mean, pr_aucs



def plot_combined_results(loader, model, DEVICE, seed=None, best_model_path=None):
    """Main function to generate combined plot
    Args:
        loader: Data loader for the model
        model: Either a model object or a filepath to a saved .pth file
        seed: Optional seed for filename differentiation
    """
    # Handle case where model is a filepath to a .pth file
    if isinstance(model, str) and model.endswith('.pth'):
        # Load the model from state dict
        model_obj = torch.load(model, weights_only = False)
        if isinstance(model_obj, torch.nn.Module):
            # If the .pth file contains the full model
            model = model_obj
        else:
            # If the .pth file contains just the state dict
            # Note: You'll need to know the model architecture to properly load the state dict
            # You might want to add model_class parameter to handle this case
            raise ValueError("State dict loading requires model architecture information")
    
    # Get predictions
    probs, targets = get_model_predictions(loader, model, DEVICE)
    
    # For confusion matrix, we need class predictions
    y_pred = np.argmax(probs, axis=1)
    
    # Create figure
    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    
    # Plot PR curves
    ax1 = plt.subplot(gs[0])
    plot_pr_curves(ax1, probs, targets)
    
    # Plot confusion matrix
    ax2 = plt.subplot(gs[1])
    plot_confusion_matrix(ax2, targets, y_pred)
    
    # Save and show
    plt.tight_layout()
    model_name = model.__class__.__name__ if hasattr(model, '__class__') else model.split('/')[-1].replace('.pth', '')
    filename = f'/projects/bcrv/abrown3/jobs/models_img_meta/{model_name}_{seed}_{wandb.run.id}.png' if seed else f'/projects/bcrv/abrown3/jobs/models_img_meta/{model_name}_{wandb.run.id}.png'
    
    wandb.log({"pr, confusion matrix":wandb.Image(fig)})
    #wandb.log({'plot':wandb.Image(plt)
    #    })
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print metrics
    pr_auc_mean, pr_aucs = print_metrics(loader, model, DEVICE)
    print(f'\nSaved plots under {filename}\n')
    return pr_auc_mean, pr_aucs, plt



def calculate_pr_auc(loader, model, class_counts, DEVICE):
    model.eval()
    all_targets = []
    all_probs = []
    
    # For expert load trackings
    fusion_expert_loads = []
    classification_expert_loads = []
    import torch.nn.functional as F

    with torch.no_grad():
        for batch in loader:
            metadata = batch['metadata'].to(DEVICE)
            image = batch['image'].to(DEVICE)

            target = batch['target'].to(DEVICE)
            
            outputs = model(metadata, image=image)
            
            # Store expert weights
            try:
                fusion_expert_loads.append(outputs['fusion_weights'].cpu())
                classification_expert_loads.append(outputs['expert_weights'].cpu())
            except:
                None
            probs = F.softmax(outputs['logits'], dim=1)
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())
    
    # Calculate PR-AUCs
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()


    # class_fractions = np.average(class_counts.numpy()) / class_counts.numpy() 
    
    pr_aucs = []
    weighted_pr_aucs=[]
    for class_idx in range(len(CLASSES)):  # CLASSES = list of class names/indices
        precision, recall, _ = precision_recall_curve(
            all_targets[:, class_idx],  # Binary targets for this class
            all_probs[:, class_idx]     # Predicted probabilities for this class
        )
        pr_auc = auc(recall, precision)
        pr_aucs.append(pr_auc)
        
        # Weight by class fraction
        # weighted_pr_auc = pr_auc / class_fractions[class_idx]
        # weighted_pr_aucs.append(weighted_pr_auc)
    
    # Calculate expert load statistics
    try:
        fusion_loads = torch.cat(fusion_expert_loads).mean(dim=0).numpy()
        classification_loads = torch.cat(classification_expert_loads).mean(dim=0).numpy()
    
        return np.mean(pr_aucs), pr_aucs, fusion_loads, classification_loads
    except:
        return np.mean(pr_aucs), pr_aucs, [0,0,0], [0,0,0]



    
    
    
def calculate_val_loss(loader, model, criterion, DEVICE):
    """Calculate validation loss using the same criterion as training.
    
    Args:
        loader: DataLoader for validation data
        model: Model to evaluate
        criterion: Loss function (same as used in training)
        DEVICE: Device to run calculations on
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            metadata = batch['metadata'].to(DEVICE)
            image = batch['image'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            
            outputs = model(metadata, image)
            loss = criterion(outputs['logits'], target)
            val_loss += loss.item()
    
    return val_loss / len(loader)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth  # Prevents division by zero

    def forward(self, y_pred, y_true):
        # Convert y_true to one-hot if needed (assuming y_true is class indices)
        if y_true.dim() == 1:
            y_true = F.one_hot(y_true, num_classes=y_pred.size(1)).float()
        
        # Softmax for multi-class
        y_pred = F.softmax(y_pred, dim=1)
        
        # Compute Dice per class
        intersection = (y_pred * y_true).sum(dim=0)
        union = y_pred.sum(dim=0) + y_true.sum(dim=0)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average across classes (you can also weight classes here)
        return 1 - dice.mean()
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # Ensure alpha is properly formatted
        if alpha is not None:
            self.alpha = alpha

    def forward(self, inputs, targets):
        # Convert one-hot targets to class indices if needed
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = targets.argmax(dim=1)
        
        # Ensure targets are long integers
        targets = targets.long()
        
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.unsqueeze(1))
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            at = alpha.gather(0, targets)
            loss = loss * at

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLossWithExpertSpecialization(nn.Module):
    """
    Focal loss with auxiliary loss that encourages class-specific expert specialization.
    """
    def __init__(self, alpha=None, gamma=2, expert_weight=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.expert_weight = expert_weight  # Weight for the auxiliary expert loss
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets, expert_weights=None):
        """
        Args:
            inputs: logits from model [batch_size, num_classes]
            targets: one-hot encoded targets [batch_size, num_classes]
            expert_weights: expert weights from model [batch_size, num_experts]
        Returns:
            Combined focal loss + expert specialization loss
        """
        # Convert one-hot to class indices if needed
        if targets.dim() == 2:
            class_indices = torch.argmax(targets, dim=1)
        else:
            class_indices = targets
        
        # Calculate standard focal loss
        ce_loss = self.ce_loss(inputs, class_indices)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        if expert_weights is not None and self.expert_weight > 0:
            # Auxiliary loss: encourage each class to use its corresponding expert
            batch_size = expert_weights.shape[0]
            num_experts = expert_weights.shape[1]
            
            # Create ideal expert weights (one-hot for corresponding expert)
            ideal_expert_weights = F.one_hot(
                class_indices % num_experts,  # Cycle through experts if more classes than experts
                num_classes=num_experts
            ).float()
            
            # Calculate expert specialization loss (MSE between actual and ideal weights)
            expert_loss = F.mse_loss(expert_weights, ideal_expert_weights)
            
            # Combine losses
            total_loss = focal_loss + self.expert_weight * expert_loss
        else:
            total_loss = focal_loss
            
        return total_loss



        
class MultiClassBCELoss(nn.Module):
    def __init__(self, ignore_index=-100, epsilon=1e-8):
        super().__init__()
        self.ignore_index = ignore_index
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # Ensure proper dimensions and dtype
        targets = targets.long()
        
        if inputs.dim() == 1:
            # Handle binary classification case
            inputs = inputs.unsqueeze(1)
            inputs = torch.cat([1-inputs, inputs], dim=1)
        
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        batch_size, num_classes = probs.shape
        
        # Create correct class mask
        correct_mask = torch.zeros_like(probs)
        if targets.dim() == 1:
            # Standard class indices case
            correct_mask.scatter_(1, targets.unsqueeze(1), 1.0)
        else:
            # Handle one-hot case
            correct_mask = targets.float()
        
        # Calculate loss components
        correct_term = (probs * correct_mask).sum(dim=1) / num_classes
        incorrect_terms = ((1 - probs) * (1 - correct_mask)).sum(dim=1) / num_classes
        
        # Combine terms
        loss_per_sample = -(correct_term + incorrect_terms)
        
        return loss_per_sample.mean()

    
def get_class_counts(dataloader):
    num_classes = len(CLASSES)  # Always use coarse classes
    counts = torch.zeros(num_classes, dtype=torch.long)
    
    for batch in dataloader:
        targets = batch['target']
        # For one-hot encoded targets (assuming your targets are one-hot)
        if targets.dim() == 2:
            counts += targets.sum(dim=0).long()
        # For class index targets (alternative format)
        else:
            class_indices = targets.argmax(dim=1) if targets.dim() == 2 else targets
            counts += torch.bincount(class_indices, minlength=num_classes)
    # print(counts)
    
    return counts


    
    
    
def main(config, trial):
    tnow = t.now()

    
    config = get_config(trial)
    
    
    #config_path='/projects/bcrv/abrown3/XastroMiNN_50_epoch.json'
    #with open(config_path,'r') as f:
    #    config = json.load(f)

    BATCH_SIZE = config['batch_size']
    # LR = 0.0006777718906668259  
    LR = config['learning_rate']
    EPOCHS = config['epochs']
    PATIENCE =  config['patience']
    
    print("config!\n", config, "\n")

    run = wandb.init(project="CiDEr_image_meta",
                     notes="trials.... ", config=config, group=STUDY_NAME, 
                     reinit=True,
                     tags=['trial', '7-5'])
    
    for run in range(int(1)):
        print("Run ID:", wandb.run.id)
        
        # Log config file from local file
        #artifact_config = wandb.Artifact(name="config-file", type="file")
        #artifact_config.add_file(local_path=config_path, name=f'{wandb.run.id}-config.json')
        #wandb.log_artifact(artifact_config)
        wandb.log({"Model": 'XastroMiNN'})

        h = random.randint(100, 190)
        # loader_seed = h + (run*9)
        
        seed = config['seed']
        loader_seed=config['loader_seed']
        # print(f'using loader seed:{loader_seed}')
        # Python and numpy
        random.seed(seed)
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Configure PyTorch for deterministic behavior
        torch.backends.cudnn.deterministic = True  # This makes CUDA operations deterministic
        torch.backends.cudnn.benchmark = False     # Should be False for reproducibility
        
        # Set Python hash seed (I am paranoid)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        #============================================================
        #    Initialize model and optimize tower parameters
        # (this is where it's a bit like taming a pack of dragons)
        #============================================================
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # add
        
        model = XastroMiNN(num_classes=len(CLASSES),num_mlp_experts=config['num_experts'],
                    towers_hidden_dims = config['towers_hidden_dims'],
                    towers_outdims = config['towers_outdims'],
                    fusion_hidden_dims = config['fusion_hidden_dims'],
                    fusion_router_dims = config['fusion_router_dims'],
                    fusion_outdims = config['fusion_outdims']
                    ).to(device)

        if model.__class__.__name__ == 'XastroMiNN':
            optimizer = torch.optim.AdamW([
                    # Image/Coord towers - moderate regularization
                    {'params': model.image_tower.parameters(), 'weight_decay': config['cnn_decay'], 'lr': LR*config['cnn_lr']},
                    
                    # Metadata towers - higher regularization (more prone to overfitting)
                    {'params': model.psf_tower.parameters(), 'weight_decay': config['psf_decay'], 'lr': LR*config['psf_lr']},  
                    {'params': model.lc_tower.parameters(), 'weight_decay': config['lc_decay'], 'lr': LR*config['lc_lr']},  
                    {'params': model.mag_tower.parameters(), 'weight_decay': config['mag_decay'], 'lr': LR*config['mag_lr']},  
                    {'params': model.spatial_tower.parameters(), 'weight_decay': config['spatial_decay'], 'lr': LR*config['spatial_lr']},
                    {'params': model.coord_tower.parameters(), 'weight_decay': config['nst1_decay'], 'lr': LR*config['nst1_lr']},
                    {'params': model.nst1_tower.parameters(), 'weight_decay': config['nst1_decay'], 'lr': LR*config['nst1_lr']},
                    {'params': model.nst2_tower.parameters(), 'weight_decay': config['nst2_decay'], 'lr': LR*config['nst2_lr']},

                    # Currently using mega tower as the router essentially
                    {'params': model.mega_tower.parameters(), 'weight_decay': config['lc_decay'], 'lr': LR*config['lc_lr']},

                    # Fusion components
                    {'params': model.fusion_experts.parameters(), 
                    'weight_decay': config['fusion_decay'],  # Reduced from 5e-4
                    'lr': LR*config['fusion_lr'],
                    'betas':(config['fusion_beta1'], config['fusion_beta2'])
                    },  
                    
                    {'params': model.fusion_router.parameters(), 
                    'weight_decay': config['router_decay'],
                    'lr':LR*config['router_lr'],
                    'betas':(config['router_beta1'], config['router_beta2'])
                    }

                    ], lr=LR, 
                    # betas=(0.9205528268203966, 0.9703153101672825),
                    betas=(config['beta1'], config['beta2']), 
                    eps=config['eps'])  # Avoids divison by zero errors         
            print(f'using {model.__class__.__name__} model with test optimizing parameters')
        else:
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR, betas=(config['beta1'], config['beta2']), eps=config['eps'])


        if config['scheduler'] == 'cosine_annealing':
            scheduler = CosineAnnealingLR(optimizer, config['t_max'])
        if config['scheduler'] == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, 'min',min_lr=config['min_lr'], patience=config['sched_pat'], factor=config['sched_factor'])
        #parent_dir = str(Path(__file__).parent)
        
        wandb.log({
           "gpu":config['gpu'],
            "embed_freq":config["embed_freq"],
            "learning_rate":config["learning_rate"],  # 5e-4,
            "epochs":config["epochs"],
            "patience":config["patience"],
            "batch_size":config["batch_size"],
            "seed":config["seed"],
            "loader_seed":config["loader_seed"],
            "num_experts":config["num_experts"],
            "towers_hidden_dims":config["towers_hidden_dims"],
            "towers_outdims":config["towers_outdims"],
            "embedding":config["embedding"],
            
            
            "dataset classes":config["dataset classes"],
            "classes": config["classes"],
            
            "fusion_hidden_dims":config["fusion_hidden_dims"],
            "fusion_router_dims":config["fusion_router_dims"],
            "fusion_outdims":config["seed"],
            
            "cnn_lr":config["cnn_lr"],
            "cnn_decay":config["cnn_decay"],
            "psf_lr":config["psf_lr"],
            "psf_decay":config["psf_decay"],
            "mag_lr": config["mag_lr"],
            "mag_decay": config["mag_decay"],
            "lc_lr": config["lc_lr"],
            "lc_decay":config["lc_decay"],
            "spatial_lr":config["spatial_lr"],
            "spatial_decay":config["spatial_decay"],
            "coord_lr": config["coord_lr"],
            "coord_decay": config["coord_decay"],
            
            "nst1_lr":config["nst1_lr"],
            "nst1_decay":config["nst1_decay"],
            "nst2_lr":config["nst2_lr"],
            "nst2_decay":config["nst2_decay"],
            
            "fusion_lr":config["fusion_lr"],
            "fusion_decay":config["fusion_decay"],
            "fusion_beta1":config["fusion_beta1"],
            "fusion_beta2":config["fusion_beta2"],
            
            "router_decay":config["router_decay"],
            "router_lr":config['router_lr'],
            "router_beta1":config["router_decay"],
            "router_beta2":config["router_decay"],
            "router_lr_2":config['router_lr_2'],
            "router_beta1_2":config["router_beta1_2"],
            "router_beta2_2":config["router_beta2_2"],
            
            "classifier_decay":config['classifier_decay'],
            "classifier_lr":config['classifier_lr'],
            "classifier_beta1":config["classifier_beta1"],
            "classifier_beta2":config["classifier_beta2"],
            
            "beta1":config["beta1"],
            "beta2":config["beta2"],
            "eps":config["eps"],
            "sched_pat":config["sched_pat"],
            "sched_factor":config["sched_pat"],
            "min_lr":config["min_lr"],
            "weight_exp":config["weight_exp"],
            
            "gamma":config["gamma"],
            "criterion": config["criterion"],
            "scheduler": config["scheduler"], 
            
            "t_max": config['t_max'],
            "max_norm":config['max_norm']})
        
        
        
        # Initialize data loaders'
        print("Loading data...")
        train_loader = get_data_loader(npy_dir='/work/nvme/bcrv/abrown3/preprocessed_data/data_multi/day10/',
                              alert_file_list=Train['file'].to_list(), batch_size=32, random_alert_per_epoch=False)
                              # alert_file_list=custom_train_list, batch_size=32, random_alert_per_epoch=False)
        
        val_loader = get_data_loader(npy_dir='/work/nvme/bcrv/abrown3/preprocessed_data/data_multi/day10/',
                                    #alert_file_list=custom_val_list, batch_size=32, random_alert_per_epoch=False)
                                     alert_file_list=Val['file'].to_list(), batch_size=32, random_alert_per_epoch=False)
        
        test_loader = get_data_loader(npy_dir='/work/nvme/bcrv/abrown3/preprocessed_data/data_multi/day10/',
                              #alert_file_list=custom_test_list, batch_size=32, random_alert_per_epoch=False)
                              alert_file_list=Test['file'].to_list(), batch_size=32, random_alert_per_epoch=False)
        
        print("Finished Loading Data.")

        # assign class weights 
        class_counts = get_class_counts(train_loader)
        print("class_counts:", class_counts)
        
        weight_exponent = config['weight_exp']
        if weight_exponent==1:
            numberator = 30000
        else:
            numberator=5000
        coarse_weights = torch.tensor([
                numberator/(int(count)**(weight_exponent))                  
                for idx, count in enumerate(class_counts)
            ], device=device, dtype=torch.float32)

        # Initialize criterion with the class weights
        if config['criterion'].lower() == 'focalexpert':

            criterion = FocalLossWithExpertSpecialization(
            #criterion = lc.FocalLossWithExpertSpecialization(
                alpha=coarse_weights, 
                gamma=config['gamma'],
                expert_weight=.5  # Add this to your config
            )

        if config['criterion'].lower() == 'focal':
            criterion = FocalLoss(alpha=coarse_weights, gamma=config['gamma'])

        if config['criterion'].lower() == 'binary':
            criterion = nn.BCELoss(weight=coarse_weights)
        if config['criterion'].lower() == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(weight=coarse_weights)

        # else:
        #     print('selected criterion not found in supported list: "focal", "binary", "cross_entropy"')
        # criterion = lc.MultiClassBCELoss()

        #============================================================
        # Main Training Loop
        #============================================================

        best_pr_auc = 0
        best_val_loss = 10
        epochs_no_improve = 0
        # small_batch = config['']
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader,total=len(train_loader), desc='Training', leave=False):
                #print("batch!!: ", batch, "\n")
                
                metadata = batch['metadata'].to(device)
                image = batch['image'].to(device)
                target = batch['target'].to(device)
                #print("image:\n", image,"\n" )
                #print("meta:\n", metadata )

                optimizer.zero_grad()

                outputs = model(metadata, image=image)
                
                #print("outputs.shape", outputs.shape)
                
                if config['criterion'].lower() == 'focalexpert':
                    loss = criterion(outputs['logits'], target, outputs['fusion_weights'])
                else:
                    loss = criterion(outputs['logits'], target)

                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_norm'])
                optimizer.step()
                train_loss += loss.item()

        
            val_pr_auc_mean, val_pr_aucs, _, _ = calculate_pr_auc(val_loader, model, class_counts,  device)
            # val_loss = calculate_val_loss(val_loader, model, criterion, DEVICE)

            train_loss /= len(train_loader)

            if config['scheduler'] == 'cosine_annealing':
                scheduler.step()
            if config['scheduler'] == 'reduce_on_plateau':
                # scheduler.step(val_loss)
                scheduler.step(1-val_pr_auc_mean)

            if val_pr_auc_mean > best_pr_auc:
            # if best_val_loss > val_loss:
                print(val_pr_auc_mean)
                best_pr_auc = val_pr_auc_mean
                # best_val_loss=val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), f'/projects/bcrv/abrown3/jobs/models_img_meta/best_model{seed}_{wandb.run.id}.pth')
                model_best = copy.deepcopy(model) # there's definitely a better way to do this but I don't wanna look it up
                
            else:
                epochs_no_improve += 1
                if epochs_no_improve == PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    
                    wandb.log({
                        "early stop at":{epoch+1},
                    })        
                
                    break
            
 
                    
            wandb.log({
                "epoch":epoch,
                "train_loss": train_loss,
                # "val_loss": val_loss,
                "val_auprc_mean": val_pr_auc_mean,
                **{f"val_pr_auc_{name}": auc for name, auc in zip(SHOW_CLASSES, val_pr_aucs)},
                "best_val_loss": best_val_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "best_val_auprc": best_pr_auc
            })
            
            pr_auc_str = "|".join([f"{name}:{auc:.3f}" for name, auc in zip(SHOW_CLASSES, val_pr_aucs)])
            print(f"Epoch {epoch+1}/{EPOCHS}|"
                f"Train Loss:{train_loss:.4f}|"
                
                # f"Val loss:{val_loss:.3f}|"
                
                  f"Macro mean AUPRC:{val_pr_auc_mean:.4f}|"
                f"Class AUPRCs:{pr_auc_str}")
        print(f'best loss{best_val_loss}')
        # Evaluation
        random_stats = random_baseline_pr_auc(test_loader, n_trials=1000)
        print(f"Random Baseline PR-AUCs (mean ± std):")
        for i, class_name in enumerate(CLASSES):
            print(f"{class_name}: {random_stats['mean'][i]:.3f} ± {random_stats['std'][i]:.3f}")
        
        # Plot and save results
        pr_auc_mean, pr_aucs, plt = plot_combined_results(test_loader, model_best, device)
        
        
        
        wandb.log({
            #'confusion_matrix': wandb.Image(plt), # saving plots after plt.show() saves blank
            'test_auprc_mean': pr_auc_mean,
             **{f"test_auprc_{name}": auc for name, auc in zip(SHOW_CLASSES, pr_aucs)}
        })
        
        wandb.save(f'models/best_model_{seed}_{wandb.run.id}.pth')
        print(f'Time taken so far: {t.now()-tnow}')
        print("Run ID:", wandb.run.id)
        wandb.finish()

# NO! 
#def parse_args():
#    parser = argparse.ArgumentParser(description='select a model')
#    parser.add_argument('runs', type=str, nargs='?', default='XastroMiNN',
#                      help='which model do you want to use? (default: XastroMiNN)')
#    return parser.parse_args().runs
def random_baseline_pr_auc(loader, n_trials=1000):
    all_targets = []
    for batch in loader:
        targets = batch['target']  # Directly use the target tensor

        # Convert one-hot to class indices if needed
        if targets.dim() == 2:
            targets = torch.argmax(targets, dim=1)

        all_targets.append(targets.cpu().numpy())
    targets = np.concatenate(all_targets)
    
    num_classes = len(CLASSES) 
    trial_pr_aucs = np.zeros((n_trials, num_classes))

    
    for trial in range(n_trials):
        np.random.seed(trial)
        # Generate random probabilities that sum to 1
        random_probs = np.random.dirichlet(np.ones(num_classes), size=len(targets))
        
        for class_idx in range(num_classes):
            precision, recall, _ = precision_recall_curve(
                (targets == class_idx).astype(int),
                random_probs[:, class_idx]
            )
            trial_pr_aucs[trial, class_idx] = sklearn_auc(recall, precision)
    
    return {
        'mean': np.mean(trial_pr_aucs, axis=0),
        'std': np.std(trial_pr_aucs, axis=0),
        'all_trials': trial_pr_aucs
    }

#if __name__ == "__main__":
#    main(config, trial)
    
    
    
def get_config(trial):
    
    config = {
        
        "mode": 'trial',
        "fusion trial": True,
        "classifier trial": True,
        "basic trial":True,
        
        "dataset classes":[['SN Ia','SN Ic','SN Ib'],[ 'SN IIP', 'SN IIn','SN II'], ['Cataclysmic'], ['AGN'], ['Tidal Disruption Event']],
        "classes": ['SN I', 'SN II', 'Cataclysmic', 'AGN','Tidal Disruption Event'],

        
        
        "gpu":1,
        "embed_freq": 7,
        
        "learning_rate":1.6e-4, 
        
        #"epochs":50,
        "patience":3,
        "batch_size":128,
        "seed":135,
        "loader_seed":125,
        
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
        "max_norm":5
}

    if STUDY_NAME.startswith('img_meta'):
        config['epochs'] = 100

    else:
        raise NotImplementedError(f"Unknown study name {STUDY_NAME}")

    if config["basic trial"]:
        config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2)
        config['beta1'] = trial.suggest_float('beta1', 0.7, 0.95)
        config['beta2'] = trial.suggest_float('beta2', 0.7, 0.95)
        
    #"fusion trial": True,
    #"classifier trial": True,
    
    if config["classifier trial"]:
        config['classifier_decay'] = trial.suggest_float('classifier_decay',  1e-5, 1e-2)
        config['classifier_lr'] = trial.suggest_float('classifier_lr', 1.6, 2.6)
        config['classifer_beta1'] = trial.suggest_float('classifer_beta1', 0.75, 0.99)
        config['classifer_beta2'] = trial.suggest_float('classifer_beta2',  0.75, 0.99)
        
    if config["fusion trial"]:
        config['fusion_lr'] = trial.suggest_float('fusion_lr', 1e-5, 1e-2, log=True)
        config['fusion_decay'] = trial.suggest_float('fusion_decay', 1e-6, 1e-3)
        config['fusion_beta1'] = trial.suggest_float('fusion_beta1', 0.80, 0.9999)
        config['fusion_beta2'] = trial.suggest_float('fusion_beta2',  0.80, 0.9999)
        
    config['study_name'] = STUDY_NAME

    return config


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
        out = self.main_path(first_half)*gating  + self.skip_path(x)

        return out
    
    

# 🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏 REAL MODEL 🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏🍏


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
        
        # 🤡🤡🤡 add .to(device)
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