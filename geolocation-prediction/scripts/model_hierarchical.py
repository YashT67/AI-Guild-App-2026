import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def cartesian_to_latlon(xyz):
    """
    Convert 3D Cartesian coordinates to latitude and longitude.
    xyz: tensor of shape (batch_size, 3) representing (x, y, z) on a unit sphere.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # Use eps to avoid NaNs at exactly 1.0 or -1.0
    lat_rad = torch.asin(torch.clamp(z, min=-1.0 + 1e-7, max=1.0 - 1e-7))
    lon_rad = torch.atan2(y, x)
    
    # Convert to degrees
    pred_lat = torch.rad2deg(lat_rad)
    pred_lon = torch.rad2deg(lon_rad)
    
    return pred_lat, pred_lon

class GeoguessrHierarchicalModel(nn.Module):
    def __init__(self, backbone_name='vit_base_patch16_siglip_224', pretrained=True):
        super(GeoguessrHierarchicalModel, self).__init__()
        
        # 1. Vision Backbone (SigLIP)
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.num_features = self.backbone.num_features # Should be 768 for SigLIP base
        
        # 2. Hierarchical Classification Heads (Layer 1)
        self.head_1 = nn.Linear(self.num_features, 4)
        self.head_2 = nn.Linear(self.num_features, 36)
        self.head_3 = nn.Linear(self.num_features, 216)
        
        # 3. Coordinate Regressor (Layer 2)
        # Input dimension = 768 (features) + 4 (probs_1) + 36 (probs_2) + 216 (probs_3) = 1024
        self.coord_input_dim = self.num_features + 4 + 36 + 216
        
        self.coordinate_head = nn.Sequential(
            nn.BatchNorm1d(self.coord_input_dim),
            nn.Linear(self.coord_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        
    def forward(self, x, force_probs_1=None, force_probs_2=None, force_probs_3=None):
        # Extract features
        features = self.backbone(x)
        
        # Compute hierarchical logits
        logits_1 = self.head_1(features)
        logits_2 = self.head_2(features)
        logits_3 = self.head_3(features)
        
        # Convert to probabilities (use forced ones if provided, e.g., during coordinate head isolated training)
        probs_1 = force_probs_1 if force_probs_1 is not None else F.softmax(logits_1, dim=1)
        probs_2 = force_probs_2 if force_probs_2 is not None else F.softmax(logits_2, dim=1)
        probs_3 = force_probs_3 if force_probs_3 is not None else F.softmax(logits_3, dim=1)
        
        # Concatenate features and all three probability vectors
        coord_input = torch.cat([features, probs_1, probs_2, probs_3], dim=1)
        
        # Regress 3D Cartesian coordinates
        raw_xyz = self.coordinate_head(coord_input)
        
        # Normalize to enforce unit sphere projection
        pred_xyz = F.normalize(raw_xyz, p=2, dim=1)
        
        # Convert back to Latitude and Longitude
        pred_lat, pred_lon = cartesian_to_latlon(pred_xyz)
        
        return {
            'features': features,
            'logits_1': logits_1,
            'logits_2': logits_2,
            'logits_3': logits_3,
            'probs_1': probs_1,
            'probs_2': probs_2,
            'probs_3': probs_3,
            'pred_xyz': pred_xyz,
            'pred_lat': pred_lat,
            'pred_lon': pred_lon
        }

    def freeze_backbone(self):
        """Utility to freeze the backbone during Layer 1 & 2 fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Utility to unfreeze the backbone for the Golden Polish phase."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_layer_1(self):
        """Freeze the classification heads to train Layer 2 in isolation."""
        for head in [self.head_1, self.head_2, self.head_3]:
            for param in head.parameters():
                param.requires_grad = False
                
    def unfreeze_layer_1(self):
        """Unfreeze the classification heads."""
        for head in [self.head_1, self.head_2, self.head_3]:
            for param in head.parameters():
                param.requires_grad = True
