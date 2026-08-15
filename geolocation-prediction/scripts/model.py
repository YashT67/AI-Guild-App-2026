import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from scripts.losses import cartesian_to_latlon

class GeoguessrModel(nn.Module):
    def __init__(self, num_countries, backbone_name='efficientnet_b0', pretrained=True):
        """
        Initialize the Geoguessr Model.
        Contains Part 1 (Backbone), Part 2 (Country Classifier), and Part 3 (Coordinate Regressor).
        """
        super(GeoguessrModel, self).__init__()
        
        # Part 1: The Vision Backbone
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0
        )
        self.num_features = self.backbone.num_features
        
        # Part 2: Head 1 - Country Classifier
        self.country_head = nn.Linear(self.num_features, num_countries)
        
        # Part 3: Head 2 - Coordinate Regressor
        # Input is concatenation of visual features and country probabilities
        self.coordinate_head = nn.Sequential(
            nn.Linear(self.num_features + num_countries, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )
        
    def forward(self, x, force_country_probs=None):
        """
        Forward pass for an image batch 'x'.
        If force_country_probs is provided, it overrides the country_head's prediction.
        """
        # 1. Get visual features
        features = self.backbone(x)
        
        # 2. Get country logits and probabilities
        country_logits = self.country_head(features)
        
        if force_country_probs is not None:
            country_probs = force_country_probs
        else:
            country_probs = F.softmax(country_logits, dim=1)
        
        # 3. Predict coordinates
        # Concatenate features and country_probs along the feature dimension (dim=1)
        coord_input = torch.cat([features, country_probs], dim=1)
        raw_xyz = self.coordinate_head(coord_input)
        
        # Normalize to force it onto the unit sphere
        pred_xyz = F.normalize(raw_xyz, p=2, dim=1)
        
        # Convert back to lat/lon for evaluation and inference
        pred_lat, pred_lon = cartesian_to_latlon(pred_xyz)
        
        return {
            'features': features,
            'country_logits': country_logits,
            'pred_xyz': pred_xyz,
            'pred_lat': pred_lat,
            'pred_lon': pred_lon
        }
