import torch.nn as nn
import timm

class GeoguessrModel(nn.Module):
    def __init__(self, num_countries, backbone_name='efficientnet_b0', pretrained=True):
        """
        Initialize the Geoguessr Model.
        Now containing Part 1 (Backbone) and Part 2 (Country Classifier).
        """
        super(GeoguessrModel, self).__init__()
        
        # Part 1: The Vision Backbone
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            num_classes=0
        )
        self.num_features = self.backbone.num_features
        
        # Part 2: Head 1 - Country Classifier (Geography Department)
        # This linear layer takes the 1280 features and outputs probabilities for every country.
        self.country_head = nn.Linear(self.num_features, num_countries)
        
        # --- Part 3 (Coordinate Regressor) will go here later ---
        
    def forward(self, x):
        """
        Forward pass for an image batch 'x'.
        """
        # Get the image summary from the backbone
        features = self.backbone(x)
        
        # Pass the summary through the Geography Department to get country predictions
        country_logits = self.country_head(features)
        
        # We return a dictionary so it's easy to add Latitude/Longitude later
        return {
            'features': features,
            'country_logits': country_logits
        }
