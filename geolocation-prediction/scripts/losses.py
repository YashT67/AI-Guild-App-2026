import torch
import torch.nn as nn

def latlon_to_cartesian(lat, lon):
    """
    Converts latitude and longitude (in degrees) to 3D Cartesian coordinates on a unit sphere.
    """
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    
    # Stack into a single tensor of shape (batch_size, 3)
    return torch.stack([x, y, z], dim=-1)

def cartesian_to_latlon(xyz):
    """
    Converts 3D Cartesian coordinates on a unit sphere back to latitude and longitude (in degrees).
    Expects xyz to be a tensor of shape (..., 3).
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    
    lon_rad = torch.atan2(y, x)
    # Clamp z to [-1, 1] to avoid NaNs from floating point inaccuracies
    lat_rad = torch.asin(torch.clamp(z, -1.0, 1.0))
    
    lat = torch.rad2deg(lat_rad)
    lon = torch.rad2deg(lon_rad)
    
    return lat, lon

def haversine_distance(pred_lat, pred_lon, target_lat, target_lon):
    """
    Calculates the great-circle distance between two points on Earth in kilometers.
    Uses PyTorch tensors.
    """
    # Earth radius in kilometers
    R = 6371.0
    
    pred_lat_rad = torch.deg2rad(pred_lat)
    pred_lon_rad = torch.deg2rad(pred_lon)
    target_lat_rad = torch.deg2rad(target_lat)
    target_lon_rad = torch.deg2rad(target_lon)
    
    dlat = target_lat_rad - pred_lat_rad
    dlon = target_lon_rad - pred_lon_rad
    
    a = torch.sin(dlat / 2)**2 + torch.cos(pred_lat_rad) * torch.cos(target_lat_rad) * torch.sin(dlon / 2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    return R * c

class CoordinateLoss(nn.Module):
    def __init__(self):
        super(CoordinateLoss, self).__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred_xyz, target_lat, target_lon):
        """
        Calculates the MSE loss between the predicted 3D coordinates and the ground truth.
        Minimizing MSE on the unit sphere perfectly correlates with minimizing the chordal distance.
        """
        target_xyz = latlon_to_cartesian(target_lat, target_lon)
        return self.mse(pred_xyz, target_xyz)
