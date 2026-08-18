import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def latlon_to_cartesian(lat, lon):
    """
    Converts PyTorch tensors of lat/lon (degrees) to 3D unit sphere Cartesian.
    Returns tensor of shape (batch_size, 3).
    """
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    
    return torch.stack([x, y, z], dim=1)

def cartesian_to_latlon(xyz):
    """
    Convert 3D unit sphere Cartesian coordinates back to lat/lon (degrees).
    xyz: tensor of shape (batch_size, 3).
    Returns: (pred_lat, pred_lon) tensors.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat_rad = torch.asin(torch.clamp(z, min=-1.0 + 1e-7, max=1.0 - 1e-7))
    lon_rad = torch.atan2(y, x)
    return torch.rad2deg(lat_rad), torch.rad2deg(lon_rad)

def haversine_distance(pred_lat, pred_lon, target_lat, target_lon, R=6371.0):
    """
    Calculate the exact Earth great-circle distance in kilometers using PyTorch tensors.
    """
    lat1 = torch.deg2rad(pred_lat)
    lon1 = torch.deg2rad(pred_lon)
    lat2 = torch.deg2rad(target_lat)
    lon2 = torch.deg2rad(target_lon)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    # Clip a to prevent nan from floating point inaccuracies
    a = torch.clamp(a, 0.0, 1.0)
    c = 2 * torch.asin(torch.sqrt(a))
    
    return R * c

class HaversineSmoothedCrossEntropy(nn.Module):
    def __init__(self, dist_matrix_npy_path, sigma, device='cuda'):
        """
        dist_matrix_npy_path: Path to the precomputed distance matrix (.npy)
        sigma: The bandwidth for the Gaussian smoothing in kilometers
        """
        super().__init__()
        import numpy as np
        
        # Load distance matrix (K x K)
        dist_matrix = np.load(dist_matrix_npy_path)
        dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
        
        # Compute soft target distributions using Gaussian decay based on Haversine distance
        # y_i = exp(-d^2 / (2 * sigma^2))
        raw_weights = torch.exp(-(dist_tensor ** 2) / (2 * (sigma ** 2)))
        
        # Normalize each row so it sums to 1 (making it a valid probability distribution)
        self.soft_targets = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).to(device)
        
        # KLDivLoss expects input in log-space and target in prob-space
        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, logits, targets):
        """
        logits: (batch_size, num_classes) - raw model predictions
        targets: (batch_size,) - ground truth class indices
        """
        # Get the precomputed soft target distribution for each ground truth class
        batch_soft_targets = self.soft_targets[targets]
        
        # Convert logits to log-probabilities
        log_probs = F.log_softmax(logits, dim=1)
        
        return self.kldiv(log_probs, batch_soft_targets)

class GeometricHuberLoss(nn.Module):
    def __init__(self, threshold_km=500.0, R=6371.0, eps=1e-7):
        """
        Piecewise loss:
        - Haversine loss if error > threshold_km (constant strong gradient)
        - MSE loss (scaled) if error <= threshold_km (safe convex bowl near zero)
        """
        super().__init__()
        self.threshold_km = threshold_km
        self.R = R
        self.eps = eps
        
        # Calculate the angular threshold in radians
        self.theta_threshold = threshold_km / R

    def forward(self, pred_xyz, target_xyz):
        """
        pred_xyz: (batch_size, 3) - predicted unit sphere coordinates
        target_xyz: (batch_size, 3) - ground truth unit sphere coordinates
        """
        # Calculate cosine of angle between pred and target
        # Dot product of unit vectors = cos(theta)
        cos_theta = torch.sum(pred_xyz * target_xyz, dim=-1)
        
        # Clamp strictly inside (-1, 1) to avoid NaN derivatives from acos
        # This is critical for preventing crashes near exact 0 or 180 degree errors
        clamped_cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        
        # Angular error in radians
        theta = torch.acos(clamped_cos_theta)
        
        # 1. Haversine Loss Component (Direct Geodesic)
        # Distance = R * theta
        haversine_loss_vals = self.R * theta
        
        # 2. MSE Loss Component (Chordal Distance squared)
        # ||u - v||^2 = 2 - 2 * cos(theta)
        mse_loss_vals = 2.0 - 2.0 * cos_theta
        
        # We need to scale the MSE loss so it roughly matches the magnitude of the 
        # Haversine loss at the boundary threshold. This prevents massive gradient 
        # discontinuities when an image crosses the 500km boundary.
        # At boundary theta_t, Haversine = R * theta_t
        # At boundary theta_t, MSE = 2 - 2 * cos(theta_t)
        # Scale factor = (R * theta_t) / (2 - 2 * cos(theta_t))
        scale_factor = self.threshold_km / (2.0 - 2.0 * math.cos(self.theta_threshold))
        scaled_mse_loss_vals = mse_loss_vals * scale_factor
        
        # 3. Dynamic Routing (The "Huber" switch)
        # Determine which images have error > 500km
        mask_large_error = haversine_loss_vals > self.threshold_km
        
        # torch.where(condition, x, y) -> returns x if condition is true, else y
        final_loss_vals = torch.where(
            mask_large_error,
            haversine_loss_vals,    # Use strong Haversine for big errors
            scaled_mse_loss_vals    # Use safe MSE for small errors
        )
        
        return final_loss_vals.mean()
