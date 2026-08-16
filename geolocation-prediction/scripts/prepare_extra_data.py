import pandas as pd
import numpy as np
import os
import torch

def latlon_to_cartesian(lat, lon):
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x, y, z))

def process_extra_data():
    base_dir = r"c:\Users\Yash T\Desktop\geoguessr\geolocation-prediction"
    extra_dir = os.path.join(base_dir, "extra_training_dataset")
    csv_path = os.path.join(extra_dir, "coordinates.csv")
    out_csv_path = os.path.join(extra_dir, "extra_clustered.csv")
    
    if os.path.exists(out_csv_path):
        print(f"--> Found {out_csv_path}. Skipping preprocessing.")
        return

    print("Loading 210k extra coordinates...")
    df = pd.read_csv(csv_path)
    df['id'] = df['id'].astype(str)
    
    print("Scanning physical folders (00 to 04) to map image locations...")
    id_to_folder = {}
    for folder in ['00', '01', '02', '03', '04']:
        folder_path = os.path.join(extra_dir, "images", folder)
        if not os.path.exists(folder_path): continue
        for f in os.listdir(folder_path):
            if f.endswith('.jpg'):
                id_to_folder[f.replace('.jpg', '')] = folder
                
    df['folder'] = df['id'].map(id_to_folder)
    df = df.dropna(subset=['folder']).reset_index(drop=True)
    
    print(f"Matched {len(df)} images to physical files.")
    
    centroids_path = os.path.join(base_dir, "training_dataset", "noised_dataset", "cluster_centroids.csv")
    if not os.path.exists(centroids_path):
        print("ERROR: cluster_centroids.csv not found! Run prepare_data.py first.")
        return
        
    centroids_df = pd.read_csv(centroids_path)
    centroid_coords = centroids_df[['x', 'y', 'z']].values
    cluster_ids = centroids_df['cluster_id'].values
    
    print("Mapping 210k coordinates to the 176 K-Means geographic clusters...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    chunk_size = 10000
    all_assigned_clusters = []
    
    xyz = latlon_to_cartesian(df['latitude'].values, df['longitude'].values)
    xyz_tensor = torch.tensor(xyz, dtype=torch.float32, device=device)
    centroid_tensor = torch.tensor(centroid_coords, dtype=torch.float32, device=device)
    
    for i in range(0, len(xyz_tensor), chunk_size):
        chunk = xyz_tensor[i:i+chunk_size]
        dist_sq = torch.sum((chunk.unsqueeze(1) - centroid_tensor.unsqueeze(0))**2, dim=2)
        closest_idxs = torch.argmin(dist_sq, dim=1).cpu().numpy()
        all_assigned_clusters.extend([cluster_ids[idx] for idx in closest_idxs])
        
    df['cluster_id'] = all_assigned_clusters
    
    # Standardize columns to match our original pipeline
    df = df[['id', 'latitude', 'longitude', 'folder', 'cluster_id']]
    df = df.rename(columns={'id': 'image_id'})
    
    print(f"Saving prepared extra data to {out_csv_path}...")
    df.to_csv(out_csv_path, index=False)
    print("SUCCESS! Data is perfectly mapped and ready for training.")

if __name__ == "__main__":
    process_extra_data()
