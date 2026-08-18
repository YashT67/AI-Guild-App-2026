import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import torch
import math

def latlon_to_cartesian(lat, lon):
    """Convert latitude and longitude to 3D Cartesian coordinates on a unit sphere."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z

def cartesian_to_latlon(x, y, z):
    """Convert 3D Cartesian coordinates back to latitude and longitude."""
    lat_rad = np.arcsin(np.clip(z, -1.0, 1.0))
    lon_rad = np.arctan2(y, x)
    return np.degrees(lat_rad), np.degrees(lon_rad)

def haversine_distance_matrix(latitudes, longitudes, R=6371.0):
    """
    Calculate the pairwise Haversine distance matrix between a list of lat/lons.
    """
    # Convert to radians
    lat_rad = np.radians(latitudes)
    lon_rad = np.radians(longitudes)
    
    # Create meshgrid for vectorized pairwise calculation
    lat1, lat2 = np.meshgrid(lat_rad, lat_rad)
    lon1, lon2 = np.meshgrid(lon_rad, lon_rad)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    # Clip to [0, 1] to avoid math domain errors due to floating point inaccuracies
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def perform_clustering_and_analysis(df, xyz_coords, k, level_name):
    print(f"\n{'='*50}")
    print(f"Executing K-Means Clustering for {level_name} (K={k})")
    print(f"{'='*50}")
    
    # Run K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(xyz_coords)
    
    # Assign labels to dataframe
    col_name = f'cluster_{level_name}_{k}'
    df[col_name] = [f'CLUSTER_{lbl}' for lbl in cluster_labels]
    
    # Get centroids and convert back to lat/lon
    centroids_xyz = kmeans.cluster_centers_
    # Normalize centroids back to unit sphere (KMeans centroids might be slightly inside the sphere)
    norms = np.linalg.norm(centroids_xyz, axis=1, keepdims=True)
    centroids_xyz = centroids_xyz / norms
    
    centroids_lat, centroids_lon = cartesian_to_latlon(centroids_xyz[:, 0], centroids_xyz[:, 1], centroids_xyz[:, 2])
    
    # Create centroid dataframe
    centroids_df = pd.DataFrame({
        'cluster_id': [f'CLUSTER_{i}' for i in range(k)],
        'x': centroids_xyz[:, 0],
        'y': centroids_xyz[:, 1],
        'z': centroids_xyz[:, 2],
        'latitude': centroids_lat,
        'longitude': centroids_lon
    })
    
    # Calculate pairwise Haversine distance matrix
    dist_matrix = haversine_distance_matrix(centroids_lat, centroids_lon)
    
    # --- Analysis Phase ---
    cluster_counts = df[col_name].value_counts()
    
    print(f"--- Analysis for {level_name} (K={k}) ---")
    print(f"Total Clusters: {k}")
    print(f"Largest Cluster Size: {cluster_counts.max()} images")
    print(f"Smallest Cluster Size: {cluster_counts.min()} images")
    print(f"Average Cluster Size: {cluster_counts.mean():.1f} images")
    print(f"Median Cluster Size: {cluster_counts.median():.1f} images")
    
    # Distance analysis (excluding diagonal 0s)
    dist_matrix_no_diag = dist_matrix[~np.eye(dist_matrix.shape[0], dtype=bool)]
    print(f"Average distance between cluster centers: {dist_matrix_no_diag.mean():.2f} km")
    print(f"Minimum distance between distinct centers: {dist_matrix_no_diag.min():.2f} km")
    print(f"Maximum distance between centers: {dist_matrix_no_diag.max():.2f} km")
    
    return df, centroids_df, dist_matrix

def main():
    base_dir = r"C:\Users\Yash T\Desktop\geoguessr\geolocation-prediction"
    data_dir = os.path.join(base_dir, "training_dataset", "noised_dataset")
    csv_path = os.path.join(data_dir, "ground_truth_coordinates.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find ground truth file at {csv_path}")
        return
        
    print(f"Loading ground truth data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} images.")
    
    # Add a safety check in case the user's data contains existing cluster columns
    cols_to_keep = ['image_id', 'latitude', 'longitude']
    if 'ISO_A2' in df.columns:
        cols_to_keep.append('ISO_A2')
    df = df[cols_to_keep]
    
    # Convert all coordinates to 3D Cartesian
    print("Converting coordinates to 3D Cartesian mapping on unit sphere...")
    x, y, z = latlon_to_cartesian(df['latitude'].values, df['longitude'].values)
    xyz_coords = np.column_stack((x, y, z))
    
    # Configuration for hierarchical heads
    hierarchies = [
        {'k': 4, 'name': 'macro'},
        {'k': 36, 'name': 'meso'},
        {'k': 216, 'name': 'micro'}
    ]
    
    out_dir = os.path.join(data_dir, "hierarchical_clusters")
    os.makedirs(out_dir, exist_ok=True)
    
    for h in hierarchies:
        df, centroids_df, dist_matrix = perform_clustering_and_analysis(df, xyz_coords, h['k'], h['name'])
        
        # Save centroids
        centroids_path = os.path.join(out_dir, f"centroids_{h['name']}_{h['k']}.csv")
        centroids_df.to_csv(centroids_path, index=False)
        
        # Save distance matrix (as npy for fast loading in PyTorch)
        dist_matrix_path = os.path.join(out_dir, f"dist_matrix_{h['name']}_{h['k']}.npy")
        np.save(dist_matrix_path, dist_matrix)
        
        print(f"Saved centroids to {centroids_path}")
        print(f"Saved distance matrix to {dist_matrix_path}")
        
    # Save the master enriched dataset
    master_csv_path = os.path.join(data_dir, "hierarchical_training_data.csv")
    df.to_csv(master_csv_path, index=False)
    print(f"\n{'='*50}")
    print(f"Successfully saved master hierarchical training dataset with {len(df)} rows to:")
    print(f"{master_csv_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
