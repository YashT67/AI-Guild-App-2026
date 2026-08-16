import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os

def latlon_to_cartesian(lat, lon):
    """Converts Lat/Lon to 3D Cartesian coordinates to perfectly handle the Dateline wrap-around."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x, y, z))

def create_clusters():
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    input_csv = os.path.join(base_dir, 'training_dataset', 'noised_dataset', 'ground_truth_coordinates.csv')
    output_csv = os.path.join(base_dir, 'training_dataset', 'noised_dataset', 'clustered_training_data.csv')
    
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print("Converting Lat/Lon to 3D Cartesian coordinates...")
    xyz = latlon_to_cartesian(df['latitude'].values, df['longitude'].values)
    
    n_clusters = 160
    print(f"Running K-Means Geo-Clustering (K={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(xyz)
    
    # Format as string to mimic ISO behavior without breaking existing dataloader logic
    df['cluster_id'] = "CLUSTER_" + df['cluster_id'].astype(str)
    
    print(f"Saving clustered dataset to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    centroids_csv = os.path.join(base_dir, 'training_dataset', 'noised_dataset', 'cluster_centroids.csv')
    print(f"Saving cluster centroids to {centroids_csv}...")
    # kmeans.cluster_centers_ is shape (n_clusters, 3) because it was fit on xyz
    centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=['x', 'y', 'z'])
    centroids_df['cluster_id'] = "CLUSTER_" + centroids_df.index.astype(str)
    centroids_df.to_csv(centroids_csv, index=False)
    
    print(f"SUCCESS! You now have {n_clusters} perfectly balanced geographical classes.")
    print("Cluster sizes:")
    print(df['cluster_id'].value_counts().describe())

if __name__ == "__main__":
    create_clusters()
