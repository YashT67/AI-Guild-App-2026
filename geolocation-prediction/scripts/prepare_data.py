import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

def prepare_data():
    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    csv_path = os.path.join(base_dir, 'training_dataset', 'noised_dataset', 'ground_truth_coordinates.csv')
    geojson_path = os.path.join(base_dir, 'country_boundaries.geojson')
    output_path = os.path.join(base_dir, 'training_dataset', 'noised_dataset', 'enriched_training_data.csv')
    
    print(f"Loading coordinates from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create Point geometries from longitude and latitude
    # Note: Longitude is x, Latitude is y
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    
    print("Creating GeoDataFrame...")
    gdf_points = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    
    print(f"Loading country boundaries from {geojson_path}...")
    # Read the geojson
    gdf_countries = gpd.read_file(geojson_path)
    
    # Keep ONLY ISO_A2
    cols = gdf_countries.columns.tolist()
    columns_to_keep = ['geometry']
    if 'ISO_A2' in cols:
        columns_to_keep.append('ISO_A2')
            
    gdf_countries = gdf_countries[columns_to_keep]
        
    print("Performing spatial join (this may take a minute)...")
    # First pass: strict intersection
    enriched_gdf = gpd.sjoin(gdf_points, gdf_countries, how="left", predicate="intersects")
    
    # Separate matched and unmatched
    matched_gdf = enriched_gdf[enriched_gdf['ISO_A2'].notnull()]
    unmatched_gdf = gdf_points[~gdf_points.index.isin(matched_gdf.index)]
    
    if len(unmatched_gdf) > 0:
        print(f"Found {len(unmatched_gdf)} points outside strict boundaries (likely coasts/islands). Finding nearest country...")
        # Second pass: nearest neighbor for the unmatched points
        # Suppress the UserWarning about sjoin_nearest with geographic CRS
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nearest_gdf = gpd.sjoin_nearest(unmatched_gdf, gdf_countries, how="left")
        
        # Combine them back together
        enriched_gdf = pd.concat([matched_gdf, nearest_gdf]).sort_index()

    # Drop the spatial join index and geometry columns for the final CSV
    enriched_df = pd.DataFrame(enriched_gdf.drop(columns=['geometry', 'index_right'], errors='ignore'))
    
    missing_country = enriched_df['ISO_A2'].isnull().sum()
    print(f"Final missing countries: {missing_country}")
    
    print(f"Saving enriched data to {output_path}...")
    enriched_df.to_csv(output_path, index=False)
    print("Done! Data preparation is complete.")

if __name__ == "__main__":
    prepare_data()
