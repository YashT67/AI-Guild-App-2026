import pandas as pd
import json

print("Loading GeoJSON...")
with open('c:/Users/Yash T/Desktop/geoguessr/geolocation-prediction/country_boundaries.geojson', 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

geojson_countries = set()
geojson_iso = set()
for feature in geojson_data.get('features', []):
    props = feature.get('properties', {})
    name = props.get('country_name')
    # Try common ISO 2-letter code property names
    iso = props.get('ISO_A2') or props.get('iso_a2') or props.get('iso_a2_eh')
    if name:
        geojson_countries.add(name)
    if iso:
        geojson_iso.add(iso)

print(f"GeoJSON has {len(geojson_countries)} unique country names.")
# Print sample property keys to debug if ISO codes are named differently
sample_props = geojson_data['features'][0]['properties'].keys() if geojson_data.get('features') else []
print(f"GeoJSON property keys: {list(sample_props)}")

print("\nLoading coordinates.csv...")
df = pd.read_csv('c:/Users/Yash T/Desktop/geoguessr/geolocation-prediction/extra_training_dataset/coordinates.csv')
csv_countries = set(df['country'].dropna().unique())

print(f"CSV has {len(csv_countries)} unique 'country' values.")
print(f"Sample CSV 'country' values: {list(csv_countries)[:5]}")

print("\n--- Matching Analysis ---")
# Let's see if the CSV 'country' matches ISO_A2 codes or Country Names
missing_by_name = csv_countries - geojson_countries
missing_by_iso = csv_countries - geojson_iso

print(f"Missing if matching by ISO Code: {len(missing_by_iso)} countries.")
if len(missing_by_iso) < 20:
    print(f"Missing ISOs: {missing_by_iso}")

print(f"Missing if matching by Country Name: {len(missing_by_name)} countries.")
if len(missing_by_name) < 20:
    print(f"Missing Names: {missing_by_name}")
