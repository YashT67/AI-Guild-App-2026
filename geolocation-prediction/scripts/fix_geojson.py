import json
import os
import shutil

def fix_geojson():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    geojson_path = os.path.join(base_dir, 'country_boundaries.geojson')
    
    print(f"Loading {geojson_path}...")
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    mapping = {
        'United Kingdom': 'GB',
        'Portugal': 'PT',
        'Netherlands': 'NL',
        'Belgium': 'BE',
        'Serbia': 'RS',
        'Taiwan': 'TW',
        'Bosnia and Herzegovina': 'BA',
        'Georgia': 'GE',
        'Iraq': 'IQ',
        'Kosovo': 'XK',
        'Mayotte': 'YT',
        'French Guiana': 'GF',
        'Antigua and Barbuda': 'AG',
        'Somalia': 'SO',
        'Reunion': 'RE',
        'Papua New Guinea': 'PG',
        'Palestine': 'PS',
        'Guadeloupe': 'GP',
        'Martinique': 'MQ',
        'China': 'CN',
        'United Republic of Tanzania': 'TZ',
        'South Korea': 'KR',
        'Republic of Serbia': 'RS',
        'Northern Cyprus': 'CY',
        'Cyprus No Mans Area': 'CY',
        'Coral Sea Islands': 'AU',
        'Clipperton Island': 'FR',
        'Ashmore and Cartier Islands': 'AU',
        'Baykonur Cosmodrome': 'KZ',
        'Scarborough Reef': 'PH',
        'Bir Tawil': 'EG',
        'Southern Patagonian Ice Field': 'AR',
        'Bajo Nuevo Bank (Petrel Is.)': 'CO',
        'United States Minor Outlying Islands': 'UM',
        'Akrotiri Sovereign Base Area': 'GB',
        'Spratly Islands': 'VN',
        'Siachen Glacier': 'IN',
        'North Korea': 'KP',
        'Brazilian Island': 'BR',
        'Serranilla Bank': 'CO',
        'US Naval Base Guantanamo Bay': 'CU',
        'Dhekelia Sovereign Base Area': 'GB'
    }
    
    fixed_count = 0
    mapped_count = 0
    
    for feature in data.get('features', []):
        props = feature.get('properties', {})
        name = props.get('country_name')
        
        # 1. Fix the bugged -99 ISO codes commonly found in Natural Earth Data
        if props.get('ISO_A2') == '-99':
            if name == 'France':
                props['ISO_A2'] = 'FR'
                fixed_count += 1
            elif name == 'Norway':
                props['ISO_A2'] = 'NO'
                fixed_count += 1
            elif name == 'Somaliland':
                props['ISO_A2'] = 'SO'
                fixed_count += 1
                
        # 2. Inject missing ISO codes based on mapping
        if name in mapping:
            props['ISO_A2'] = mapping[name]
            mapped_count += 1
            
    print(f"Fixed -99 issues for {fixed_count} countries.")
    print(f"Mapped {mapped_count} missing ISO codes from our list.")
    
    output_path = os.path.join(base_dir, 'country_boundaries_fixed.geojson')
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    
    # Overwrite original
    shutil.move(output_path, geojson_path)
    print("Successfully patched country_boundaries.geojson!")

if __name__ == "__main__":
    fix_geojson()
