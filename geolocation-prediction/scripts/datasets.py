import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class GeoguessrDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None, label_mapping=None, geojson_path=None):
        """
        csv_path: Path to enriched_training_data.csv
        image_dir: Path to the folder containing actual .jpg images
        transform: Albumentations transforms to apply to the images
        label_mapping: Dictionary mapping country_name to integer. If None, it will be built automatically.
        geojson_path: Optional path to country_boundaries.geojson to guarantee all countries are mapped.
        """
        # keep_default_na=False stops pandas from parsing Namibia's 'NA' country code as a NaN value!
        self.df = pd.read_csv(csv_path, keep_default_na=False, na_values=[''])
        
        # Drop any rows where ISO_A2 is missing, just to be safe
        self.df = self.df.dropna(subset=['ISO_A2'])
        self.df = self.df.reset_index(drop=True)
        
        self.image_dir = image_dir
        self.transform = transform
        
        # Build or use provided label mapping
        if label_mapping is None:
            if geojson_path is not None and os.path.exists(geojson_path):
                import json
                with open(geojson_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                unique_countries = set()
                for feature in data.get('features', []):
                    name = feature.get('properties', {}).get('ISO_A2')
                    if name:
                        unique_countries.add(name)
                # Ensure CSV countries are included just in case
                csv_countries = set(self.df['ISO_A2'].unique())
                unique_countries = sorted(list(unique_countries.union(csv_countries)))
            else:
                unique_countries = sorted(self.df['ISO_A2'].unique())
                
            self.label_mapping = {country: idx for idx, country in enumerate(unique_countries)}
        else:
            self.label_mapping = label_mapping
            
        # Store inverse mapping for easy inference later
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Image paths are usually just the image_id + '.jpg' or just image_id depending on the CSV
        image_name = row['image_id']
        if not str(image_name).endswith('.jpg'):
            image_name = str(image_name) + '.jpg'
            
        img_path = os.path.join(self.image_dir, image_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # If an image is corrupt or missing, print error and return a black image
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        image_np = np.array(image)
        
        # Apply transforms
        if self.transform is not None:
            # Albumentations expects a numpy array
            try:
                augmented = self.transform(image=image_np)
                image_tensor = augmented['image']
            except TypeError:
                # Fallback if using standard torchvision transforms
                image_tensor = self.transform(image)
        else:
            # Fallback if no transform is provided (not recommended for training)
            from torchvision import transforms
            fallback_transform = transforms.ToTensor()
            image_tensor = fallback_transform(image)
            
        # Get labels
        country_name = row['ISO_A2']
        country_label = self.label_mapping[country_name]
        
        lat = torch.tensor(row['latitude'], dtype=torch.float32)
        lon = torch.tensor(row['longitude'], dtype=torch.float32)
        country_label = torch.tensor(country_label, dtype=torch.long)
        
        return {
            'image': image_tensor,
            'country_label': country_label,
            'latitude': lat,
            'longitude': lon,
            'image_id': row['image_id']
        }
        
    def get_num_classes(self):
        return len(self.label_mapping)
