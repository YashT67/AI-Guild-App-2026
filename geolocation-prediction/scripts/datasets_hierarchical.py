import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# GLOBAL IMPORT: Fixes the massive Python GIL overhead bottleneck
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

class GeoguessrHierarchicalDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        """
        csv_path: Path to the hierarchical training data CSV
        image_dir: Path to the folder containing the images
        transform: Albumentations transforms
        """
        # keep_default_na=False stops pandas from parsing 'NA' (Namibia) as NaN
        self.df = pd.read_csv(csv_path, keep_default_na=False, na_values=[''])
        
        # Ensure target columns exist
        required_cols = ['cluster_macro_4', 'cluster_meso_36', 'cluster_micro_216', 'latitude', 'longitude']
        self.df = self.df.dropna(subset=required_cols).reset_index(drop=True)
        
        self.image_dir = image_dir
        self.transform = transform
        
        # BUILD LABEL MAPPINGS FOR ALL 3 HEADS
        # Sort unique clusters to ensure deterministic mapping (e.g. CLUSTER_0 -> 0, CLUSTER_1 -> 1)
        self.mapping_1 = {c: i for i, c in enumerate(sorted(self.df['cluster_macro_4'].unique()))}
        self.mapping_2 = {c: i for i, c in enumerate(sorted(self.df['cluster_meso_36'].unique()))}
        self.mapping_3 = {c: i for i, c in enumerate(sorted(self.df['cluster_micro_216'].unique()))}
        
        # IN-MEMORY LISTS FOR EXTREME SPEED
        # This fixes the massive Pandas .iloc overhead bottleneck
        self.image_ids = self.df['image_id'].values
        self.latitudes = self.df['latitude'].values.astype(np.float32)
        self.longitudes = self.df['longitude'].values.astype(np.float32)
        
        # Pre-map string clusters to integer labels for speed
        self.labels_1 = np.array([self.mapping_1[c] for c in self.df['cluster_macro_4'].values], dtype=np.int64)
        self.labels_2 = np.array([self.mapping_2[c] for c in self.df['cluster_meso_36'].values], dtype=np.int64)
        self.labels_3 = np.array([self.mapping_3[c] for c in self.df['cluster_micro_216'].values], dtype=np.int64)
        
        # Optional: For the extra dataset where images are in subfolders 00, 01, 02...
        self.has_folder_col = 'folder' in self.df.columns
        if self.has_folder_col:
            # We enforce saving them as strings with leading zeros (e.g. '00')
            self.folders = self.df['folder'].astype(str).str.zfill(2).values
        else:
            self.folders = None

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = str(self.image_ids[idx])
        
        # Fix image extension if missing
        if not image_id.endswith('.jpg'):
            image_name = image_id + '.jpg'
        else:
            image_name = image_id
            
        # Handle subfolders if this is the 210k extra dataset
        if self.has_folder_col:
            img_path = os.path.join(self.image_dir, self.folders[idx], image_name)
        else:
            img_path = os.path.join(self.image_dir, image_name)
            
        # HIGH-SPEED IMAGE LOADING
        image_np = None
        if HAS_CV2:
            image_np = cv2.imread(img_path)
            if image_np is not None:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                
        # Fallback if cv2 fails or isn't installed
        if image_np is None:
            try:
                # Use context manager to prevent "Too many open files" OS errors
                with Image.open(img_path) as img:
                    image_np = np.array(img.convert('RGB'))
            except Exception as e:
                # Fail-safe: return black image so training doesn't crash
                print(f"Error loading {img_path}: {e}")
                image_np = np.zeros((224, 224, 3), dtype=np.uint8)
                
        # APPLY TRANSFORMS
        # Removed the redundant PIL conversion here. Albumentations eats raw numpy arrays.
        if self.transform is not None:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            # Absolute fallback if no transform is supplied (should not happen)
            import torchvision.transforms.functional as TF
            image_tensor = TF.to_tensor(image_np)
            
        # RETURN BATCH DICT
        return {
            'image': image_tensor,
            'label_1': torch.tensor(self.labels_1[idx]),
            'label_2': torch.tensor(self.labels_2[idx]),
            'label_3': torch.tensor(self.labels_3[idx]),
            'latitude': torch.tensor(self.latitudes[idx]),
            'longitude': torch.tensor(self.longitudes[idx]),
            'image_id': image_id
        }
