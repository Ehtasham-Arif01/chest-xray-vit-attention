"""PyTorch Dataset classes for chest X-ray data."""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

logger = logging.getLogger(__name__)


class ChestXRayDataset(Dataset):
    """NIH Chest X-ray dataset with multi-label disease classification."""
    
    # Disease labels (14 conditions)
    DISEASES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
        'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        split: str = "train",
        use_bbox: bool = False,
        bbox_dir: Optional[str] = None,
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Path to NIH dataset directory
            transform: Image transforms
            split: 'train', 'val', or 'test'
            use_bbox: Whether to use bounding box annotations
            bbox_dir: Path to bounding box annotations
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.use_bbox = use_bbox
        
        # Load labels
        labels_path = self.data_dir / "Data_Entry_2017.csv"
        self.df = pd.read_csv(labels_path)
        
        # Filter by split
        split_path = self.data_dir / f"{split}_list.txt"
        if split_path.exists():
            with open(split_path, 'r') as f:
                split_images = [line.strip() for line in f]
            self.df = self.df[self.df['Image Index'].isin(split_images)]
        
        # Process labels
        self.image_paths = []
        self.labels = []
        
        for idx, row in self.df.iterrows():
            image_name = row['Image Index']
            image_path = self.data_dir / "images" / image_name
            
            if not image_path.exists():
                continue
                
            # Parse disease labels
            disease_labels = row['Finding Labels'].split('|')
            label_vector = torch.zeros(len(self.DISEASES))
            
            for disease in disease_labels:
                if disease in self.DISEASES:
                    label_vector[self.DISEASES.index(disease)] = 1.0
                elif disease != "No Finding":
                    logger.warning(f"Unknown disease: {disease}")
            
            self.image_paths.append(image_path)
            self.labels.append(label_vector)
        
        # Load bounding boxes if needed
        self.bbox_dict = {}
        if use_bbox and bbox_dir:
            self._load_bboxes(bbox_dir)
        
        logger.info(f"Loaded {len(self.image_paths)} images for split '{split}'")
    
    def _load_bboxes(self, bbox_dir: str):
        """Load bounding box annotations."""
        bbox_path = Path(bbox_dir) / "BBox_List_2017.csv"
        if bbox_path.exists():
            bbox_df = pd.read_csv(bbox_path)
            for _, row in bbox_df.iterrows():
                image_name = row['Image Index']
                if image_name not in self.bbox_dict:
                    self.bbox_dict[image_name] = []
                
                bbox = {
                    'disease': row['Finding Label'],
                    'x': row['BBox [x'],
                    'y': row['y'],
                    'width': row['w'],
                    'height': row['h']
                }
                self.bbox_dict[image_name].append(bbox)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get image, label, and optional bbox."""
        image_path = self.image_paths[idx]
        labels = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        sample = {
            'image': image,
            'labels': labels.float(),
            'image_path': str(image_path),
        }
        
        # Add bounding box if available
        if self.use_bbox:
            image_name = image_path.name
            bboxes = self.bbox_dict.get(image_name, [])
            sample['bboxes'] = bboxes
        
        return sample


class BBoxDataset(Dataset):
    """Dataset for images with bounding box annotations."""
    
    def __init__(
        self,
        data_dir: str,
        bbox_dir: str,
        transform=None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize bbox dataset.
        
        Args:
            data_dir: Path to images
            bbox_dir: Path to bounding box annotations
            transform: Image transforms
            target_size: Target size for mask generation
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Load bbox annotations
        bbox_path = Path(bbox_dir) / "BBox_List_2017.csv"
        self.bbox_df = pd.read_csv(bbox_path)
        
        # Filter for valid images
        self.valid_samples = []
        for idx, row in self.bbox_df.iterrows():
            image_path = self.data_dir / "images" / row['Image Index']
            if image_path.exists():
                self.valid_samples.append(row)
        
        logger.info(f"Loaded {len(self.valid_samples)} bbox samples")
    
    def _create_attention_mask(self, bbox_row) -> torch.Tensor:
        """Create attention mask from bounding box."""
        mask = torch.zeros(self.target_size)
        
        # Parse bbox coordinates
        x = int(bbox_row['BBox [x'])
        y = int(bbox_row['y'])
        w = int(bbox_row['w'])
        h = int(bbox_row['h'])
        
        # Ensure coordinates are within bounds
        x_end = min(x + w, self.target_size[1])
        y_end = min(y + h, self.target_size[0])
        x = max(x, 0)
        y = max(y, 0)
        
        # Fill mask
        if x < x_end and y < y_end:
            mask[y:y_end, x:x_end] = 1.0
        
        # Normalize to match attention shape (14x14 for ViT-B/16)
        # ViT uses 14x14 patches for 224x224 images (16x16 patches)
        patch_size = 16
        num_patches = self.target_size[0] // patch_size  # 14
        patch_mask = torch.zeros(num_patches, num_patches)
        
        for i in range(num_patches):
            for j in range(num_patches):
                patch_x = j * patch_size
                patch_y = i * patch_size
                patch_region = mask[
                    patch_y:patch_y + patch_size,
                    patch_x:patch_x + patch_size
                ]
                if patch_region.sum() > 0:
                    patch_mask[i, j] = 1.0
        
        return patch_mask.flatten()
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get image and attention mask."""
        bbox_row = self.valid_samples[idx]
        
        # Load image
        image_path = self.data_dir / "images" / bbox_row['Image Index']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Create attention mask
        attention_mask = self._create_attention_mask(bbox_row)
        
        return {
            'image': image,
            'attention_mask': attention_mask,
            'disease': bbox_row['Finding Label'],
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_bbox: bool = False,
    bbox_dir: Optional[str] = None,
):
    """Create train, val, test dataloaders."""
    from src.data.transforms import get_train_transforms, get_val_transforms
    
    # Create datasets
    train_dataset = ChestXRayDataset(
        data_dir=data_dir,
        transform=get_train_transforms(),
        split='train',
        use_bbox=use_bbox,
        bbox_dir=bbox_dir,
    )
    
    val_dataset = ChestXRayDataset(
        data_dir=data_dir,
        transform=get_val_transforms(),
        split='val',
        use_bbox=False,
    )
    
    test_dataset = ChestXRayDataset(
        data_dir=data_dir,
        transform=get_val_transforms(),
        split='test',
        use_bbox=False,
    )
    
    # Handle class imbalance
    if use_bbox:
        # Weighted sampling for bbox images
        weights = []
        for sample in train_dataset:
            if sample['bboxes']:
                weights.append(3.0)  # Higher weight for bbox samples
            else:
                weights.append(1.0)
        sampler = WeightedRandomSampler(weights, len(weights))
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
