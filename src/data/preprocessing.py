"""Data preprocessing utilities."""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from typing import Tuple, Optional, List
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocess chest X-ray data."""
    
    def __init__(self, data_dir: str, output_dir: str):
        """Initialize preprocessor.
        
        Args:
            data_dir: Raw data directory
            output_dir: Processed output directory
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_intensity(
        self,
        image: np.ndarray,
        percentile_low: float = 0.5,
        percentile_high: float = 99.5,
    ) -> np.ndarray:
        """Normalize image intensity.
        
        Args:
            image: Input image array
            percentile_low: Lower percentile for clipping
            percentile_high: Upper percentile for clipping
        
        Returns:
            Normalized image
        """
        p_low = np.percentile(image, percentile_low)
        p_high = np.percentile(image, percentile_high)
        
        image_clipped = np.clip(image, p_low, p_high)
        image_normalized = (image_clipped - p_low) / (p_high - p_low)
        
        return image_normalized
    
    def create_patch_masks(
        self,
        bbox_coords: List[Tuple[int, int, int, int]],
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
    ) -> np.ndarray:
        """Create patch-level attention masks from bounding boxes.
        
        Args:
            bbox_coords: List of (x, y, w, h) bounding boxes
            image_size: (height, width) of image
            patch_size: Size of each patch
        
        Returns:
            Patch-level attention mask
        """
        num_patches_h = image_size[0] // patch_size
        num_patches_w = image_size[1] // patch_size
        patch_mask = np.zeros((num_patches_h, num_patches_w))
        
        for x, y, w, h in bbox_coords:
            # Convert bbox to patch coordinates
            start_patch_x = max(0, x // patch_size)
            start_patch_y = max(0, y // patch_size)
            end_patch_x = min(num_patches_w, (x + w) // patch_size + 1)
            end_patch_y = min(num_patches_h, (y + h) // patch_size + 1)
            
            patch_mask[start_patch_y:end_patch_y, start_patch_x:end_patch_x] = 1.0
        
        return patch_mask.flatten()
    
    def compute_class_weights(self, labels_df: pd.DataFrame) -> torch.Tensor:
        """Compute class weights to handle imbalance.
        
        Args:
            labels_df: DataFrame with disease labels
        
        Returns:
            Class weights tensor
        """
        pos_counts = labels_df.sum()
        neg_counts = len(labels_df) - pos_counts
        
        # Compute positive weights: neg_count / pos_count
        pos_weights = neg_counts / pos_counts
        
        # Cap at 637 (Hernia has 637:1 ratio)
        pos_weights = np.minimum(pos_weights, 637)
        
        # Normalize
        pos_weights = pos_weights / pos_weights.mean()
        
        return torch.FloatTensor(pos_weights)
    
    def process_nih_dataset(self):
        """Process NIH dataset."""
        logger.info("Processing NIH dataset...")
        
        # Load labels
        labels_path = self.data_dir / "Data_Entry_2017.csv"
        df = pd.read_csv(labels_path)
        
        # Compute class weights
        disease_columns = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        # One-hot encode labels
        for disease in disease_columns:
            df[disease] = df['Finding Labels'].str.contains(disease).astype(int)
        
        class_weights = self.compute_class_weights(df[disease_columns])
        
        # Save class weights
        weights_path = self.output_dir / "class_weights.pt"
        torch.save(class_weights, weights_path)
        
        logger.info(f"Class weights saved to {weights_path}")
        logger.info(f"Weights: {class_weights.numpy()}")
        
        return class_weights
    
    def create_dataset_splits(self, test_ratio: float = 0.1, val_ratio: float = 0.1):
        """Create train/val/test splits.
        
        Args:
            test_ratio: Proportion for test set
            val_ratio: Proportion for validation set
        """
        labels_path = self.data_dir / "Data_Entry_2017.csv"
        df = pd.read_csv(labels_path)
        
        # Get unique patients
        patients = df['Patient ID'].unique()
        np.random.shuffle(patients)
        
        n_test = int(len(patients) * test_ratio)
        n_val = int(len(patients) * val_ratio)
        
        test_patients = set(patients[:n_test])
        val_patients = set(patients[n_test:n_test + n_val])
        train_patients = set(patients[n_test + n_val:])
        
        # Create split files
        splits = {
            'train': train_patients,
            'val': val_patients,
            'test': test_patients,
        }
        
        for split_name, patient_set in splits.items():
            split_df = df[df['Patient ID'].isin(patient_set)]
            image_list = split_df['Image Index'].tolist()
            
            split_path = self.output_dir / f"{split_name}_list.txt"
            with open(split_path, 'w') as f:
                for image_name in image_list:
                    f.write(f"{image_name}\n")
            
            logger.info(f"{split_name}: {len(image_list)} images")
