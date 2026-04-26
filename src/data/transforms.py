"""Data augmentation and transformation pipelines."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Tuple


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> A.Compose:
    """Get training data augmentations.
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.RandomRotate90(p=0.3),
        A.Rotate(limit=10, p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=5,
            p=0.3,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
) -> A.Compose:
    """Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_inference_transform(
    image_size: Tuple[int, int] = (224, 224),
) -> A.Compose:
    """Get inference transforms.
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
