"""
Data loading and preprocessing module for Plant Disease Detection.
Handles dataset loading, train/val/test splits, normalization, and data augmentation.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard ImageNet statistics (compatible with MobileNetV2 / ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build training (with rotation, flips, zoom/crop) and evaluation transforms.
    """
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        # Zoom via RandomResizedCrop
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        # Rotation
        transforms.RandomRotation(degrees=30),
        # Flips
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        # Color Jitter
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def get_data_loaders(
    data_dir: str or Path,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Loads train, validation, and test datasets from folder structure:
      data_dir/train/<class_name>/*.jpg
      data_dir/val/<class_name>/*.jpg
      data_dir/test/<class_name>/*.jpg
    Returns:
      train_loader, val_loader, test_loader, class_names
    """
    data_path = Path(data_dir).resolve()
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found at: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found at: {val_dir}")

    train_transform, eval_transform = get_transforms(image_size=image_size)

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=eval_transform)

    # If test folder doesn't exist, fallback gracefully to val
    if test_dir.exists() and any(test_dir.iterdir()):
        test_dataset = datasets.ImageFolder(root=str(test_dir), transform=eval_transform)
    else:
        print("[!] Warning: test folder missing or empty; using validation set for test.")
        test_dataset = val_dataset

    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_names


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized PyTorch tensor (C, H, W) back to an RGB image array (H, W, C) in [0, 255].
    """
    import numpy as np
    img = tensor.cpu().detach().clone()
    for c, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        img[c] = img[c] * s + m
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).numpy()
    return (img_np * 255).astype(np.uint8)
