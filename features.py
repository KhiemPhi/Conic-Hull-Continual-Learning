"""
features.py
-----------
Image transforms, single-image feature extraction, and batch feature
extraction over a full torchvision test split.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from typing import Dict, List, Optional
from tqdm import tqdm


def get_default_transform(img_size: int = 224) -> transforms.Compose:
    """Standard ImageNet normalisation transform (used for val/test)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def extract_features(
    model:     nn.Module,
    image:     "Image.Image | np.ndarray | torch.Tensor | str",
    transform: Optional[transforms.Compose] = None,
    device:    Optional[str]                = None,
) -> np.ndarray:
    """
    Extract the final-layer feature vector from a single image.

    Accepts: PIL Image, NumPy array (H×W×C uint8), torch.Tensor (C×H×W),
             or a file-path string.

    Returns
    -------
    np.ndarray of shape (feature_dim,)
    """
    if device is None:
        device = next(model.parameters()).device
    if transform is None:
        transform = get_default_transform()

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    if isinstance(image, Image.Image):
        tensor = transform(image).unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        tensor = image if image.ndim == 4 else image.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    tensor = tensor.to(device)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze(0).cpu().numpy()


def build_feature_dict(
    model:        nn.Module,
    dataset_root: str           = "./data",
    dataset_name: str           = "CIFAR10",
    batch_size:   int           = 64,
    num_workers:  int           = 2,
    device:       Optional[str] = None,
    train:        bool          = False,
) -> Dict[str, np.ndarray]:
    """
    Download a torchvision dataset, run inference on the test split, and
    return a dict mapping class name → feature matrix (N_class, D).

    Supported datasets: 'CIFAR10', 'CIFAR100', 'STL10'
    """
    if device is None:
        device = next(model.parameters()).device

    transform = get_default_transform()
    name = dataset_name.upper()

    if name == "CIFAR10":
        ds = datasets.CIFAR10(dataset_root, train=train,
                              download=True, transform=transform)
        class_names = ds.classes
    elif name == "CIFAR100":
        ds = datasets.CIFAR100(dataset_root, train=train,
                               download=True, transform=transform)
        class_names = ds.classes
    elif name == "STL10":
        ds = datasets.STL10(dataset_root, split="test",
                            download=True, transform=transform)
        class_names = ds.classes
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. "
                         f"Choose from CIFAR10, CIFAR100, STL10.")

    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    all_features: List[np.ndarray] = []
    all_labels:   List[int]        = []

    print(f"[build_test_feature_dict] {len(ds)} images  "
          f"({len(class_names)} classes)  dataset={dataset_name}")

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc="Extracting features",
                    unit="batch", dynamic_ncols=True)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            feats = model(imgs)
            all_features.append(feats.cpu().numpy())
            all_labels.extend(labels.tolist())
            pbar.set_postfix({"samples": len(all_labels)})

    all_features_np = np.concatenate(all_features, axis=0)
    all_labels_np   = np.array(all_labels)

    feature_dict: Dict[str, np.ndarray] = {}
    for idx, cls in enumerate(class_names):
        mask = all_labels_np == idx
        feature_dict[cls] = all_features_np[mask]
        print(f"  '{cls}': {mask.sum()} samples  shape={feature_dict[cls].shape}")

    return feature_dict
