"""
cached_cifar.py
---------------
Disk-cached, pre-resized CIFAR datasets.

The pipeline upsamples every 32x32 CIFAR image to 224x224 on the CPU on *every*
pass — training, per-stage hull fitting, replay/stat snapshots, and evaluation.
That bilinear resize (not the GPU) dominates wall-clock: each full-dataset pass
re-resizes 50k/10k images from scratch.

This caches the *resized uint8 images* once to a memmapped .npy.  Subsequent
passes skip the resize and only do the cheap ToTensor + Normalize at access
time.  Because the cache stores images *before* normalization, the SAME file is
reused across different Normalize stats (the pipeline uses (0.5,0.5); features.py
uses ImageNet stats) — just pass different mean/std.

It is a drop-in for torchvision's CIFAR100/CIFAR10: exposes `.targets` and
`.classes`, and yields (normalized_tensor, label) bit-for-bit identical to
transforms.Compose([Resize(size), ToTensor(), Normalize(mean, std)]) (the resize
is done with the same torchvision transform, so features are unchanged).
"""
import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class CachedResizeCIFAR(Dataset):
    def __init__(
        self,
        dataset_cls,
        root: str = "./data",
        train: bool = True,
        size: int = 224,
        mean: Sequence[float] = (0.5,),
        std: Sequence[float] = (0.5,),
        cache_dir: str = None,
        download: bool = True,
    ):
        self.size = size
        # Single-value stats broadcast across channels (Normalize((0.5,),(0.5,)));
        # 3-tuples become (3,1,1) tensors (ImageNet stats).
        self.mean = float(mean[0]) if len(mean) == 1 else torch.tensor(mean).view(-1, 1, 1)
        self.std = float(std[0]) if len(std) == 1 else torch.tensor(std).view(-1, 1, 1)

        base = dataset_cls(root=root, train=train, download=download)  # raw PIL, no transform
        self.targets = list(base.targets)
        self.classes = list(base.classes)

        name = dataset_cls.__name__.lower()
        split = "train" if train else "test"
        cdir = cache_dir or root
        os.makedirs(cdir, exist_ok=True)
        self._path = os.path.join(cdir, f"{name}_{split}_r{size}_u8.npy")

        if os.path.exists(self._path):
            self.data = np.load(self._path, mmap_mode="r")
            if self.data.shape != (len(self.targets), size, size, 3):
                # stale/incompatible cache → rebuild
                self.data = self._build(base)
        else:
            self.data = self._build(base)

    def _build(self, base) -> np.ndarray:
        n = len(base)
        arr = np.empty((n, self.size, self.size, 3), dtype=np.uint8)
        resize = transforms.Resize(self.size)  # same op as the live pipeline
        for i in tqdm(range(n), desc=f"caching resize->{self.size}", unit="img"):
            img, _ = base[i]                       # PIL 32x32
            arr[i] = np.asarray(resize(img))       # PIL 224x224 -> HWC uint8
        tmp = self._path + ".tmp.npy"           # np.save keeps .npy as-is
        np.save(tmp, arr)
        os.replace(tmp, self._path)             # atomic publish
        return np.load(self._path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(self.data[i]))  # writable copy of HWC uint8
        x = x.permute(2, 0, 1).float().div_(255.0)               # CHW float [0,1] == ToTensor
        x = (x - self.mean) / self.std                            # == Normalize
        return x, self.targets[i]
