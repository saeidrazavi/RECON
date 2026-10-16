"""MS-COCO 256 dataset for MMDiT: the arrow splits of preprocessing/extract_mscoco_features.py.

Columns:
    id       : int32   (COCO image id)
    image    : Image   (256x256 RGB, U-ViT center-crop + Lanczos)
    captions : [string]                    (all COCO captions, ann-id order)
    moments  : Array3D(2C, H, W) float32   (VAE posterior mean/std concat; (8, 32, 32) for SD-VAE)
    contexts : Array3D(None, 77, 768) f16  (CLIP ViT-L/14 text last_hidden_state)
The dataset root also has `empty_context.npy` (77, 768) float32 (CLIP("")) and `vae_name.txt`
(the --vae-name of the moments).
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk


def features_vae_name(root):
    """The --vae-name the features in root were extracted with. Features without vae_name.txt
    (U-ViT's preprocessing, as hosted on the Hub) hold SD-VAE latents."""
    path = os.path.join(root, "vae_name.txt")
    if not os.path.exists(path):
        return "sdvae-ft-ema-f8d4"
    with open(path) as f:
        return f.read().strip()


class MSCOCOFeatureDataset(Dataset):
    """One split. Returns (raw_image_uint8, moments, context, raw_caption)
    with a caption picked uniformly at random per access, as in U-ViT/REPA."""

    def __init__(self, root):
        self.root = root
        self.ds = load_from_disk(root)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        x = np.array(row["image"].convert("RGB")).transpose(2, 0, 1)  # uint8 CHW
        z = np.asarray(row["moments"], dtype=np.float32)
        contexts = np.asarray(row["contexts"], dtype=np.float32)
        k = random.randint(0, contexts.shape[0] - 1)
        c = contexts[k]
        caption = row["captions"][k]
        return torch.from_numpy(x), torch.from_numpy(z), torch.from_numpy(c), caption


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, z, y, caption = self.dataset[item]
        if random.random() < self.p_uncond:
            y = torch.from_numpy(self.empty_token.copy())
            caption = ""
        return x, z, y, caption


class MSCOCO256Features:
    """Factory matching REPA's MSCOCO256Features(path, ...) interface."""

    TRAIN_SIZE = 82783   # COCO train2014
    VAL_SIZE = 40504     # COCO val2014

    def __init__(self, path, cfg=True, p_uncond=0.1, mode="train", ret_caption=True):
        self.empty_context = np.load(os.path.join(path, "empty_context.npy")).astype(np.float32)
        assert self.empty_context.shape == (77, 768), self.empty_context.shape
        self.train = None
        self.test = None
        if mode == "val":
            self.test = MSCOCOFeatureDataset(os.path.join(path, "val"))
            assert len(self.test) == self.VAL_SIZE, len(self.test)
        else:
            self.train = MSCOCOFeatureDataset(os.path.join(path, "train"))
            assert len(self.train) == self.TRAIN_SIZE, len(self.train)
            if cfg:  # classifier free guidance
                assert p_uncond is not None
                self.train = CFGDataset(self.train, p_uncond, self.empty_context)
