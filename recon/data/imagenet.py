from datasets import load_from_disk
from torch.utils.data import Dataset
import os
import torch
import numpy as np


def _split_path(base, split):
    # In this data layout the train split lives at the dataset root and the
    # val split (when present) is a `val/` subdir saved via save_to_disk.
    return base if split != "val" else os.path.join(base, "val")


class HFImgLatentDataset(Dataset):
    def __init__(self, vae_name, data_dir, split="train"):
        self.img_dataset = load_from_disk(_split_path(os.path.join(data_dir, "imagenet-latents-images"), split))
        self.latent_dataset = load_from_disk(_split_path(os.path.join(data_dir, f"imagenet-latents-{vae_name}"), split))
        assert len(self.img_dataset) == len(self.latent_dataset), "Image and latent dataset must have the same length"

    def __getitem__(self, idx):
        img_elem = self.img_dataset[idx]
        image, label = img_elem["image"], img_elem["label"]
        image = np.array(image.convert("RGB")).transpose(2, 0, 1)
        latent = self.latent_dataset[idx]["data"]
        return torch.from_numpy(image), torch.tensor(latent), torch.tensor(label)

    def __len__(self):
        return len(self.img_dataset)
