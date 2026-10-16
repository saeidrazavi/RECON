"""BLIP3o WebDataset loader for training DDT (see preprocessing/README.md for the shards)."""

from functools import partial
from pathlib import Path

import webdataset as wds
from torchvision import transforms


def _decode_blip3o(transform, sample):
    """(image tensor in [0, 1] or None, caption) from a decoded BLIP3o sample."""
    image = sample.get("jpg") or sample.get("png") or sample.get("jpeg") or sample.get("webp")
    caption = sample.get("txt", b"")
    caption = caption.decode("utf-8").strip() if isinstance(caption, bytes) else str(caption).strip()
    return (transform(image) if image is not None else None), caption


def _has_image(sample):
    return sample[0] is not None


def blip3o_loader(data_dir, splits, batch_size, num_workers, epoch, steps_per_epoch, seed=42,
                  shuffle_buffer=10000, image_size=256):
    """Batches of (images in [0, 1], captions) from the BLIP3o shards <data_dir>/<split>/*.tar, for
    one virtual epoch of steps_per_epoch batches per rank. The shards are split across ranks and
    dataloader workers, as in DiffusionBench (src/data/blip3o_wds_dataset.py), and the shards and
    samples are shuffled with seed + epoch. DiffusionBench leaves the sample shuffle unseeded, so
    webdataset seeds it from the process id and the clock and no two runs see the same batches."""
    shards = [str(f) for split in splits for f in sorted(Path(data_dir, split).glob("*.tar"))]
    assert shards, f"no .tar shards for splits {splits} in {data_dir}"
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    dataset = (
        wds.WebDataset(shards, nodesplitter=wds.split_by_node, shardshuffle=1000, seed=seed + epoch)
        .shuffle(shuffle_buffer, initial=shuffle_buffer // 2, seed=seed + epoch)
        .decode("pil", handler=wds.ignore_and_continue)  # skip corrupt images
        .map(partial(_decode_blip3o, transform), handler=wds.ignore_and_continue)
        .select(_has_image)
    )
    loader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return loader.with_epoch(steps_per_epoch)
