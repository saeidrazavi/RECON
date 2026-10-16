"""Preprocess MS-COCO 2014 into arrow datasets for MMDiT, as in U-ViT (https://github.com/baofff/U-ViT):
256x256 center crops, VAE moments (--vae-name, recorded in <out>/vae_name.txt) and CLIP ViT-L/14 text features.

Prebuilt output: https://huggingface.co/datasets/SaeedRazavi/mscoco256-t2i-features (features/)

Usage:
    uv run python preprocessing/extract_mscoco_features.py --split train \
        --coco-root data/coco --out data/mscoco256/features
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recon.models.vae import HF_VAE_DS_FACTOR, HF_VAE_MAPPER

from datasets import Dataset as HFDataset
from datasets import Features, Value, Sequence, Array3D
from datasets import Image as HFImage

EXPECTED = {"train": 82783, "val": 40504}


def center_crop(width, height, img):
    """U-ViT datasets.py center_crop, verbatim."""
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except Exception:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


def load_annotations(ann_file):
    """Return list of (image_id, file_name, [captions]) sorted by image id,
    captions sorted by annotation id (pycocotools-free)."""
    with open(ann_file) as f:
        ann = json.load(f)
    file_names = {im["id"]: im["file_name"] for im in ann["images"]}
    caps = {im_id: [] for im_id in file_names}
    for a in ann["annotations"]:
        caps[a["image_id"]].append((a["id"], a["caption"]))
    out = []
    for im_id in sorted(file_names):
        cc = [c for _, c in sorted(caps[im_id])]
        assert len(cc) > 0, f"image {im_id} has no captions"
        out.append((im_id, file_names[im_id], cc))
    return out


@torch.no_grad()
def encode_clip(text_model, tokenizer, captions, device):
    """U-ViT libs/clip.py FrozenCLIPEmbedder.encode."""
    batch_encoding = tokenizer(
        captions, truncation=True, max_length=77, return_length=True,
        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"].to(device)
    outputs = text_model(input_ids=tokens)
    return outputs.last_hidden_state  # (B, 77, 768) float32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--coco-root", type=str, required=True,
                        help="dir containing train2014/, val2014/, annotations/")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--vae-name", type=str, default="sdvae-ft-ema-f8d4", choices=list(HF_VAE_MAPPER))
    parser.add_argument("--clip", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-shards", type=int, default=64,
                        help="arrow shards written by save_to_disk")
    parser.add_argument("--limit", type=int, default=0,
                        help="debug: only process the first N images (skips count assert)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = args.resolution
    latent = res // HF_VAE_DS_FACTOR[args.vae_name]

    from diffusers.models import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    vae = AutoencoderKL.from_pretrained(HF_VAE_MAPPER[args.vae_name]).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.clip)
    text_model = CLIPTextModel.from_pretrained(args.clip).to(device).eval()

    img_dir = os.path.join(args.coco_root, f"{args.split}2014")
    ann_file = os.path.join(args.coco_root, "annotations", f"captions_{args.split}2014.json")
    records = load_annotations(ann_file)
    if not args.limit:
        assert len(records) == EXPECTED[args.split], \
            f"{args.split}: expected {EXPECTED[args.split]} images, got {len(records)}"
    else:
        records = records[:args.limit]
    print(f"[{args.split}] {len(records)} images from {img_dir}")

    os.makedirs(args.out, exist_ok=True)

    # the VAE of the moments, read by t2i_train.py; both splits must use the same one
    vae_path = os.path.join(args.out, "vae_name.txt")
    if os.path.exists(vae_path):
        with open(vae_path) as f:
            assert f.read().strip() == args.vae_name, f"{args.out} holds features of another VAE ({vae_path})"
    else:
        with open(vae_path, "w") as f:
            f.write(args.vae_name + "\n")

    # empty context for CFG (written once, tiny)
    empty_path = os.path.join(args.out, "empty_context.npy")
    if not os.path.exists(empty_path):
        empty = encode_clip(text_model, tokenizer, [""], device)[0].cpu().numpy().astype(np.float32)
        np.save(empty_path, empty)
        print(f"saved {empty_path} {empty.shape}")

    features = Features({
        "id": Value("int32"),
        "image": HFImage(),
        "captions": Sequence(Value("string")),
        "moments": Array3D(shape=(2 * vae.config.latent_channels, latent, latent), dtype="float32"),
        "contexts": Array3D(shape=(None, 77, 768), dtype="float16"),
    })

    def generator():
        for start in tqdm(range(0, len(records), args.batch_size), desc=f"extract-{args.split}"):
            batch = records[start:start + args.batch_size]
            pixels, pils = [], []
            for im_id, fname, caps in batch:
                img = Image.open(os.path.join(img_dir, fname)).convert("RGB")
                arr = center_crop(res, res, np.array(img).astype(np.uint8))
                pils.append(Image.fromarray(arr))  # stored losslessly (PNG) in arrow
                pixels.append(arr)
            x = torch.from_numpy(np.stack(pixels)).to(device)
            x = (x.permute(0, 3, 1, 2).float() / 127.5) - 1.0  # U-ViT: [-1, 1]
            with torch.no_grad():
                posterior = vae.encode(x).latent_dist
                moments = torch.cat([posterior.mean, posterior.std], dim=1).cpu().numpy().astype(np.float32)

                flat_caps = [c for _, _, caps in batch for c in caps]
                ctx = encode_clip(text_model, tokenizer, flat_caps, device).cpu().numpy().astype(np.float16)

            ofs = 0
            for (im_id, fname, caps), pil, mom in zip(batch, pils, moments):
                yield {
                    "id": im_id,
                    "image": pil,
                    "captions": caps,
                    "moments": mom,
                    "contexts": ctx[ofs:ofs + len(caps)],
                }
                ofs += len(caps)
            assert ofs == len(flat_caps)

    # keep the from_generator cache next to the output (not in HF_HOME) and drop it after save
    cache_dir = os.path.join(args.out, f".cache_{args.split}")
    ds = HFDataset.from_generator(
        generator, features=features, cache_dir=cache_dir, writer_batch_size=200,
    )
    out_dir = os.path.join(args.out, args.split)
    ds.save_to_disk(out_dir, num_shards=args.num_shards)
    print(f"saved {len(ds)} rows to {out_dir}")
    del ds
    import gc, shutil
    gc.collect()
    shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
