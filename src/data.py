# data.py
# CSV split builder + PyTorch Dataset that emits (image, mask) tensors.
from PIL import Image      # <-- add this at module level
import numpy as np
import albumentations as A
import cv2
import argparse, os, glob, csv, random
from pathlib import Path
from typing import Optional, Tuple
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

def read_yaml(p):
    with open(p, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(root):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff")
    out = []
    for e in exts:
        out += glob.glob(str(Path(root) / "**" / e), recursive=True)
    return sorted(out)

def find_mask_for_image(img_path: str, masks_root: str) -> Optional[str]:
    """Match by stem; prefer *_crack.png else any *_ir*.png."""
    stem = Path(img_path).with_suffix("").name
    crack = Path(masks_root).glob(f"**/{stem}_crack.png")
    crack = list(crack)
    if crack:
        return str(crack[0])
    irr = list(Path(masks_root).glob(f"**/{stem}_ir*.png"))
    if irr:
        return str(random.choice(irr))
    return None

def build_splits(raw_root: str, masks_root: str, out_dir: str, train=0.8, val=0.1, test=0.1, seed=42):
    random.seed(seed)
    imgs = list_images(raw_root)
    ensure_dir(Path(out_dir))
    rows = []
    for p in imgs:
        m = find_mask_for_image(p, masks_root)
        if m is None:
            continue
        rows.append((p, m))
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n*train)
    n_val = int(n*val)
    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train+n_val]
    test_rows = rows[n_train+n_val:]
    for name, subset in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        with open(Path(out_dir)/f"{name}.csv","w",newline="") as f:
            w = csv.writer(f); w.writerow(["image_path","mask_path"]); w.writerows(subset)
    print(f"[splits] wrote {len(train_rows)} train, {len(val_rows)} val, {len(test_rows)} test")

class ImageMaskDataset(Dataset):
    def __init__(self, csv_file: str, img_size: int = 512, augment: bool = True):
        self.paths = []
        with open(csv_file,"r") as f:
            for i, row in enumerate(csv.DictReader(f)):
                self.paths.append((row["image_path"], row["mask_path"]))
        self.img_size = img_size
        if augment:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size, min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0),
                    mask_value=0,
                ),

                A.HorizontalFlip(p=0.5),
                A.ColorJitter(p=0.2, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
                ToTensorV2(),
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=(0,0,0)),
                ToTensorV2(),
            ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        ip, mp = self.paths[idx]
        img = np.array(Image.open(ip).convert("RGB"))
        mask = np.array(Image.open(mp).convert("L"))
        mask = (mask > 127).astype(np.uint8)  # 0/1
        # apply same spatial transform to both
        r = self.tf(image=img, mask=mask)
        img_t = r["image"].float() / 255.0               # [3,H,W]
        # ...
        m = r["mask"]  # could be np.ndarray, PIL, or torch.Tensor

        # If Albumentations already gave you a torch.Tensor, keep it.
        if torch.is_tensor(m):
            mask_t = m
        else:
            # if it's PIL → to np
            try:
                from PIL import Image
                if isinstance(m, Image.Image):
                    m = np.array(m)
            except Exception:
                pass
            # now ensure numpy → tensor
            mask_t = torch.from_numpy(m)

        # shape to [1,H,W], float in {0,1}
        if mask_t.ndim == 3:
            # if it came as [H,W,1] or [H,W,3], squeeze/channel-pick
            if mask_t.shape[-1] > 1:
                # take first channel if mask has 3 channels
                mask_t = mask_t[..., 0]
            mask_t = mask_t.permute(2, 0, 1) if mask_t.ndim == 3 else mask_t
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

        mask_t = mask_t.float()
        # Normalize to 0/1 if values are 0..255
        if mask_t.max() > 1.5:
            mask_t = mask_t / 255.0
        # binarize (optional but safer)
        mask_t = (mask_t > 0.5).float()

        return {"image": img_t, "mask": mask_t, "image_path": ip, "mask_path": mp}

def make_loader(csv_file, img_size, batch, shuffle, num_workers=4, augment=True):
    ds = ImageMaskDataset(csv_file, img_size=img_size, augment=augment)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", type=str, required=True)
    ap.add_argument("--make-splits", action="store_true")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="data/splits")
    args = ap.parse_args()
    cfg = read_yaml(args.data_config)
    if args.make_splits:
        build_splits(cfg["raw_root"], cfg["masks_root"], args.out, args.train, args.val, args.test)
