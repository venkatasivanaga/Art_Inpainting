# infer.py
# Batch/single inference from a CSV or folder; optional heatmap saving.
import argparse, os, csv
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from data import ImageMaskDataset
from model import GatedUNet

def to_pil(x):
    x = (x.clamp(0,1)*255).byte().permute(1,2,0).cpu().numpy()
    return Image.fromarray(x)

@torch.no_grad()
def run(ckpt, csv_path, save_dir, heatmap=False, img_size=512, band_px=12, edge_guidance=True, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = GatedUNet(in_ch=4, base=64, edge_guidance=edge_guidance).to(device)
    state = torch.load(ckpt, map_location=device)
    G.load_state_dict(state["g"]); G.eval()

    ds = ImageMaskDataset(csv_path, img_size=img_size, augment=False)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for i in range(len(ds)):
        item = ds[i]
        img = item["image"].unsqueeze(0).to(device)
        mask = item["mask"].unsqueeze(0).to(device)
        # simple edge map
        gray = img.mean(1, keepdim=True)
        sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=img.dtype, device=device)
        sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=img.dtype, device=device)
        gx = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
        edges = (gx.abs()+gy.abs()).clamp(0,1)

        out = G(img, mask, edges=edges, band_px=band_px)[0]
        ipath = item["image_path"]
        name = Path(ipath).with_suffix("").name
        to_pil(out).save(Path(save_dir)/f"{name}_restored.png")

        if heatmap:
            hm = (out - item["image"]).abs().mean(0, keepdim=True)  # [1,H,W]
            hm = (hm / (hm.max().clamp_min(1e-6))).repeat(3,1,1)
            to_pil(hm).save(Path(save_dir)/f"{name}_heatmap.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--save", type=str, required=True)
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--band-px", type=int, default=12)
    args = ap.parse_args()
    run(args.ckpt, args.csv, args.save, args.heatmap, args.img_size, args.band_px)
