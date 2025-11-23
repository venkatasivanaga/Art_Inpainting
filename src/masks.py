# masks.py
# Generate irregular and crack-like (Canny/Frangi) masks and save PNGs mirroring image names.
import argparse, os, sys, glob
from pathlib import Path
import numpy as np
import cv2
from skimage.filters import frangi, meijering
from skimage.morphology import skeletonize, dilation, disk
from PIL import Image

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def save_mask(mask, out_path):
    mask = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_path)

def random_irregular_mask(h, w):
    mask = np.zeros((h, w), np.uint8)
    # random strokes
    num_strokes = np.random.randint(8, 16)
    for _ in range(num_strokes):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        for _ in range(np.random.randint(10, 40)):
            x2 = np.clip(x1 + np.random.randint(-30, 31), 0, w-1)
            y2 = np.clip(y1 + np.random.randint(-30, 31), 0, h-1)
            thickness = np.random.randint(8, 24)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
            x1, y1 = x2, y2
    # random holes
    for _ in range(np.random.randint(2, 5)):
        rx, ry = np.random.randint(20, 80), np.random.randint(20, 80)
        cx, cy = np.random.randint(rx, w-rx), np.random.randint(ry, h-ry)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    return (mask > 0).astype(np.uint8)

def crack_mask_from_edges(img, canny_low=50, canny_high=150, use_frangi=True, widen=2):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    can = cv2.Canny(g, canny_low, canny_high)  # [0,255]
    can = (can > 0).astype(np.uint8)
    if use_frangi:
        # Frangi returns float [0,1]
        ff = frangi(g / 255.0)
        ff = (ff > np.percentile(ff, 85)).astype(np.uint8)
        edges = np.clip(can + ff, 0, 1)
    else:
        edges = can
    # skeletonize then widen to get crack width 1-4px
    sk = skeletonize(edges.astype(bool)).astype(np.uint8)
    if widen > 0:
        sk = dilation(sk, footprint=disk(widen))
    return sk.astype(np.uint8)

def process_folder(image_root, out_masks, per_image=2, edge="crack",
                   canny_low=50, canny_high=150, frangi_flag=True):
    ensure_dir(Path(out_masks))
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    img_paths = []
    for e in exts:
        img_paths += glob.glob(str(Path(image_root) / "**" / e), recursive=True)
    img_paths = sorted(img_paths)
    print(f"[masks] Found {len(img_paths)} images under {image_root}")

    for p in img_paths:
        try:
            img = load_rgb(p)
            h, w = img.shape[:2]
            stem = Path(p).with_suffix("").name
            for i in range(per_image):
                m_ir = random_irregular_mask(h, w)
                out_ir = Path(out_masks) / f"{stem}_ir{i}.png"
                save_mask(m_ir * 255, out_ir)

            if edge == "crack":
                m_ed = crack_mask_from_edges(img, canny_low, canny_high, frangi_flag, widen=2)
                out_ed = Path(out_masks) / f"{stem}_crack.png"
                save_mask(m_ed * 255, out_ed)
        except Exception as e:
            print(f"[warn] {p}: {e}", file=sys.stderr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", type=str, required=False, help="unused here; kept for symmetry")
    ap.add_argument("--image-root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--per-image", type=int, default=2)
    ap.add_argument("--edge", type=str, default="crack")
    ap.add_argument("--canny-low", type=int, default=50)
    ap.add_argument("--canny-high", type=int, default=150)
    ap.add_argument("--frangi", action="store_true")
    args = ap.parse_args()
    process_folder(args.image_root, args.out, args.per_image, args.edge,
                   args.canny_low, args.canny_high, args.frangi)
