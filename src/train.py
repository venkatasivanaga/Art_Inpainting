# train.py
# Single train loop for GAN inpainting (generator + boundary PatchGAN).
import argparse, os, time, yaml
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from data import make_loader, read_yaml
from model import GatedUNet, PatchDiscriminator, sample_boundary_tiles, LossBundle, l1_in_hole

def save_ckpt(path, model_g, opt_g, step, best_val):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"g": model_g.state_dict(), "opt_g": opt_g.state_dict(),
                "step": step, "best_val": best_val}, path)

def train(args):
    data_cfg = read_yaml(args.data_config)
    cfg = read_yaml(args.config)
    img_size = cfg.get("img_size", 512)
    batch = cfg.get("batch_size", 8)
    epochs = cfg.get("epochs", 80)
    band_px = cfg.get("band_px", 12)
    edge_guidance = cfg.get("edge_guidance", True)
    mp = cfg.get("mixed_precision", True)

    train_loader = make_loader(os.path.join(args.splits, "train.csv"), img_size, batch, True, augment=True)
    val_loader   = make_loader(os.path.join(args.splits, "val.csv"), img_size, batch, False, augment=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = GatedUNet(in_ch=4, base=64, edge_guidance=edge_guidance).to(device)
    D = PatchDiscriminator(in_ch=3).to(device)

    opt_g = optim.Adam(G.parameters(), lr=cfg["optim"]["g_lr"], betas=tuple(cfg["optim"]["betas"]))
    opt_d = optim.Adam(D.parameters(), lr=cfg["optim"]["d_lr"], betas=tuple(cfg["optim"]["betas"]))
    losses = LossBundle(**cfg["loss"])

    scaler_g = GradScaler(enabled=mp)
    scaler_d = GradScaler(enabled=mp)

    best_val = 1e9
    step = 0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = out_dir / "best.ckpt"

    for ep in range(1, epochs+1):
        G.train(); D.train()
        t0 = time.time()
        tr_g, tr_d = 0.0, 0.0
        for batch_i, batch in enumerate(train_loader):
            step += 1
            img = batch["image"].to(device)        # [0,1]
            mask = batch["mask"].to(device)        # {0,1}
            with autocast(enabled=mp):
                # optional edge channel = Sobel magnitude from context
                sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=img.dtype, device=device)
                sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=img.dtype, device=device)
                gray = img.mean(1, keepdim=True)
                gx = F.conv2d(gray, sobel_x, padding=1)
                gy = F.conv2d(gray, sobel_y, padding=1)
                edges = (gx.abs() + gy.abs()).clamp(0,1)

                pred = G(img, mask, edges=edges, band_px=band_px)
                comp = pred  # already blended inside

                # ----- D update (hinge) on boundary tiles -----
                opt_d.zero_grad(set_to_none=True)
                real_tiles = sample_boundary_tiles(img, mask, tile=96, k=4)
                fake_tiles = sample_boundary_tiles(comp.detach(), mask, tile=96, k=4)
                real_logit = D(real_tiles)
                fake_logit = D(fake_tiles)
                d_loss = losses.adv_hinge_d(real_logit, fake_logit)

            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ----- G update -----
            with autocast(enabled=mp):
                opt_g.zero_grad(set_to_none=True)
                # rebuild band mask for style
                kernel = torch.ones(1,1,band_px*2+1, band_px*2+1, device=device)
                dil = (F.conv2d(mask, kernel, padding=band_px) > 0).float()
                band = torch.clamp(dil - mask, 0, 1)

                # losses
                l_l1 = losses.masked_l1(comp, img, mask, band_px=band_px)
                l_perc = losses.perceptual(comp, img)
                l_style = losses.style(comp, img, band)
                fake_tiles = sample_boundary_tiles(comp, mask, tile=96, k=4)
                g_adv = losses.adv_hinge_g(D(fake_tiles))
                l_tv = losses.tv(comp, mask)
                g_loss = (losses.lambda_l1*l_l1 + losses.lambda_perc*l_perc +
                          losses.lambda_style*l_style + losses.lambda_adv*g_adv +
                          losses.lambda_tv*l_tv)

            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            tr_g += g_loss.item()
            tr_d += d_loss.item()

        # validation (L1-in-hole as quick proxy)
        G.eval()
        val_metric = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=img.dtype, device=device)
                sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=img.dtype, device=device)
                gray = img.mean(1, keepdim=True)
                gx = F.conv2d(gray, sobel_x, padding=1)
                gy = F.conv2d(gray, sobel_y, padding=1)
                edges = (gx.abs() + gy.abs()).clamp(0,1)
                comp = G(img, mask, edges=edges, band_px=cfg.get("band_px",12))
                val_metric += l1_in_hole(comp, img, mask).item()
                val_batches += 1
        val_metric /= max(1, val_batches)

        print(f"[ep {ep:03d}] train G {tr_g/len(train_loader):.4f} | D {tr_d/len(train_loader):.4f} "
              f"| val L1(hole) {val_metric:.4f} | {time.time()-t0:.1f}s")

        if val_metric < best_val:
            best_val = val_metric
            save_ckpt(ckpt_best, G, opt_g, step, best_val)
            print(f"  -> saved {ckpt_best} (best val {best_val:.4f})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-config", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--splits", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()
    train(args)
