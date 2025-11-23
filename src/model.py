# model.py
# Gated-UNet generator, boundary PatchGAN discriminator, losses, simple metrics.
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Helpers ----------
def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    x = feat.view(b, c, -1)
    g = torch.bmm(x, x.transpose(1, 2)) / (c * h * w)
    return g

def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, :-1, :] - x[:, :, 1:, :]).abs().mean() + (x[:, :, :, :-1] - x[:, :, :, 1:]).abs().mean()

def try_lpips():
    try:
        import lpips  # type: ignore
        return lpips.LPIPS(net='vgg')
    except Exception:
        return None

# ---------- Gated conv ----------
class GatedConv2d(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.feat = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
        self.gate = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        f = self.feat(x)
        g = torch.sigmoid(self.gate(x))
        return self.act(f) * g

class Down(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.block = nn.Sequential(
            GatedConv2d(in_c, out_c, k, 2, k//2),
            GatedConv2d(out_c, out_c, k, 1, k//2),
        )
    def forward(self, x): return self.block(x)

class Up(nn.Module):
    def __init__(self, in_c, out_c, k=3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            GatedConv2d(out_c*2, out_c, k, 1, k//2),
            GatedConv2d(out_c, out_c, k, 1, k//2),
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class GatedUNet(nn.Module):
    """Generator: input = [image(3), mask(1), optional edges(1)] => restored RGB (3)."""
    def __init__(self, in_ch=4, base=64, edge_guidance=True):
        super().__init__()
        self.edge_guidance = edge_guidance
        inc = in_ch + (1 if edge_guidance else 0)
        self.enc1 = nn.Sequential(GatedConv2d(inc, base), GatedConv2d(base, base))
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.bott = nn.Sequential(GatedConv2d(base*8, base*8, k=3), GatedConv2d(base*8, base*8, k=3))
        self.up3 = Up(base*8, base*4)
        self.up2 = Up(base*4, base*2)
        self.up1 = Up(base*2, base)
        self.outc = nn.Conv2d(base, 3, kernel_size=3, padding=1)

        # Local edge-tuned enhancer for boundary band
        self.enh = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1)
        )

    def forward(self, image, mask, edges=None, band_px=12):
        # mask: 1 where HOLE to fill
        x = torch.cat([image, mask], dim=1)
        if self.edge_guidance and edges is not None:
            x = torch.cat([x, edges], dim=1)

        e1 = self.enc1(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        b  = self.bott(e4)
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        out = torch.tanh(self.outc(d1))  # [-1,1] preferable; we use [0,1], so clamp later
        out = (out + 1) / 2.0

        # Combine only hole pixels from generator, keep context
        comp = image * (1 - mask) + out * mask

        # Enhance boundary band
        if band_px > 0:
            kernel = torch.ones(1, 1, band_px*2+1, band_px*2+1, device=mask.device)
            dil = (F.conv2d(mask, kernel, padding=band_px) > 0).float()
            band = torch.clamp(dil - mask, 0, 1)
            comp = comp + self.enh(comp) * band
            comp = torch.clamp(comp, 0, 1)
        return comp

# ---------- PatchGAN on boundary tiles ----------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        def C(c_in, c_out, k=4, s=2, p=1, norm=True):
            layers = [nn.Conv2d(c_in, c_out, k, s, p), nn.LeakyReLU(0.2, inplace=True)]
            if norm:
                layers.insert(1, nn.InstanceNorm2d(c_out, affine=True))
            return nn.Sequential(*layers)
        self.net = nn.Sequential(
            C(in_ch, base, norm=False),
            C(base, base*2),
            C(base*2, base*4),
            nn.Conv2d(base*4, 1, kernel_size=4, stride=1, padding=1)
        )
    def forward(self, x):  # x in [B,3,H,W]
        return self.net(x)

def sample_boundary_tiles(img, mask, tile=96, k=4):
    """Return k tiles around boundary band."""
    b, _, h, w = mask.shape
    tiles = []
    for bi in range(b):
        m = mask[bi,0].cpu().numpy()
        ys, xs = (m > 0).nonzero()
        if len(xs) == 0:
            # fallback random tiles
            for _ in range(k):
                y = torch.randint(0, max(1,h-tile), (1,)).item()
                x = torch.randint(0, max(1,w-tile), (1,)).item()
                tiles.append(img[bi:bi+1, :, y:y+tile, x:x+tile])
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        for _ in range(k):
            rx = max(0, min(w - tile, int(cx + torch.randint(-64, 65, (1,)).item())))
            ry = max(0, min(h - tile, int(cy + torch.randint(-64, 65, (1,)).item())))
            tiles.append(img[bi:bi+1, :, ry:ry+tile, rx:rx+tile])
    return torch.cat(tiles, dim=0)

# ---------- Loss bundle ----------
class LossBundle:
    def __init__(self, lambda_l1=1.0, lambda_perc=0.5, lambda_style=0.25, lambda_adv=0.8, lambda_tv=0.02):
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_style = lambda_style
        self.lambda_adv = lambda_adv
        self.lambda_tv = lambda_tv
        self.lpips = try_lpips()  # may be None

    def masked_l1(self, pred, target, mask, band_px=12):
        loss = ( (pred - target).abs() * mask ).mean()
        if band_px > 0:
            kernel = torch.ones(1,1,band_px*2+1, band_px*2+1, device=mask.device)
            dil = (F.conv2d(mask, kernel, padding=band_px) > 0).float()
            band = torch.clamp(dil - mask, 0, 1)
            loss = loss + ( (pred-target).abs() * band ).mean()
        return loss

    def perceptual(self, pred, target):
        if self.lpips is None:
            return torch.tensor(0.0, device=pred.device)
        return self.lpips(pred*2-1, target*2-1).mean()

    def style(self, pred, target, band_mask):
        # compute Gram on band region
        def masked_feat(x, band):
            return x * band
        # downsample features a bit
        p = F.avg_pool2d(pred, 2)
        t = F.avg_pool2d(target, 2)
        b = F.avg_pool2d(band_mask, 2)
        g1 = gram_matrix(masked_feat(p, b))
        g2 = gram_matrix(masked_feat(t, b))
        return (g1 - g2).abs().mean()

    def adv_hinge_d(self, real, fake):
        return (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())

    def adv_hinge_g(self, fake):
        return -fake.mean()

    def tv(self, pred, mask):
        return total_variation(pred * mask)

# ---------- Simple metrics ----------
def l1_in_hole(pred, target, mask):
    denom = torch.clamp(mask.sum(), min=1.0)
    return ((pred - target).abs() * mask).sum() / denom
