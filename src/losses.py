# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import lpips  # optional
except Exception:
    lpips = None

def _to_float(mask):
    m = mask
    if m.dtype != torch.float32:
        m = m.float()
    # assume mask is 1.0 in HOLE (region to fill), 0.0 elsewhere
    return m.clamp(0, 1)

def l1_in_hole(pred, target, hole_mask, eps: float = 1e-6):
    """L1 over the HOLE region (mask==1)."""
    m = _to_float(hole_mask)
    num = (m.sum(dim=(1,2,3)) + eps).view(-1, 1, 1, 1)
    return ((pred - target).abs() * m).sum(dim=(1,2,3), keepdim=True) / num

def l1_on_band(pred, target, hole_mask, band_px: int = 12, eps: float = 1e-6):
    """L1 on a narrow boundary band around the hole (to keep seams sharp)."""
    if band_px <= 0:
        return torch.zeros((pred.size(0),1,1,1), device=pred.device)
    m = _to_float(hole_mask)

    # Build a dilated mask to approximate the band
    k = torch.ones(1, 1, 2*band_px+1, 2*band_px+1, device=pred.device)
    if m.ndim == 4 and m.size(1) != 1:
        # convert CHW mask to single channel if needed
        m = m.mean(dim=1, keepdim=True)
    dil = torch.clamp(F.conv2d(m, k, padding=band_px), 0, 1)
    band = torch.clamp(dil - m, 0, 1)  # ring around the hole
    denom = (band.sum(dim=(1,2,3)) + eps).view(-1,1,1,1)
    return ((pred - target).abs() * band).sum(dim=(1,2,3), keepdim=True) / denom

def total_variation_in_hole(pred, hole_mask, eps: float = 1e-6):
    """TV only inside the hole (encourages smoothness just in synthesized area)."""
    m = _to_float(hole_mask)
    dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs()
    dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs()
    mx = (m[:, :, :, 1:] + m[:, :, :, :-1]).clamp(0,1)
    my = (m[:, :, 1:, :] + m[:, :, :-1, :]).clamp(0,1)
    tv = (dx * mx).mean() + (dy * my).mean()
    return tv

class LossBundle(nn.Module):
    """
    Combines masked L1 (in-hole), boundary L1 band, optional perceptual (LPIPS),
    optional style/Gram (cheap proxy), and TV in hole.
    Discriminator adversarial loss can be added externally if you’re using PatchGAN.
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        band_weight: float = 0.5,
        lpips_weight: float = 0.2,
        tv_weight: float = 0.01,
        band_px: int = 12,
        use_lpips: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.l1_w = l1_weight
        self.band_w = band_weight
        self.lpips_w = lpips_weight
        self.tv_w = tv_weight
        self.band_px = band_px

        if use_lpips and lpips is not None:
            self.lpips = lpips.LPIPS(net="vgg").to(device).eval()
            for p in self.lpips.parameters():
                p.requires_grad = False
        else:
            self.lpips = None

    def perceptual(self, pred, target):
        if self.lpips is None or self.lpips_w <= 0:
            return pred.new_zeros(())
        # LPIPS expects [-1,1] range
        p = pred * 2 - 1
        t = target * 2 - 1
        return self.lpips(p, t).mean()

    def forward(self, pred, target, hole_mask):
        """
        pred/target: BxCxHxW in [0,1]
        hole_mask:  Bx1xHxW with 1=hole (area to fill), 0=context
        """
        l1_hole = l1_in_hole(pred, target, hole_mask).mean()
        l1_band = l1_on_band(pred, target, hole_mask, band_px=self.band_px).mean()
        tv = total_variation_in_hole(pred, hole_mask)
        pcp = self.perceptual(pred, target)

        total = (
            self.l1_w * l1_hole
            + self.band_w * l1_band
            + self.tv_w * tv
            + self.lpips_w * pcp
        )

        return {
            "total": total,
            "l1_hole": l1_hole.detach(),
            "l1_band": l1_band.detach(),
            "tv": tv.detach(),
            "lpips": pcp.detach() if torch.is_tensor(pcp) else torch.tensor(0.0, device=pred.device),
        }
