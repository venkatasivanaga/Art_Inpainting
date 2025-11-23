# api.py
# FastAPI server: POST /api/inpaint accepts (image, optional mask) and returns PNG bytes.
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
import io
import torch
from PIL import Image
import numpy as np
from model import GatedUNet
import uvicorn
# in src/api.py
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ArtRepair API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load once
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_G = None

def load_model(ckpt_path: str, edge_guidance=True):
    global _G
    G = GatedUNet(in_ch=4, base=64, edge_guidance=edge_guidance).to(_device).eval()
    state = torch.load(ckpt_path, map_location=_device)
    G.load_state_dict(state["g"])
    _G = G

def pil_to_tensor(img: Image.Image, size=512):
    img = img.convert("RGB")
    img = img.copy()
    img.thumbnail((size, size))  # keep aspect
    w, h = img.size
    canvas = Image.new("RGB", (size, size), (0,0,0))
    canvas.paste(img, ((size-w)//2, (size-h)//2))
    arr = np.array(canvas).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return t

def mask_to_tensor(mask: Image.Image, size=512):
    mask = mask.convert("L")
    mask = mask.copy()
    mask.thumbnail((size, size))
    w, h = mask.size
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(mask, ((size-w)//2, (size-h)//2))
    arr = (np.array(canvas) > 127).astype(np.float32)
    t = torch.from_numpy(arr)[None,None,:,:]
    return t

def tensor_to_png_bytes(t):
    t = t.clamp(0,1)[0].permute(1,2,0).cpu().numpy()
    img = Image.fromarray((t*255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

@app.on_event("startup")
def _startup():
    # You can change this path after training:
    try:
        load_model("runs/gan_512/best.ckpt", edge_guidance=True)
        print("[api] Loaded model runs/gan_512/best.ckpt")
    except Exception as e:
        print("[api] WARNING: could not load checkpoint yet:", e)

@app.post("/api/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    model: str = Form("gan"),
    strength: float = Form(0.85),
    seed: int = Form(1234),
    edgeAware: bool = Form(True)
):
    torch.manual_seed(seed)
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_t = pil_to_tensor(img, size=512).to(_device)

    if mask is not None:
        m_bytes = await mask.read()
        m_img = Image.open(io.BytesIO(m_bytes)).convert("L")
        mask_t = mask_to_tensor(m_img, size=512).to(_device)
    else:
        # If no mask provided, make a conservative auto-mask (low-contrast edges)
        g = np.array(img.convert("L"))
        can = cv2.Canny(g, 40, 120)
        mask_t = torch.from_numpy((can > 0).astype(np.float32))[None,None].to(_device)

    # simple edge map
    gray = img_t.mean(1, keepdim=True)
    sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=img_t.dtype, device=_device)
    sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=img_t.dtype, device=_device)
    gx = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
    edges = (gx.abs()+gy.abs()).clamp(0,1)

    if _G is None:
        # Return original as a placeholder if model not loaded yet
        print("[api] model not loaded; echoing input")
        out = img_t
    else:
        with torch.no_grad():
            out = _G(img_t, mask_t, edges=edges if edgeAware else None, band_px=12)

    return Response(content=tensor_to_png_bytes(out), media_type="image/png")

if __name__ == "__main__":
    # For local testing: python -m src.api
    uvicorn.run(app, host="0.0.0.0", port=8000)
