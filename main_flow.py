import time, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from tools import load_and_resize_image, load_and_scale_image, enhance_contrast
from nets  import TemplateMatchingNet, UNetLite   # 确保 nets.py 中已包含 UNetLite

# ---------- 配置 ----------
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoints
CKPT_TMNET  = r"train_out\exp7\model_best.pth"          # TemplateMatchingNet
CKPT_UNET   = r"train_out\unet_exp3_b\model_best.pth"     # 你的 UNetLite

# # # # 测试图片
# SRC_IMG  = r"data_gen\test_data\item26\composite.png"
# SEED_IMG = r"data_gen\test_data\item26\seed.png"


SRC_IMG  = r"E:\python_prj\2025_5_26\image\demo2\composite.png"
SEED_IMG = r"E:\python_prj\2025_5_26\image\demo2\seed.png"

canvas  = (640, 640)   # 统一分辨率 (H,W)
pad     = 64           # 反射填充



# ---------- 1. 读图 + 缩放 ----------
src, sc_x, sc_y   = load_and_resize_image(SRC_IMG, *canvas)  # [1,3,H,W]
tem, *_           = load_and_scale_image(SEED_IMG, sc_x, sc_y)
src, tem = src.to(device), tem.to(device)

# ---------- 2. 生成 Base Density ----------
src_pad = F.pad(src, (pad, pad, pad, pad), mode='reflect')
tm_net  = TemplateMatchingNet().to(device)
tm_net.load_state_dict(torch.load(CKPT_TMNET, map_location=device))
tm_net.eval()

t0 = time.time()
with torch.no_grad():
    base = tm_net(src_pad, tem)          # [1,1,h,w]
    if base.shape[-2:] != src_pad.shape[-2:]:
        base = F.interpolate(base, size=src_pad.shape[-2:], mode='bilinear', align_corners=False)
base = base[..., pad:-pad, pad:-pad]     # 去掉 padding → [1,1,H,W]
tm_time = time.time() - t0

# ---------- 3. 用 UNet 精化 ----------
unet = UNetLite(in_ch=4, base_ch=32).to(device)
unet.load_state_dict(torch.load(CKPT_UNET, map_location=device))
unet.eval()

# 拼 4 通道
inp = torch.cat([src, base], dim=1)      # [1,4,H,W]

t0 = time.time()
with torch.no_grad():
    refined = unet(inp)                  # [1,1,H,W]，已 sigmoid
unet_time = time.time() - t0

print(f"TMNet  time: {tm_time:.3f}s | UNet time: {unet_time:.3f}s")

# ---------- 4. 转 numpy 便于可视化 ----------
src_np     = src.squeeze().permute(1,2,0).cpu().numpy().clip(0,1)
base_np    = enhance_contrast(base.squeeze().cpu().numpy())
refine_np  = enhance_contrast(refined.squeeze().cpu().numpy())

# ---------- 5. 可视化 ----------
fz = 18
plt.figure(figsize=(14,4))

plt.subplot(1,4,1)
plt.imshow(src_np); plt.title("RGB",fontsize = fz); plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(base_np+1e-6, cmap='viridis')
plt.title("Base",fontsize = fz); plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1,4,3)
plt.imshow(refine_np+1e-6, cmap='viridis')
plt.title("Refined",fontsize = fz); plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

# ===============================
from scipy.ndimage import label
threshold = 0.5
mask = refine_np > threshold
_, num_regions = label(mask)
print("Number of connected regions above threshold:", num_regions)
from scipy.ndimage import label
# 在图像右上角添加文字
h, w = refine_np.shape
plt.text(w * 0.95, h * 0.05, f"nums: {num_regions},real:120",
         color='white', fontsize=14, ha='right', va='top',
         bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
# ===============================

plt.subplot(1,4,4)
plt.imshow(src_np)
plt.imshow(refine_np+1e-6, cmap='jet', alpha=0.55)
plt.title("Overlay (Refined)",fontsize = fz); plt.axis('off')

plt.tight_layout(); plt.show()
