# mydatasets.py
import random, numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T


class SeedCountingDataset(Dataset):
    """
    读取 train_data/ 或 val_data/ 文件夹，返回 (composite, seed, density) 三个张量
    ├─ itemX/
       ├─ composite.png
       ├─ seed.png
       └─ density.npy
    """
    def __init__(self, root_dir, canvas_size=(256, 256), seed_resize=None):
        """
        root_dir     : train_data 或 val_data 目录
        canvas_size  : (H,W) -> 是否把 composite / density resize 到统一分辨率
                       None  表示保持原尺寸
        seed_resize  : int 或 None，若设定则把 seed 图等比缩放到该高度
        """
        self.root_dir = Path(root_dir)
        self.items    = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        if not self.items:
            raise RuntimeError(f"No item folders found in {root_dir}")
        self.canvas_size = canvas_size
        self.seed_resize = seed_resize

        # 基础 transform
        self.to_tensor = T.ToTensor()  # 0-1 float, shape [C,H,W]

        self.resize_comp = (
            T.Resize(canvas_size, interpolation=Image.BILINEAR)
            if canvas_size else None
        )

    def __len__(self):
        return len(self.items)

    def _load_density(self, npy_path):
        den = torch.from_numpy(np.load(npy_path)).float()  # [H,W]
        den = den.unsqueeze(0)  # [1,H,W]
        if self.canvas_size:
            den = F.interpolate(
                den.unsqueeze(0), size=self.canvas_size,
                mode="bilinear", align_corners=False
            ).squeeze(0)        # [1,H,W]
        return den

    def _load_seed(self, img_path):
        img = Image.open(img_path).convert("RGB")
        if self.seed_resize:
            new_h = self.seed_resize
            new_w = int(img.width * new_h / img.height)
            img   = img.resize((new_w, new_h), Image.BILINEAR)
        return self.to_tensor(img)

    def __getitem__(self, idx):
        item_dir = self.items[idx]
        comp_path = item_dir / "composite.png"
        seed_path = item_dir / "seed.png"
        den_path  = item_dir / "density.npy"

        # composite
        comp_img = Image.open(comp_path).convert("RGB")
        if self.resize_comp:
            comp_img = self.resize_comp(comp_img)
        comp = self.to_tensor(comp_img)            # [3,H,W]

        # seed
        seed = self._load_seed(seed_path)          # [3,h,w]

        # density
        density = self._load_density(den_path)     # [1,H,W]

        return comp, seed, density





class DensityRefineDataset(Dataset):
    """
    用于 U-Net 精化：返回 (rgb+base_density, gt_density)

    目录结构:
      data_root/
        ├─ itemX/
        │    ├─ composite.png        # RGB
        │    ├─ density_base.png     # 基础密度图（单通道 PNG，8/16-bit 都可）
        │    └─ density.npy          # GT 高斯峰密度 (H, W)
    """
    def __init__(
        self,
        root_dir,
        canvas_size=None,              # (H,W) or None  -> 是否统一尺寸
        base_name="density_base.png",  # 基础密度文件名
        gt_name="density.npy",         # GT 文件名
        normalize_base=True,           # 是否把 base density 归一化到 0-1
    ):
        self.root_dir   = Path(root_dir)
        self.items      = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        if not self.items:
            raise RuntimeError(f"No item folders found in {root_dir}")

        self.canvas_size   = canvas_size
        self.base_name     = base_name
        self.gt_name       = gt_name
        self.normalize_base = normalize_base

        # torchvision 常用变换
        self.to_tensor = T.ToTensor()                     # → [C,H,W], float32 0-1
        self.resize_img = (
            T.Resize(canvas_size, interpolation=Image.BILINEAR)
            if canvas_size else None
        )

    # ---------- 帮助函数 ----------
    def _load_rgb(self, path):
        img = Image.open(path).convert("RGB")
        if self.resize_img:
            img = self.resize_img(img)
        return self.to_tensor(img)                        # [3,H,W]

    def _load_base_density(self, path, tgt_hw):
        """读取单通道 PNG/TIFF，返回 [1,H,W]，并插值到 tgt_hw"""
        img = Image.open(path)
        arr = np.array(img)                               # uint8 / uint16
        # 转 float32
        if arr.dtype == np.uint16:
            arr = arr.astype(np.float32) / 65535.0
        else:  # uint8 / others
            arr = arr.astype(np.float32) / 255.0

        if self.normalize_base:                           # 再整体归一化到 0-1
            arr -= arr.min()
            if arr.max() > 0:
                arr /= arr.max()

        tens = torch.from_numpy(arr).unsqueeze(0)         # [1,h,w]
        # 若尺寸与 RGB 不同，则插值
        if tens.shape[-2:] != tgt_hw:
            tens = F.interpolate(
                tens.unsqueeze(0), size=tgt_hw,
                mode="bilinear", align_corners=False
            ).squeeze(0)
        return tens                                       # [1,H,W]

    def _load_gt_density(self, path, tgt_hw):
        gt = torch.from_numpy(np.load(path)).float()      # [H,W] 或 [h,w]
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)                          # [1,H,W]
        if self.canvas_size and gt.shape[-2:] != tgt_hw:
            gt = F.interpolate(
                gt.unsqueeze(0), size=tgt_hw,
                mode="bilinear", align_corners=False
            ).squeeze(0)
        return gt                                         # [1,H,W]

    # ---------- Dataset 接口 ----------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_dir = self.items[idx]
        rgb_path   = item_dir / "composite.png"
        base_path  = item_dir / self.base_name
        gt_path    = item_dir / self.gt_name

        # 1. RGB
        rgb = self._load_rgb(rgb_path)                    # [3,H,W]
        H, W = rgb.shape[-2:]

        # 2. Base Density
        base = self._load_base_density(base_path, (H, W)) # [1,H,W]

        # 3. 组合输入 (4 通道)
        inp = torch.cat([rgb, base], dim=0)               # [4,H,W]

        # 4. GT Density
        gt = self._load_gt_density(gt_path, (H, W))       # [1,H,W]

        return inp, gt




# ------------------- Demo / 可视化 -------------------
if __name__ == "__main__":
    ds = SeedCountingDataset(r"data_gen\train_data", canvas_size=(256,256), seed_resize=32)
    print(f"Dataset length: {len(ds)}")

    # 随机取一个样本
    comp, seed, den = ds[random.randrange(len(ds))]
    print("composite:", comp.shape, "seed:", seed.shape, "density:", den.shape)

    # Tensor -> numpy for plotting
    comp_np = comp.permute(1,2,0).numpy()         # HWC
    seed_np = seed.permute(1,2,0).numpy()
    den_np  = den.squeeze(0).numpy()

    # 绘图
    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(comp_np)
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(seed_np)
    axs[1].set_title("Seed Example")
    axs[1].axis("off")

    axs[2].imshow(den_np, cmap="hot")
    axs[2].set_title("Density Map")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
