# train_UNET.py
import os, time, math, json
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import pandas as pd
from my_loss import mixed_loss,GaussianDensityLoss,generalized_density_loss

from mydatasets import DensityRefineDataset
from nets import UNetLite   # 你的 UNet 实现

# ---------- 配置 ----------
CFG = {
    "train_dir":  r"E:\python_prj\2025_5_26\solid_bg_data",
    "val_dir":    r"E:\python_prj\2025_5_26\solid_bg_data",
    "save_dir":   "train_out/unet_exp3_b",
    "canvas":     (640, 640),       # 统一分辨率; None → 原图
    "batch":      1,
    "epochs":     40,
    "lr":         1e-3,
    "base_ch":    32,               # UNetLite 通道基数
    "amp":        True,             # 混合精度
    "num_workers":4,
    "bg_penalty": False,            # 如果需要背景惩罚可设 True
}

os.makedirs(CFG["save_dir"], exist_ok=True)

# ---------- 数据 ----------
train_ds = DensityRefineDataset(
    CFG["train_dir"], canvas_size=CFG["canvas"],
    base_name="density_pred.png",  # 如有变化自行改
)
val_ds   = DensityRefineDataset(
    CFG["val_dir"], canvas_size=CFG["canvas"],
    base_name="density_pred.png",
)
train_loader = DataLoader(train_ds, CFG["batch"], True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   CFG["batch"], False,
                          num_workers=0, pin_memory=True)

# ---------- 模型 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = UNetLite(in_ch=4, base_ch=CFG["base_ch"]).to(device)
# criterion = nn.MSELoss()
criterion = generalized_density_loss
optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])
sched     = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CFG["epochs"])

# scaler = GradScaler(enabled=CFG["amp"])

# 其余超参、模型、数据加载等保持不变
scaler = GradScaler()

def train_epoch(epoch):
    model.train()
    running = 0.0
    pbar = tqdm(train_loader, desc=f"Train {epoch}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with autocast(enabled=CFG["amp"]):
            pred  = model(x)
            loss  = criterion(pred, y)
            # 可选背景惩罚
            if CFG["bg_penalty"]:
                bg_mask = (y < 1e-3).float()
                loss += torch.mean(bg_mask * pred**2)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * x.size(0)
        pbar.set_postfix(loss=loss.item())

    return running / len(train_ds)


@torch.no_grad()
def eval_epoch(epoch):
    model.eval()
    total = 0.0
    for x, y in tqdm(val_loader, desc=f"Val {epoch}"):
        x, y = x.to(device), y.to(device)
        with autocast(enabled=CFG["amp"]):
            pred  = model(x)
            loss  = criterion(pred, y)
        total += loss.item() * x.size(0)

    return total / len(val_ds)


best_loss = math.inf
history   = []

for ep in range(1, CFG["epochs"] + 1):
    t0 = time.time()
    train_loss = train_epoch(ep)
    val_loss   = eval_epoch(ep)
    sched.step()

    history.append({
        "epoch": ep,
        "train": train_loss,
        "val":   val_loss,
        "lr":    optimizer.param_groups[0]["lr"]
    })

    print(f"Epoch {ep:03d}: "
          f"train {train_loss:.4f}  val {val_loss:.4f}  "
          f"time {(time.time()-t0):.1f}s")

    # ---- 保存模型 ----
    ckpt_dir = Path(CFG["save_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model_last.pth")

    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), ckpt_dir / "model_best.pth")
        print(f"  ↳ New best! ({best_loss:.4f})")

# ---- 保存日志为 Excel ----
df = pd.DataFrame(history)                                       # ← 关键一步
excel_path = Path(CFG["save_dir"]) / "loss_history.xlsx"
df.to_excel(excel_path, index=False)                             # ← 关键一步
print(f"Training finished, log saved to {excel_path}")

# def train_epoch(epoch):
#     model.train()
#     running = 0.0
#     pbar = tqdm(train_loader, desc=f"Train {epoch}")
#     for x, y in pbar:
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         with autocast(enabled=CFG["amp"]):
#             pred = model(x)
#             loss = criterion(pred, y)
#             # 可选背景惩罚
#             if CFG["bg_penalty"]:
#                 bg_mask = (y < 1e-3).float()
#                 loss += torch.mean(bg_mask * pred ** 2)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#
#         running += loss.item() * x.size(0)
#         pbar.set_postfix(loss=loss.item())
#     return running / len(train_ds)
#
# @torch.no_grad()
# def eval_epoch(epoch):
#     model.eval()
#     total = 0.0
#     for x, y in tqdm(val_loader, desc=f"Val {epoch}"):
#         x, y = x.to(device), y.to(device)
#         with autocast(enabled=CFG["amp"]):
#             pred = model(x)
#             loss = criterion(pred, y)
#         total += loss.item() * x.size(0)
#     return total / len(val_ds)
#
# best_loss = math.inf
# history = []
#
# for ep in range(1, CFG["epochs"] + 1):
#     t0 = time.time()
#     train_loss = train_epoch(ep)
#     val_loss   = eval_epoch(ep)
#     sched.step()
#
#     history.append({"epoch": ep,
#                     "train": train_loss,
#                     "val": val_loss,
#                     "lr":   optimizer.param_groups[0]["lr"]})
#     # ---- 记录 ----
#     print(f"Epoch {ep:03d}: "
#           f"train {train_loss:.4f}  val {val_loss:.4f}  "
#           f"time {(time.time()-t0):.1f}s")
#
#     # ---- 保存 ----
#     ckpt_path = Path(CFG["save_dir"]) / "model_last.pth"
#     torch.save(model.state_dict(), ckpt_path)
#     if val_loss < best_loss:
#         best_loss = val_loss
#         torch.save(model.state_dict(),
#                    Path(CFG["save_dir"]) / "model_best.pth")
#         print(f"  ↳ New best! ({best_loss:.4f})")
#
# # 保存日志
# with open(Path(CFG["save_dir"]) / "log.json", "w") as f:
#     json.dump(history, f, indent=2)
# print("Training finished.")
