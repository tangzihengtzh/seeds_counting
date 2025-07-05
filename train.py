import torch, os
from torch import nn, optim
from torch.utils.data import DataLoader
from mydatasets import SeedCountingDataset
from nets import TemplateMatchingNet        # 把上面网络单独放 model.py 更整洁
import os, torch, matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from my_loss import mixed_loss,GaussianDensityLoss,generalized_density_loss
import pandas as pd  # 确保已导入 pandas






# ---------- 可调参数 ----------
# data_root   = r"data_gen\train_data"
data_root   = r"E:\python_prj\2025_5_26\solid_bg_data"
val_root    = r"data_gen\val_data"
epochs      = 10
batch_size  = 1
lr          = 1e-4
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备：",device)

# ---------- 数据 ----------
train_set = SeedCountingDataset(data_root,  canvas_size=(256,256), seed_resize=32)
print("训练集数量：",train_set.__len__())
val_set   = SeedCountingDataset(val_root,   canvas_size=(256,256), seed_resize=32)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=1,          shuffle=False)

# ---------- 模型 / 损失 / 优化 ----------
model = TemplateMatchingNet().to(device)
# criterion = nn.MSELoss()
# criterion = mixed_loss
criterion = generalized_density_loss
# criterion = GaussianDensityLoss(alpha=1.0, beta=1.0, gamma=0.5)
optimizer = optim.Adam(model.parameters(), lr=lr)

# ---------------- exp 目录自动递增 ----------------
out_root = Path("train_out")
out_root.mkdir(exist_ok=True)
exp_idx = 1
while (out_root / f"exp{exp_idx}").exists():
    exp_idx += 1
exp_dir = out_root / f"exp{exp_idx}"
exp_dir.mkdir()
print(f"➤ Results will be saved to {exp_dir}")

# ---------------- 训练循环 ----------------
train_mse_hist, val_mse_hist = [], []
best_val = float("inf")

for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    model.train()
    running_loss = 0.0

    # ---------- Train ----------
    for comp, seed, den in tqdm(train_loader, desc="Train", unit="batch", leave=False):
        comp, seed, den = comp.to(device), seed.to(device), den.to(device)
        pred = model(comp, seed)
        if pred.shape[-2:] != den.shape[-2:]:
            pred = torch.nn.functional.interpolate(pred, size=den.shape[-2:],
                                                   mode="bilinear", align_corners=False)
        loss = criterion(pred, den)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * comp.size(0)

    train_mse = running_loss / len(train_set)
    train_mse_hist.append(train_mse)
    print(f"  train MSE: {train_mse:.4f}")

    # ---------- Val ----------
    model.eval()
    val_err = 0.0
    with torch.no_grad():
        for comp, seed, den in tqdm(val_loader, desc=" Val ", unit="batch", leave=False):
            comp, seed, den = comp.to(device), seed.to(device), den.to(device)
            pred = model(comp, seed)
            if pred.shape[-2:] != den.shape[-2:]:
                pred = torch.nn.functional.interpolate(pred, size=den.shape[-2:],
                                                       mode="bilinear", align_corners=False)
            val_err += criterion(pred, den).item()

    val_mse = val_err / len(val_loader)
    val_mse_hist.append(val_mse)
    print(f"  val   MSE: {val_mse:.4f}")

    # ---------- Checkpoint ----------
    if epoch % 5 == 0:
        ckpt_epoch = exp_dir / f"model_epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt_epoch)
        print(f"    📌 Checkpoint saved: {ckpt_epoch}")

    if val_mse < best_val:
        best_val = val_mse
        torch.save(model.state_dict(), exp_dir / "model_best.pth")

# ---------------- 训练结束 ----------------
ckpt_final = exp_dir / "model_last.pth"
torch.save(model.state_dict(), ckpt_final)
print(f"✔️  Final model saved: {ckpt_final}")

# # ---------- 绘制 Loss 曲线 ----------
# plt.figure(figsize=(6,4))
# plt.plot(train_mse_hist, label="Train MSE")
# plt.plot(val_mse_hist,   label="Val MSE")
# plt.xlabel("Epoch")
# plt.ylabel("MSE")
# plt.title("Training Curve")
# plt.legend()
# plt.grid(True)
# curve_path = exp_dir / "loss_curve.png"
# plt.savefig(curve_path, dpi=150)
# plt.close()
# print(f"📈 Loss curve saved to {curve_path}")



# ---------- 绘制 Loss 曲线 ----------
plt.figure(figsize=(6,4))
plt.plot(train_mse_hist, label="Train MSE")
plt.plot(val_mse_hist,   label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training Curve")
plt.legend()
plt.grid(True)
curve_path = exp_dir / "loss_curve.png"
plt.savefig(curve_path, dpi=150)
plt.close()
print(f"📈 Loss curve saved to {curve_path}")

# ---------- 保存 Loss 数据到 Excel ----------
loss_data = pd.DataFrame({
    "Epoch": list(range(1, len(train_mse_hist)+1)),
    "Train MSE": train_mse_hist,
    "Val MSE": val_mse_hist
})
excel_path = exp_dir / "loss_curve.xlsx"
loss_data.to_excel(excel_path, index=False)
print(f"📊 Loss data saved to {excel_path}")



# # ---------------- 训练循环 ----------------
# train_mse_hist, val_mse_hist = [], []
#
# for epoch in range(1, epochs + 1):
#     print(f"\nEpoch {epoch}/{epochs}")
#     model.train()
#     running_loss = 0.0
#
#     for comp, seed, den in tqdm(train_loader, desc="Train", unit="batch", leave=False):
#         comp, seed, den = comp.to(device), seed.to(device), den.to(device)
#
#         pred = model(comp, seed)
#         if pred.shape[-2:] != den.shape[-2:]:
#             pred = torch.nn.functional.interpolate(pred, size=den.shape[-2:], mode='bilinear', align_corners=False)
#
#         # loss,_ = criterion(pred, den)
#         loss = criterion(pred, den)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item() * comp.size(0)
#
#     train_mse = running_loss / len(train_set)
#     train_mse_hist.append(train_mse)
#     print(f"  train MSE: {train_mse:.4f}")
#
#     # ---------- 验证 ----------
#     model.eval()
#     val_err = 0.0
#     with torch.no_grad():
#         for comp, seed, den in tqdm(val_loader, desc=" Val ", unit="batch", leave=False):
#             comp, seed, den = comp.to(device), seed.to(device), den.to(device)
#             pred = model(comp, seed)
#             if pred.shape[-2:] != den.shape[-2:]:
#                 pred = torch.nn.functional.interpolate(pred, size=den.shape[-2:], mode='bilinear', align_corners=False)
#             val_err += criterion(pred, den).item()
#             # val_err += criterion(pred, den)[0].item()
#
#     val_mse = val_err / len(val_loader)
#     val_mse_hist.append(val_mse)
#     print(f"  val   MSE: {val_mse:.4f}")
#
# # ---------------- 保存权重 ----------------
# ckpt_path = exp_dir / "model.pth"
# torch.save(model.state_dict(), ckpt_path)
# print(f"✔️  Model saved: {ckpt_path}")
#
# # ---------------- 绘制 & 保存 loss 曲线 ----------------
# plt.figure(figsize=(6,4))
# plt.plot(train_mse_hist, label="Train MSE")
# plt.plot(val_mse_hist,   label="Val MSE")
# plt.xlabel("Epoch")
# plt.ylabel("MSE")
# plt.title("Training Curve")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# curve_path = exp_dir / "loss_curve.png"
# plt.savefig(curve_path, dpi=300)
# plt.close()
# print(f"✔️  Loss curve saved: {curve_path}")
#
# # （可选）记录到 txt
# with open(exp_dir / "train_log.txt", "w") as f:
#     for ep, (tr, va) in enumerate(zip(train_mse_hist, val_mse_hist), 1):
#         f.write(f"Epoch {ep}\ttrain={tr:.6f}\tval={va:.6f}\n")
# print("全部完成！")










# # ---------- 训练循环 ----------
# for epoch in range(1, epochs+1):
#     print("epoch：",epochs)
#     model.train()
#     running_loss = 0.0
#     for comp, seed, den in tqdm(train_loader, desc="Train", unit="batch", leave=False):
#         # print("获取数据条目：",comp.shape, seed.shape, den.shape)
#         # exit(1)
#         comp, seed, den = comp.to(device), seed.to(device), den.to(device)
#
#         # pred = model(comp, seed)          # [B,1,H,W]
#         # print("当前计算结果：",pred.shape)
#         # loss = criterion(pred, den)
#
#         pred = model(comp, seed)  # [B,1,*,*]
#         if pred.shape[-2:] != den.shape[-2:]:
#             pred = torch.nn.functional.interpolate(
#                 pred, size=den.shape[-2:], mode='bilinear', align_corners=False)
#         loss = criterion(pred, den)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * comp.size(0)
#
#     avg_loss = running_loss / len(train_set)
#     print(f"Epoch {epoch}/{epochs}  train MSE: {avg_loss:.4f}")
#
#     # ---- 简单验证 ----
#     model.eval()
#     with torch.no_grad():
#         val_err = 0.0
#         for comp, seed, den in val_loader:
#             comp, seed, den = comp.to(device), seed.to(device), den.to(device)
#             pred = model(comp, seed)  # [B,1,*,*]
#             if pred.shape[-2:] != den.shape[-2:]:
#                 pred = torch.nn.functional.interpolate(
#                     pred, size=den.shape[-2:], mode='bilinear', align_corners=False)
#             val_err += criterion(pred, den).item()
#     print(f"               val MSE : {val_err/len(val_loader):.4f}")
#
# # ---------- 保存 ----------
# torch.save(model.state_dict(), "template_match_net.pth")
