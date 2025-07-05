# nets.py

from tools import load_and_resize_image
from tools import load_and_scale_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF



# class RotatePool(nn.Module):
#     """
#     对输入特征图 x 做 0/90/180/270° 旋转后逐像素取最大（或平均），
#     输出与 x 同形状，实现旋转不变。
#     mode: 'max' 或 'mean'
#     """
#     def __init__(self, mode='max'):
#         super().__init__()
#         assert mode in ('max', 'mean')
#         self.mode = mode
#
#     def forward(self, x):
#         rotated = [x,
#                    torch.rot90(x, 1, dims=(2, 3)),
#                    torch.rot90(x, 2, dims=(2, 3)),
#                    torch.rot90(x, 3, dims=(2, 3))]
#         stacked = torch.stack(rotated, dim=0)   # [4,B,C,H,W]
#         if self.mode == 'max':
#             out = stacked.max(dim=0)[0]
#         else:
#             out = stacked.mean(dim=0)
#         return out

class RotatePool(nn.Module):
    def __init__(self, mode='max'):
        super().__init__()
        assert mode in ('max', 'mean')
        self.mode = mode

    def forward(self, x):
        # 原始大小
        H, W = x.shape[-2:]
        rotated = [
            x,
            torch.rot90(x, 1, dims=(2, 3)),
            torch.rot90(x, 2, dims=(2, 3)),
            torch.rot90(x, 3, dims=(2, 3))
        ]
        # 将尺寸≠(H,W) 的旋转图插值回原大小
        rotated = [
            r if r.shape[-2:] == (H, W)
            else F.interpolate(r, size=(H, W), mode='nearest')
            for r in rotated
        ]
        stacked = torch.stack(rotated, dim=0)           # [4,B,C,H,W]
        return stacked.max(dim=0)[0] if self.mode == 'max' else stacked.mean(dim=0)




# ---------- Depthwise-Separable 3×3 conv ----------
class DWConv3x3(nn.Sequential):
    def __init__(self, ch, stride=1):
        super().__init__(
            nn.Conv2d(ch, ch, 3, stride, 1, groups=ch, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )

# ---------- Squeeze-Excitation ----------
class SE(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class RConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, angles=[0, 45, 90, 135]):
        super().__init__()
        self.angles = angles
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        outputs = []
        for angle in self.angles:
            x_rot = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR)
            feat = self.conv(x_rot)
            feat_inv = TF.rotate(feat, -angle, interpolation=TF.InterpolationMode.BILINEAR)
            outputs.append(feat_inv)
        return torch.mean(torch.stack(outputs, dim=0), dim=0)


class RInvariantNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        self.layer1 = RConv(in_channels, base_channels)        # C1
        self.layer2 = RConv(base_channels, base_channels * 2)  # C2
        self.layer3 = RConv(base_channels * 2, base_channels * 4)  # C3
        self.layer4 = RConv(base_channels * 4, base_channels * 8)  # C4 (P4)
        self.layer5 = RConv(base_channels * 8, base_channels * 8)  # C5 (P5)

        self.out_channels = base_channels * 16  # After concatenation

    def forward(self, x):
        x = self.layer1(x)    # → C1
        x = self.layer2(x)    # → C2
        x = self.layer3(x)    # → C3
        p4 = self.layer4(x)   # → C4 (P4)
        p5 = self.layer5(p4)  # → C5 (P5)

        # Upsample p5 to match p4 size
        p5_upsampled = F.interpolate(p5, size=p4.shape[-2:], mode='bilinear', align_corners=False)
        # p4_upsampled = F.interpolate(p4, size=x1.shape[-2:], mode='bilinear', align_corners=False)

        # Concatenate to form feature pyramid
        fused = torch.cat([p4, p5_upsampled], dim=1)  # dim=1 is channel dim

        return fused  # Shape: [B, C4+C5, H, W]

class SimpleFPN(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(SimpleFPN, self).__init__()

        # 主干网络（Backbone） - 3个stage，提取C1, C2, C3
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),  # C1
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),  # C2
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),  # C3
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )

        # 横向连接，统一通道数为 base_channels
        self.lateral3 = nn.Conv2d(base_channels * 4, base_channels, kernel_size=1)
        self.lateral2 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)
        self.lateral1 = nn.Conv2d(base_channels, base_channels, kernel_size=1)

    def forward(self, x):
        c1 = self.stage1(x)  # 原图大小
        c2 = self.stage2(c1)  # 1/2 尺度
        c3 = self.stage3(c2)  # 1/4 尺度

        # 构建金字塔 P3 -> P2 -> P1
        p3 = self.lateral3(c3)  # 最底层
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest')

        # —— 补全尺寸统一 ——（将所有特征图统一为 p1 尺寸）
        target_size = p1.shape[2:]
        p2_up = F.interpolate(p2, size=target_size, mode='nearest')
        p3_up = F.interpolate(p3, size=target_size, mode='nearest')

        # —— 通道拼接后返回一个特征图 ——（维度：[B, 3*C, H, W]）
        return torch.cat([p1, 100*p2_up, p3_up], dim=1)

        # return [p1, p2, p3]  # 从高分辨率到低分辨率的特征图


# class SimpleFPNShift(nn.Module):
#     """
#     结构：
#       C1:  Conv3x3 → DWConv3x3 + SE   (H, W)
#       C2:  Conv3x3 stride=2           (H/2, W/2)
#       C3:  Conv3x3 stride=2           (H/4, W/4)
#       P3→P2→P1 与之前相同，拼接后再 DWConv 融合
#     返回: [B, 3*base_channels, H, W]
#     """
#     def __init__(self, in_channels=3, base_channels=32):
#         super().__init__()
#         C = base_channels
#         # ---- C1 更深语义 & 大感受野
#         self.c1 = nn.Sequential(
#             nn.Conv2d(in_channels, C, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(C), nn.ReLU(inplace=True),
#             DWConv3x3(C), SE(C)
#         )
#         # ---- 下采样层
#         self.c2 = nn.Sequential(
#             nn.Conv2d(C, C*2, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(C*2), nn.ReLU(inplace=True)
#         )
#         self.c3 = nn.Sequential(
#             nn.Conv2d(C*2, C*4, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(C*4), nn.ReLU(inplace=True)
#         )
#
#         # ---- 下采样层
#         self.c4 = nn.Sequential(
#             nn.Conv2d(C*4, C*4, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(C*2), nn.ReLU(inplace=True)
#         )
#         self.c5 = nn.Sequential(
#             nn.Conv2d(C*4, C*4, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(C*4), nn.ReLU(inplace=True)
#         )
#
#         # ---- lateral 统一通道
#         self.lat3 = nn.Conv2d(C*4, C, 1)
#         self.lat2 = nn.Conv2d(C*2, C, 1)
#         self.lat1 = nn.Conv2d(C,   C, 1)
#         # ---- 拼接后再轻量融合
#         self.fuse = DWConv3x3(C*3)
#         self.rpool = RotatePool(mode='max')  # ⬅️ 新增
#
#     def forward(self, x):
#         c1 = self.c1(x)          # H
#         c2 = self.c2(c1)         # H/2
#         c3 = self.c3(c2)         # H/4
#
#         c4 = self.c4(c3)         # H/8
#         c5 = self.c5(c4)         # H/16
#
#         # print(c3.shape,c5.shape)
#
#         p3 = self.lat3(c5)
#         p2 = self.lat2(c3) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
#         p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest')
#
#         p2_up = F.interpolate(p2, size=p1.shape[2:], mode='nearest')
#         p3_up = F.interpolate(p3, size=p1.shape[2:], mode='nearest')
#         out   = torch.cat([0.1*p1, 0.5*p2_up, p3_up], dim=1)  # 3C
#
#         out = self.fuse(out)
#         out = self.rpool(out)  # ⬅️ 旋转池化
#
#         return out

class SimpleFPNShift(nn.Module):
    """
    深浅结合: 仅使用 C3 (H/4) 与 C5 (H/16) 金字塔, 最终输出尺寸 = 输入尺寸, 通道 = 2*base.
    """
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        C = base_channels
        # ── 上半同前 ─────────────────────────
        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C), nn.ReLU(inplace=True),
            DWConv3x3(C), SE(C)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(C, C * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C * 2), nn.ReLU(inplace=True)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(C * 2, C * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C * 4), nn.ReLU(inplace=True)
        )
        # ── 继续两次下采样得到 C4、C5 ─────────
        self.c4 = nn.Sequential(
            nn.Conv2d(C * 4, C * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C * 4), nn.ReLU(inplace=True)   # 修正维度
        )
        self.c5 = nn.Sequential(
            nn.Conv2d(C * 4, C * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C * 4), nn.ReLU(inplace=True)
        )
        # ── lateral 统一通道 (C) 仅 C3 & C5 ──
        self.lat5 = nn.Conv2d(C * 4, C, 1)
        self.lat3 = nn.Conv2d(C * 4, C, 1)

        # 拼接后融合 (2C → 2C)
        self.fuse = DWConv3x3(C * 2)
        self.rpool = RotatePool(mode="max")

    def forward(self, x):
        c1 = self.c1(x)           # H
        c2 = self.c2(c1)          # H/2
        c3 = self.c3(c2)          # H/4
        c4 = self.c4(c3)          # H/8
        c5 = self.c5(c4)          # H/16

        # ----- 金字塔 -----
        p5 = self.lat5(c5)                                     # H/16
        p3 = self.lat3(c3) + F.interpolate(
            p5, size=c3.shape[2:], mode="nearest")             # H/4

        # 上采样到原图尺寸
        p5_up = F.interpolate(p5, size=c1.shape[2:], mode="nearest")
        p3_up = F.interpolate(p3, size=c1.shape[2:], mode="nearest")

        out = torch.cat([p3_up, p5_up], dim=1)                 # [B, 2C, H, W]
        out = self.fuse(out)
        out = self.rpool(out)
        return out




class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        mid = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.block = nn.Sequential(
            nn.Conv2d(cin, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
            DWConv3x3(mid, stride),
            nn.Conv2d(mid, cout, 1, bias=False),
            nn.BatchNorm2d(cout)
        )
    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out

class MobileFPN(nn.Module):
    """
    取自 MobileNetV2 前几层：
      - stem conv 3x3 stride1
      - invRes(expand=1) stride1  (C1)
      - invRes(expand=6) stride2  (C2)
      - invRes(expand=6) stride2  (C3)
    其余同 FPN+拼接。
    """
    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()
        C = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C), nn.ReLU6(inplace=True)
        )
        self.ir1 = InvertedResidual(C,   C,   1, 1)   # C1
        self.ir2 = InvertedResidual(C,   C*2, 2, 6)   # C2
        self.ir3 = InvertedResidual(C*2, C*4, 2, 6)   # C3

        self.lat3 = nn.Conv2d(C*4, C, 1)
        self.lat2 = nn.Conv2d(C*2, C, 1)
        self.lat1 = nn.Conv2d(C,   C, 1)
        self.fuse = DWConv3x3(C*3)

    def forward(self, x):
        x  = self.stem(x)
        c1 = self.ir1(x)
        c2 = self.ir2(c1)
        c3 = self.ir3(c2)

        p3 = self.lat3(c3)
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='nearest')
        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='nearest')

        p2_up = F.interpolate(p2, size=p1.shape[2:], mode='nearest')
        p3_up = F.interpolate(p3, size=p1.shape[2:], mode='nearest')
        out   = torch.cat([p1, p2_up, p3_up], dim=1)
        return self.fuse(out)



class TemplateMatchingNet(nn.Module):
    def __init__(self):
        super(TemplateMatchingNet, self).__init__()
        self.feature_extractor = RInvariantNet()

    def forward(self, src_img, tem_img):
        fs = self.feature_extractor(src_img)  # [B,C,H,W]
        ft = self.feature_extractor(tem_img)  # [B,C,h,w]

        # 关键：kernel 的高宽
        kH, kW = ft.shape[-2:]
        pad = (kW // 2, kH // 2)  # (padW, padH)

        fused = F.conv2d(fs, ft, padding=pad)  # [B,1,H,W]

        # # 映射到 0-1 区间
        fused_min = fused.amin(dim=(2, 3), keepdim=True)
        fused_max = fused.amax(dim=(2, 3), keepdim=True)
        fused = (fused - fused_min) / (fused_max - fused_min + 1e-8)  # 加1e-8防止除0

        return fused
















import torch
import torch.nn as nn
import torch.nn.functional as F

# ------- 基础模块 -------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )

class Up(nn.Module):
    """Upsample → Conv 1×1 → ConvBlock"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
        )
        self.conv = conv_block(in_ch, out_ch)  # 因为 concat，所以 in_ch=skip+up

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ------- UNet-Lite -------
class UNetLite(nn.Module):
    def __init__(self, in_ch=4, base_ch=32, out_activation="sigmoid"):
        super().__init__()
        c1, c2, c3, c4 = (base_ch, base_ch*2, base_ch*4, base_ch*8)

        self.enc0 = conv_block(in_ch, c1)
        self.enc1 = conv_block(c1,  c2)
        self.enc2 = conv_block(c2,  c3)
        self.enc3 = conv_block(c3,  c4)

        self.pool = nn.MaxPool2d(2, 2)

        self.dec2 = Up(c4, c3)
        self.dec1 = Up(c3, c2)
        self.dec0 = Up(c2, c1)

        self.outc = nn.Conv2d(c1, 1, 1)

        if out_activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif out_activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        # Encoder
        s0 = self.enc0(x)      # [B,c1,H,W]
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(self.pool(s1))
        bottleneck = self.enc3(self.pool(s2))

        # Decoder
        d2 = self.dec2(bottleneck, s2)
        d1 = self.dec1(d2,        s1)
        d0 = self.dec0(d1,        s0)

        out = self.act(self.outc(d0))
        return out
