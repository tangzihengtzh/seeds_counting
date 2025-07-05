from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

def load_and_resize_image(path, H, W):
    """
    读取指定路径的JPG图像，缩放为(H, W)，并转为形状 [1, 3, H, W] 的张量。
    同时返回x和y方向的缩放系数（原始高宽 -> 目标高宽）。
    """
    image = Image.open(path).convert('RGB')
    orig_W, orig_H = image.size  # 注意PIL中是 (W, H)

    scale_x = W / orig_W
    scale_y = H / orig_H

    transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)

    return tensor, scale_x, scale_y


def load_and_scale_image(path, scale_x, scale_y):
    """
    读取指定路径的JPG图像，按给定的 scale_x 和 scale_y 进行缩放，
    返回形状为 [1, 3, H, W] 的张量和实际缩放后的尺寸 (H, W)。
    """
    image = Image.open(path).convert('RGB')
    orig_W, orig_H = image.size

    new_W = int(orig_W * scale_x)
    new_H = int(orig_H * scale_y)

    image_resized = image.resize((new_W, new_H), Image.BILINEAR)
    tensor = TF.to_tensor(image_resized).unsqueeze(0)  # [1, 3, H, W]

    return tensor, new_H, new_W

def print_cuda_usage():
    allocated = torch.cuda.memory_allocated() / 1024**2  # 已分配内存 (MB)
    reserved  = torch.cuda.memory_reserved()  / 1024**2  # 保留内存池 (MB)
    print(f"CUDA Memory Allocated: {allocated:.2f} MB")
    print(f"CUDA Memory Reserved : {reserved:.2f} MB")





def enhance_contrast(array: np.ndarray, gamma: float = 1.0, clip: bool = True) -> np.ndarray:
    """
    增强 0~1 范围内的 np.array 对比度。

    参数：
        array : np.ndarray
            输入数组，元素值应该在 [0, 1] 范围内。
        gamma : float
            伽马值，<1 会增强对比度，>1 会减弱对比度，默认值为1（不开启）。
        clip : bool
            是否将结果裁剪到 [0,1] 区间，默认True。

    返回：
        np.ndarray
            经过增强后的数组。
    """
    # Step 1: 线性拉伸到 0~1（增强对比度）
    arr_min, arr_max = array.min(), array.max()
    stretched = (array - arr_min) / (arr_max - arr_min + 1e-8)

    # Step 2: 伽马校正
    if gamma != 1.0:
        stretched = np.power(stretched, gamma)

    # Step 3: 可选裁剪
    if clip:
        stretched = np.clip(stretched, 0, 1)

    return stretched
