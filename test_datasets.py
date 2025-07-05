from mydatasets import DensityRefineDataset

root = r"E:\python_prj\2025_5_26\data_gen\train_data"  # 绝对路径
ds = DensityRefineDataset(
        root_dir=root,
        canvas_size=(640, 640),
        base_name="density_pred.png",   # ← 改成你目录里的名字
)
print("dataset len =", len(ds))
x, y = ds[0]
print("input :", x.shape, x.dtype)
print("gt    :", y.shape, y.dtype)