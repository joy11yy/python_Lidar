import rasterio
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体为黑体或微软雅黑
plt.rcParams['axes.unicode_minus'] = False
# 文件路径
file_path = r"D:\研究生\SanFrancisco\GLC_FCS30_2020_W125N40.tif"

try:
    # 打开数据集
    with rasterio.open(file_path) as src:
        # 1. 查看基本信息
        print(f"坐标系: {src.crs}")
        print(f"图像尺寸: {src.width} x {src.height}")
        print(f"像素分辨率: {src.res}")
        print(f"数据边界: {src.bounds}")

        # 2. 读取数据数组 (第一个波段)
        data = src.read(1)

        # 3. 简单统计
        import numpy as np

        unique_values, counts = np.unique(data, return_counts=True)
        print("分类代码及像素数量:")
        for val, count in zip(unique_values, counts):
            if val != src.nodata:  # 忽略无效值
                print(f"代码 {val}: {count} 像素")

    # 4. 可视化 (可选)
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='tab20')  # 使用分类 colormap
    plt.title("GLC_FCS30 地表覆盖分类")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"读取失败: {e}")