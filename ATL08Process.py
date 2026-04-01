import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
def preprocess_atl08(file_path, beam='gt1l'):
    """
    读取ATL08数据，执行论文预处理步骤：
    1. 剔除SNR<3的数据
    2. 剔除水平方向光子缺失率≥20%的数据
    3. 提取论文所需特征+参考标签（IGBP地表覆盖）
    """
    data = {}
    with h5py.File(file_path, 'r') as f:
        # 1. 提取基础参数（论文表1属性）
        # 光子数量（每个100m分段）
        data['photon_num'] = f[f'{beam}/land_segments/n_seg_ph'][:]
        # 信噪比（SNR）
        data['snr'] = f[f'{beam}/land_segments/snr'][:]
        # 太阳高度角、方位角
        data['solar_elevation'] = f[f'{beam}/land_segments/solar_elevation'][:]
        data['solar_azimuth'] = f[f'{beam}/land_segments/solar_azimuth'][:]
        # 云置信度
        data['cloud_conf'] = f[f'{beam}/land_segments/cloud_confidence'][:]

        # 2. 垂直方向光子比例（地面/冠层/冠层顶部）
        # 地面光子数
        data['terrain_ph_cnt'] = f[f'{beam}/land_segments/terrain/ph_cnt'][:]
        # 冠层光子数（含顶部）
        data['canopy_ph_cnt'] = f[f'{beam}/land_segments/canopy/ph_cnt'][:]
        # 冠层顶部光子数
        data['top_canopy_ph_cnt'] = f[f'{beam}/land_segments/canopy/top_ph_cnt'][:]

        # 3. 水平方向光子分布（5个20m子分段）
        # 地面光子水平分布（5个子分段的光子数）
        terrain_horizontal = f[f'{beam}/land_segments/terrain/subset_ph_cnt'][:]  # (n,5)
        # 冠层光子水平分布
        canopy_horizontal = f[f'{beam}/land_segments/canopy/subset_ph_cnt'][:]  # (n,5)

        # 4. 参考标签（IGBP地表覆盖分类，论文用于验证）
        data['igbp_landcover'] = f[f'{beam}/land_segments/segment_landcover'][:]

        # 5. 沿轨距离（用于去噪）
        data['dist_along'] = f[f'{beam}/land_segments/longitude'][:]  # 或用segment_id映射
        data['elevation'] = f[f'{beam}/land_segments/terrain/h_te_median'][:]

    # 转换为DataFrame便于处理
    df = pd.DataFrame(data)

    # 论文预处理步骤1：剔除SNR<3的数据
    df = df[df['snr'] >= 3].reset_index(drop=True)

    # 论文预处理步骤2：剔除水平方向光子缺失率≥20%的数据
    # 计算地面光子水平缺失率（空值数/总子分段数）
    df['terrain_missing_rate'] = np.sum(terrain_horizontal == 0, axis=1) / 5
    # 计算冠层光子水平缺失率
    df['canopy_missing_rate'] = np.sum(canopy_horizontal == 0, axis=1) / 5
    # 任一缺失率≥20%则剔除
    df = df[(df['terrain_missing_rate'] < 0.2) & (df['canopy_missing_rate'] < 0.2)].reset_index(drop=True)

    # 特征工程：计算论文表1的10类属性
    # 1. 垂直方向比例
    df['terrain_vertical_ratio'] = df['terrain_ph_cnt'] / df['photon_num']
    df['canopy_vertical_ratio'] = df['canopy_ph_cnt'] / df['photon_num']
    df['top_canopy_vertical_ratio'] = df['top_canopy_ph_cnt'] / df['photon_num']

    # 2. 水平方向比例（计算非空分布比例）
    df['terrain_horizontal_ratio'] = np.sum(terrain_horizontal > 0, axis=1) / 5
    df['canopy_horizontal_ratio'] = np.sum(canopy_horizontal > 0, axis=1) / 5

    # 3. 填充可能的NaN值
    df = df.fillna(0)

    # 构建最终特征集（论文表1的10个属性）
    features = [
        'photon_num', 'terrain_vertical_ratio', 'canopy_vertical_ratio',
        'top_canopy_vertical_ratio', 'terrain_horizontal_ratio', 'canopy_horizontal_ratio',
        'snr', 'solar_elevation', 'solar_azimuth', 'cloud_conf'
    ]

    # 构建标签（论文四类地表：水体/森林/低植被/城市/裸地）
    def map_landcover(igbp_code):
        if igbp_code == 16:  # 水体
            return 0
        elif 1 <= igbp_code <= 5:  # 森林（1-5类）
            return 1
        elif igbp_code in [6, 7, 8, 9, 10]:  # 低植被（灌丛/草原/稀树草原）
            return 2
        elif igbp_code in [13, 14]:  # 城市/裸地
            return 3
        else:  # 其他类别剔除
            return -1

    df['label'] = df['igbp_landcover'].apply(map_landcover)
    df = df[df['label'] != -1].reset_index(drop=True)  # 剔除无效标签

    # 生成去噪所需的光子点云（沿轨距离+高程）
    cloud = np.hstack([df['dist_along'].values.reshape(-1, 1), df['elevation'].values.reshape(-1, 1)])

    return df, features, cloud


