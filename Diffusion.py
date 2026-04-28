"""
GEDI + ICESat-2 数据融合
对于每个GEDI足迹：
    1. 获取GEDI足迹的经纬度、波形特征
    2. 搜索周围的ICESat-2光子
    3. 统计缓冲区内光子的特征
    4. 融合成一个样本用于分类
"""
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from geopy.distance import distance
from tqdm import tqdm
import warnings
from Load_filtered_data import load_filtered_gedi_data
warnings.filterwarnings('ignore')


# ==================== 1. GEDI数据读取函数 ====================


# ==================== 2. ICESat-2 光子搜索函数 ====================
def search_icesat2_photons(icesat2_files, target_lon, target_lat,
                           radius_m=50, conf_threshold=2, verbose=False):
    """
    在指定点周围搜索ICESat-2光子

    参数:
    ----------
    icesat2_files : list
        ICESat-2 H5文件路径列表
    target_lon, target_lat : float
        目标点经纬度
    radius_m : float
        搜索半径（米）
    conf_threshold : int
        信号置信度阈值（>= 该值视为信号光子）
    verbose : bool
        是否打印详细信息

    返回:
    ----------
    pd.DataFrame : 缓冲区内光子数据
    """

    from geopy.distance import distance

    all_photons = []

    for filepath in icesat2_files:
        try:
            with h5py.File(filepath, 'r') as f:
                beams = [k for k in f.keys() if k.startswith('gt')]

                for beam in beams:
                    if 'heights' not in f[beam]:
                        continue

                    heights = f[beam]['heights']

                    # 读取经纬度
                    lats = np.array(heights['lat_ph'][:]).flatten()
                    lons = np.array(heights['lon_ph'][:]).flatten()

                    # 快速粗筛（约5km范围）
                    lat_min, lat_max = target_lat - 0.05, target_lat + 0.05
                    lon_min, lon_max = target_lon - 0.05, target_lon + 0.05

                    mask = (lats >= lat_min) & (lats <= lat_max) & \
                           (lons >= lon_min) & (lons <= lon_max)

                    if not mask.any():
                        continue

                    # 读取其他数据
                    h_ph = np.array(heights['h_ph'][:]).flatten()
                    dist_along = np.array(
                        heights['dist_ph_along'][:]).flatten() if 'dist_ph_along' in heights else np.zeros_like(h_ph)

                    # 置信度
                    if 'signal_conf_ph' in heights:
                        conf_raw = heights['signal_conf_ph'][:]
                        if conf_raw.ndim == 2:
                            conf = conf_raw[:, 0].flatten()
                        else:
                            conf = conf_raw.flatten()
                    else:
                        conf = np.ones_like(h_ph) * 2

                    # 筛选有效光子
                    valid = (h_ph > -900) & ~np.isnan(h_ph) & mask

                    if not valid.any():
                        continue

                    # 精确计算距离
                    indices = np.where(valid)[0]
                    for idx in indices:
                        try:
                            dist_m = distance((target_lat, target_lon),
                                              (lats[idx], lons[idx])).meters
                            if dist_m <= radius_m:
                                all_photons.append({
                                    'h': h_ph[idx],
                                    'dist_along': dist_along[idx],
                                    'lat': lats[idx],
                                    'lon': lons[idx],
                                    'conf': conf[idx],
                                    'distance_to_gedi': dist_m,
                                    'beam_source': beam,
                                    'file_source': Path(filepath).name
                                })
                        except:
                            continue

        except Exception as e:
            if verbose:
                print(f"    错误: {e}")
            continue

    if not all_photons:
        return pd.DataFrame()

    df = pd.DataFrame(all_photons)
    df['is_signal'] = df['conf'] >= conf_threshold

    return df


# ==================== 3. 统计缓冲区光子特征 ====================
def compute_photon_statistics(photons_df):
    """
    计算缓冲区内光子的统计特征

    参数:
    ----------
    photons_df : pd.DataFrame
        光子数据

    返回:
    ----------
    dict : 统计特征字典
    """
    stats = {}

    if len(photons_df) == 0:
        stats['icesat2_total_count'] = 0
        stats['icesat2_signal_count'] = 0
        stats['icesat2_noise_count'] = 0
        return stats

    signal_photons = photons_df[photons_df['is_signal']]
    noise_photons = photons_df[~photons_df['is_signal']]

    # 基本计数
    stats['icesat2_total_count'] = len(photons_df)
    stats['icesat2_signal_count'] = len(signal_photons)
    stats['icesat2_noise_count'] = len(noise_photons)

    if len(signal_photons) > 0:
        # 高程统计
        stats['icesat2_h_mean'] = signal_photons['h'].mean()
        stats['icesat2_h_median'] = signal_photons['h'].median()
        stats['icesat2_h_std'] = signal_photons['h'].std()
        stats['icesat2_h_min'] = signal_photons['h'].min()
        stats['icesat2_h_max'] = signal_photons['h'].max()
        stats['icesat2_h_q25'] = signal_photons['h'].quantile(0.25)
        stats['icesat2_h_q75'] = signal_photons['h'].quantile(0.75)
        stats['icesat2_h_range'] = signal_photons['h'].max() - signal_photons['h'].min()

        # 高程百分位数
        for p in [10, 25, 50, 75, 90]:
            stats[f'icesat2_h_p{p}'] = signal_photons['h'].quantile(p / 100)

        # 距离统计
        stats['icesat2_dist_mean'] = signal_photons['distance_to_gedi'].mean()
        stats['icesat2_dist_median'] = signal_photons['distance_to_gedi'].median()
        stats['icesat2_dist_std'] = signal_photons['distance_to_gedi'].std()
        stats['icesat2_dist_min'] = signal_photons['distance_to_gedi'].min()
        stats['icesat2_dist_max'] = signal_photons['distance_to_gedi'].max()

        # 置信度分布
        for conf_val in [2, 3, 4]:
            count = (signal_photons['conf'] == conf_val).sum()
            stats[f'icesat2_conf_{conf_val}_count'] = count
            stats[f'icesat2_conf_{conf_val}_pct'] = count / len(signal_photons) * 100

        # 各波束光子数
        for beam in signal_photons['beam_source'].unique():
            stats[f'icesat2_beam_{beam}_count'] = (signal_photons['beam_source'] == beam).sum()

    if len(noise_photons) > 0:
        # 噪声光子统计（可选）
        stats['icesat2_noise_h_mean'] = noise_photons['h'].mean()
        stats['icesat2_noise_h_std'] = noise_photons['h'].std()

    return stats


# ==================== 4. 融合单个GEDI足迹 ====================
def fuse_single_footprint(gedi_footprint, icesat2_files,
                          radius_m=50, conf_threshold=2, verbose=False):
    """
    融合单个GEDI足迹和周围的ICESat-2数据

    参数:
    ----------
    gedi_footprint : pd.Series
        单个GEDI足迹数据
    icesat2_files : list
        ICESat-2文件路径列表
    radius_m : float
        搜索半径（米）
    conf_threshold : int
        ICESat-2信号置信度阈值
    verbose : bool
        是否打印详细信息

    返回:
    ----------
    dict : 融合后的特征向量
    """
    # 提取GEDI中心点
    center_lon = gedi_footprint['lon']
    center_lat = gedi_footprint['lat']

    # 搜索ICESat-2光子
    photons = search_icesat2_photons(
        icesat2_files, center_lon, center_lat, radius_m, conf_threshold, verbose=verbose
    )

    # 计算光子统计特征
    photon_stats = compute_photon_statistics(photons)

    # 融合GEDI特征和ICESat-2特征
    fused = {}

    # 添加GEDI特征
    for key, value in gedi_footprint.items():
        fused[f'gedi_{key}'] = value

    # 添加ICESat-2统计特征
    for key, value in photon_stats.items():
        fused[key] = value

    # 添加元数据
    fused['fusion_radius_m'] = radius_m
    fused['icesat2_conf_threshold'] = conf_threshold

    return fused


# ==================== 5. 批量融合 ====================
def batch_fusion(gedi_file, icesat2_files, radius_m=50, conf_threshold=2,
                 max_footprints=None, spatial_filter=None,
                 output_file=None, verbose=True):
    """
    批量融合所有GEDI足迹

    参数:
    ----------
    gedi_file : str
        GEDI H5文件路径
    icesat2_files : list
        ICESat-2文件路径列表
    radius_m : float
        搜索半径（米）
    conf_threshold : int
        ICESat-2信号置信度阈值
    max_footprints : int, optional
        最大处理数量
    spatial_filter : dict, optional
        空间过滤
    output_file : str, optional
        输出文件路径
    verbose : bool
        是否打印详细信息

    返回:
    ----------
    pd.DataFrame : 融合后的数据集
    """

    # 加载GEDI足迹
    if spatial_filter:
        gedi_df = load_filtered_gedi_data(
            gedi_file
        )
    else:
        gedi_df = load_filtered_gedi_data(gedi_file)

    if len(gedi_df) == 0:
        print("没有找到有效的GEDI足迹！")
        return pd.DataFrame()

    print(f"\n开始融合 {len(gedi_df)} 个GEDI足迹...")
    print(f"搜索半径: {radius_m} 米")
    print(f"ICESat-2置信度阈值: >= {conf_threshold}")

    # 批量融合
    fused_results = []

    for idx, row in tqdm(gedi_df.iterrows(), total=len(gedi_df), desc="融合进度"):
        try:
            result = fuse_single_footprint(
                row, icesat2_files, radius_m, conf_threshold, verbose=False
            )
            fused_results.append(result)
        except Exception as e:
            if verbose:
                print(f"  融合失败 (idx={idx}): {e}")
            continue

    # 转换为DataFrame
    fused_df = pd.DataFrame(fused_results)

    # 统计
    print(f"\n{'=' * 70}")
    print("融合完成统计")
    print(f"{'=' * 70}")
    print(f"成功融合: {len(fused_df)} / {len(gedi_df)} 个足迹")
    print(f"有ICESat-2光子的足迹: {(fused_df['icesat2_total_count'] > 0).sum()}")
    print(f"有信号光子的足迹: {(fused_df['icesat2_signal_count'] > 0).sum()}")

    # 保存结果
    if output_file and len(fused_df) > 0:
        fused_df.to_csv(output_file, index=False)
        print(f"\n融合结果已保存到: {output_file}")

    return fused_df


# ==================== 6. 主程序 ====================
if __name__ == "__main__":

    # 文件路径
    gedi_file = r"D:\研究生\SanFrancisco\GEDIdata\merged_gedi_data_20260411.h5"

    icesat2_files = [
        r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20241117131625_09512502_007_01_subsetted.h5",
        r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20240715072439_04252406_007_01_subsetted.h5",
        r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20240818173628_09512402_007_01_subsetted.h5",
        r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20241014030438_04252506_007_01_subsetted.h5",
        r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20241112014035_08672506_007_01_subsetted.h5",
    ]

    # 测试：只处理少量足迹
    print("=" * 70)
    print("GEDI + ICESat-2 数据融合")
    print("=" * 70)

    # 空间过滤（旧金山区域）
    spatial_filter = {
        'lat_min': 37.5,
        'lat_max': 38.0,
        'lon_min': -122.6,
        'lon_max': -122.3
    }

    # 批量融合（先测试100个）
    fused_data = batch_fusion(
        gedi_file=gedi_file,
        icesat2_files=icesat2_files,
        radius_m=50,  # 50米缓冲区
        conf_threshold=2,  # 置信度 >= 2 为信号
        max_footprints=100,  # 先测试100个
        spatial_filter=spatial_filter,
        output_file=r"D:\研究生\SanFrancisco\gedi_icesat2_fusion.csv",
        verbose=True
    )

    # 显示融合结果示例
    if len(fused_data) > 0:
        print(f"\n融合数据集形状: {fused_data.shape}")
        print(f"特征列数: {len(fused_data.columns)}")
        print("\n前5行数据:")
        print(fused_data.head())

        # 显示列名
        print("\n所有特征列:")
        for col in fused_data.columns:
            print(f"  {col}")