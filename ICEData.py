""""
原本GEDI数据下载的是2024年下半年的点
对于每个GEDI足迹（有经纬度+波形特征）：
    1. 筛选所有ICESat-2光子
    2. 计算每个光子到GEDI足迹中心的距离
    3. 保留距离 < R 的光子
    4. 统计缓冲区内光子的特征（均值、中位数、标准差等）
    5. 将GEDI特征 + 统计后的ICESat-2特征 → 一个融合样本
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Tools.scripts.objgraph import ignore
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings
from pathlib import Path
import glob


warnings.filterwarnings('ignore')


def explore_atl03_file(filepath):
    """探索ATL03文件结构"""
    with h5py.File(filepath, 'r') as f:
        print("=" * 70)
        print("文件顶层结构:")
        print(f"顶层keys: {list(f.keys())}")

        # 找出所有波束
        beams = [k for k in f.keys() if k.startswith('gt')]
        print(f"\n可用波束: {beams}")

        beam_info = {}
        for beam in beams:
            heights = f[beam]['heights']
            h_ph = heights['h_ph'][:]
            h_ph = np.array(h_ph).flatten()
            n_photons = len(h_ph)
            valid = (h_ph > -900) & ~np.isnan(h_ph)

            beam_info[beam] = {
                'total': n_photons,
                'valid': valid.sum(),
                'h_range': (h_ph[valid].min(), h_ph[valid].max()) if valid.sum() > 0 else (None, None)
            }
            print(f"\n波束 {beam}:")
            print(f"  总光子数: {n_photons:,}")
            print(f"  有效光子数: {valid.sum():,}")
            print(f"  高程范围: {beam_info[beam]['h_range'][0]:.2f} ~ {beam_info[beam]['h_range'][1]:.2f} m")

            # 检查是否有信号置信度
            if 'signal_conf_ph' in heights:
                conf = heights['signal_conf_ph'][:]
                if conf.ndim == 2:
                    #取第0列的代表陆地的置信度
                    conf = conf[:, 0]
                conf = np.array(conf).flatten()
                conf_valid = conf[valid]
                print(f"  信号置信度分布: {np.unique(conf_valid, return_counts=True)}")

        return beams, beam_info


def load_atl03_beam(filepath, beam, conf_threshold=2):
    """加载指定波束的数据，并筛选高质量光子"""
    with h5py.File(filepath, 'r') as f:
        if beam not in f or 'heights' not in f[beam]:
            return pd.DataFrame()

        heights = f[beam]['heights']

        # 读取数据并确保1维，每个光子就是一个高程数据，但是有些会有2维，所以确保一维
        def safe_read(key):
            data = heights[key][:]
            return np.array(data).flatten()
        #h_ph是光子高程，每个光子相对于参考椭球面的高程
        h_ph = safe_read('h_ph')
        dist_ph = safe_read('dist_ph_along')
        lat_ph = safe_read('lat_ph')
        lon_ph = safe_read('lon_ph')

        # 处理信号置信度 2代表中等置信度
        if 'signal_conf_ph' in heights:
            conf_raw = heights['signal_conf_ph'][:]
            #nidm指的是number of dimensions维度数量
            if conf_raw.ndim == 2:
                conf = conf_raw[:, 0].flatten()
            else:
                conf = conf_raw.flatten()
        else:
            conf = np.ones_like(h_ph) * 2

        # 质量标志
        if 'quality_ph' in heights:
            quality = safe_read('quality_ph')
        else:
            quality = np.ones_like(h_ph) * 0

        # 筛选有效光子
        valid = (h_ph > -900) & ~np.isnan(h_ph)

        # 按置信度筛选（推荐 conf >= 2）
        signal = valid & (conf >= conf_threshold)

        df = pd.DataFrame({
            'h': h_ph[valid],
            'dist': dist_ph[valid],
            'lat': lat_ph[valid],
            'lon': lon_ph[valid],
            'conf': conf[valid],
            'quality': quality[valid],
            'is_signal': signal[valid]
        })

        return df

#读取多个文件的数据
def load_multiple_files(filepath_list, conf_threshold=2, verbose=True):
    """
    批量读取多个ATL03文件

    参数:
    ----------
    filepath_list : list
        文件路径列表
    conf_threshold : int
        信号置信度阈值
    verbose : bool
        是否打印进度

    返回:
    ----------
    pd.DataFrame : 合并后的所有光子数据
    """
    all_dfs = []

    for i, filepath in enumerate(filepath_list, 1):
        if verbose:
            print(f"[{i}/{len(filepath_list)}] 正在读取: {Path(filepath).name}")

        # 打开文件获取所有波束
        with h5py.File(filepath, 'r') as f:
            beams = [k for k in f.keys() if k.startswith('gt')]

            # 读取每个波束的数据
            for beam in beams:
                df_beam = load_atl03_beam(filepath, beam, conf_threshold)
                if len(df_beam) > 0:
                    # 添加来源信息
                    df_beam['file_source'] = Path(filepath).name
                    df_beam['beam_source'] = beam
                    all_dfs.append(df_beam)

    if not all_dfs:
        raise ValueError("没有成功读取任何数据")

    # 合并所有数据
    df_all = pd.concat(all_dfs, ignore_index=True)

    if verbose:
        print(f"\n总计读取 {len(filepath_list)} 个文件")
        print(f"总光子数: {len(df_all):,}")
        print(f"信号光子数: {df_all['is_signal'].sum():,} ({100 * df_all['is_signal'].sum() / len(df_all):.1f}%)")

    return df_all

# ==================== 1. 读取ATL03数据 ====================
filepath = r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20240715072439_04252406_007_01_subsetted.h5"
filepathList=[
    r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20241117131625_09512502_007_01_subsetted.h5",
    r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20240715072439_04252406_007_01_subsetted.h5",
    r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20240818173628_09512402_007_01_subsetted.h5",
    r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20241014030438_04252506_007_01_subsetted.h5",
    r"D:\研究生\SanFrancisco\ICESatdata\ATL03\ATL03_20241112014035_08672506_007_01_subsetted.h5",

]

# ==================== 2. 执行探索和数据加载 ====================
print("=" * 70)
print("步骤1: 探索ATL03文件结构")
print("=" * 70)
beams, beam_info = explore_atl03_file(filepath)

print("\n" + "=" * 70)
print("步骤2: 选择最佳波束并加载数据")
print("=" * 70)
#处理所有波束
# all_beams_data={}
# dfs=[]
# for beam in beams:
#     df_temp=load_atl03_beam(filepath, beam,conf_threshold=2)
#     all_beams_data[beam]=df_temp
#     dfs.append(df_temp)
# df=pd.concat(dfs,ignore_index=True)
# # 优先选择强波束（gt1r, gt2r, gt3r）中有效光子最多的
# strong_beams = [b for b in beams if b.endswith('r')]
# if strong_beams:
#     best_beam = max(strong_beams, key=lambda b: beam_info[b]['valid'])
# else:
#     best_beam = max(beams, key=lambda b: beam_info[b]['valid'])
#
#print(f"选择波束: {best_beam}")
# df = load_atl03_beam(filepath, best_beam, conf_threshold=2)
df=load_multiple_files(filepathList,conf_threshold=2,verbose=True)

print(f"加载完成: {len(df):,} 个有效光子")
print(f"  其中高置信度信号光子 (conf>=2): {df['is_signal'].sum():,} ({100 * df['is_signal'].sum() / len(df):.1f}%)")
signal=df[df['is_signal']]
noise=df[~df['is_signal']]
# ==================== 3. 可视化光子分布 ====================
print("\n" + "=" * 70)
print("步骤3: 原始光子分布与置信度去噪的分布")
print("=" * 70)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 子图1：原始光子分布（按置信度着色）灰色代表噪声光子
# 图1a：原始所有光子（不做任何置信度筛选）
plt.figure(figsize=(14,8))
plt.scatter(df['dist'], df['h'], c='gray', s=1, alpha=0.3, label='所有光子')
plt.xlabel('沿轨距离 (m)', fontsize=11)
plt.ylabel('高程 (m)', fontsize=11)
plt.title('原始所有光子分布（未去噪）', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# 图1b：按置信度着色（你当前的图）
plt.figure(figsize=(14,8))
conf_colors = {0: '#D3D3D3', 1: '#A9A9A9', 2: '#1E88E5', 3: '#2E7D32', 4: '#C62828'}
conf_labels = {0: '噪声(0)', 1: '低(1)', 2: '中(2)', 3: '高(3)', 4: '最高(4)'}
for conf_val, color in conf_colors.items():
    mask = df['conf'] == conf_val
    if mask.sum() > 0:
        plt.scatter(df.loc[mask, 'dist'], df.loc[mask, 'h'],
                   c=color, s=1, alpha=0.5,
                   label=f'置信度{conf_val} ({conf_labels[conf_val]})')
plt.xlabel('沿轨距离 (m)', fontsize=11)
plt.ylabel('高程 (m)', fontsize=11)
plt.title('原始光子分布（按置信度着色）', fontsize=12)
plt.legend(markerscale=3, fontsize=9, loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()

# ==================== 4. 图2：去噪结果对比（信号光子 vs 噪声光子） ====================
print("\n" + "=" * 70)
print("步骤4: 生成图2 - 去噪结果对比")
print("=" * 70)

plt.figure(figsize=(14, 8))
plt.scatter(noise['dist'], noise['h'], c='lightgray', s=1, alpha=0.3,
           label=f'噪声光子 ({len(noise):,})')
plt.scatter(signal['dist'], signal['h'], c='#2E7D32', s=1, alpha=0.5,
           label=f'信号光子 ({len(signal):,})')
plt.xlabel('沿轨距离 (m)', fontsize=12)
plt.ylabel('高程 (m)', fontsize=12)
plt.title(f'去噪结果 (conf >= 2)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== 5. 图3：高程分布直方图 ====================
print("\n" + "=" * 70)
print("步骤5: 生成图3 - 高程分布直方图")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 子图3a：线性坐标
axes[0].hist(df['h'], bins=100, alpha=0.6, color='blue',
            label=f'所有光子 (n={len(df):,})', edgecolor='black', linewidth=0.5)
axes[0].hist(signal['h'], bins=100, alpha=0.6, color='green',
            label=f'信号光子 (n={len(signal):,})', edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('高程 (m)', fontsize=12)
axes[0].set_ylabel('光子数量', fontsize=12)
axes[0].set_title('高程分布直方图 (线性坐标)', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 子图3b：对数坐标
axes[1].hist(df['h'], bins=100, alpha=0.6, color='blue',
            label=f'所有光子 (n={len(df):,})', edgecolor='black', linewidth=0.5)
axes[1].hist(signal['h'], bins=100, alpha=0.6, color='green',
            label=f'信号光子 (n={len(signal):,})', edgecolor='black', linewidth=0.5)
axes[1].set_xlabel('高程 (m)', fontsize=12)
axes[1].set_ylabel('光子数量 (对数坐标)', fontsize=12)
axes[1].set_title('高程分布直方图 (对数坐标)', fontsize=12, fontweight='bold')
axes[1].set_yscale('log')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# ==================== 6. 图4：信号光子密度分布（2D直方图） ====================
print("\n" + "=" * 70)
print("步骤6: 生成图4 - 信号光子密度分布")
print("=" * 70)

plt.figure(figsize=(14, 8))
hb = plt.hist2d(signal['dist'], signal['h'], bins=[200, 100],
                cmap='viridis', cmin=1)
plt.colorbar(hb[3], label='光子计数')
plt.xlabel('沿轨距离 (m)', fontsize=12)
plt.ylabel('高程 (m)', fontsize=12)
plt.title(f'信号光子密度分布', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
# ==================== 7. 图5：沿轨剖面图（高程中位数和分位数） ====================
print("\n" + "=" * 70)
print("步骤7: 生成图5 - 沿轨剖面图")
print("=" * 70)

# 将沿轨距离分成50段
n_segments = 50
signal_sorted = signal.sort_values('dist')
signal_sorted['segment'] = pd.cut(signal_sorted['dist'], bins=n_segments)

# 计算每个段的统计量
segment_stats = signal_sorted.groupby('segment')['h'].agg([
    'median', 'mean', 'std', 'min', 'max', 'count'
]).reset_index()
# 单独计算分位数
segment_stats['q25'] = signal_sorted.groupby('segment')['h'].quantile(0.25).values
segment_stats['q75'] = signal_sorted.groupby('segment')['h'].quantile(0.75).values
# 计算每个段的中心距离
segment_centers = []
for interval in segment_stats['segment']:
    segment_centers.append((interval.left + interval.right) / 2)
segment_stats['dist_center'] = segment_centers

plt.figure(figsize=(14, 8))

# 绘制分位数范围（25%-75%）
plt.fill_between(segment_stats['dist_center'],
                 segment_stats['q25'],
                 segment_stats['q75'],
                 alpha=0.3, color='lightblue', label='25%-75% 分位数范围')

# 绘制中位数线
plt.plot(segment_stats['dist_center'], segment_stats['median'],
         'b-', linewidth=2, label='中位数')

# 绘制均值线
plt.plot(segment_stats['dist_center'], segment_stats['mean'],
         'r--', linewidth=1.5, alpha=0.7, label='均值')

# 绘制极值范围（最小-最大）
plt.fill_between(segment_stats['dist_center'],
                 segment_stats['min'],
                 segment_stats['max'],
                 alpha=0.1, color='gray', label='最小-最大范围')

plt.xlabel('沿轨距离 (m)', fontsize=12)
plt.ylabel('高程 (m)', fontsize=12)
plt.title(f'沿轨剖面图 (信号光子)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# ==================== 8. 图6：光子数量沿轨分布 ====================
print("\n" + "=" * 70)
print("步骤8: 生成图6 - 光子数量沿轨分布")
print("=" * 70)

plt.figure(figsize=(14, 8))

# 绘制所有光子的沿轨分布
dist_bins = np.linspace(df['dist'].min(), df['dist'].max(), 100)
plt.hist(df['dist'], bins=dist_bins, alpha=0.5, color='blue',
         label=f'所有光子 (总: {len(df):,})', edgecolor='black', linewidth=0.5)
plt.hist(signal['dist'], bins=dist_bins, alpha=0.5, color='green',
         label=f'信号光子 (总: {len(signal):,})', edgecolor='black', linewidth=0.5)

plt.xlabel('沿轨距离 (m)', fontsize=12)
plt.ylabel('光子数量', fontsize=12)
plt.title(f'光子数量沿轨分布', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== 9. 详细统计信息 ====================
print("\n" + "=" * 70)
print("步骤9: 详细统计信息")
print("=" * 70)

print(f"\n【所有波束数据概况】")
print(f"  有效光子总数: {len(df):,}")
print(f"  信号光子数 (conf>=2): {len(signal):,} ({100 * len(signal) / len(df):.1f}%)")
print(f"  噪声光子数: {len(noise):,} ({100 * len(noise) / len(df):.1f}%)")
print(f"  高程范围: {df['h'].min():.2f} ~ {df['h'].max():.2f} m")
print(f"  沿轨距离范围: {df['dist'].min():.2f} ~ {df['dist'].max():.2f} m")
print(f"  纬度范围: {df['lat'].min():.4f} ~ {df['lat'].max():.4f} °")
print(f"  经度范围: {df['lon'].min():.4f} ~ {df['lon'].max():.4f} °")

print(f"\n【置信度分布】")
conf_counts = df['conf'].value_counts().sort_index()
for conf_val, count in conf_counts.items():
    print(f"  conf={conf_val}: {count:,} 光子 ({100 * count / len(df):.1f}%)")

# 检查是否有地面光子候选（高程最低的5%）
ground_candidates = signal.nsmallest(int(len(signal) * 0.05), 'h')
print(f"\n【地面光子候选】")
print(f"  最低5%信号光子高程范围: {ground_candidates['h'].min():.2f} ~ {ground_candidates['h'].max():.2f} m")
print(f"  最低5%信号光子数量: {len(ground_candidates)} 个")

# 计算垂直分布范围（分20段）
segments = signal.groupby(pd.cut(signal['dist'], bins=20))
vertical_ranges = segments['h'].agg(lambda x: x.max() - x.min())
print(f"\n【垂直结构】")
print(f"  沿轨分段垂直范围: 均值={vertical_ranges.mean():.2f}m, 中位数={vertical_ranges.median():.2f}m")
print(f"  最大垂直范围: {vertical_ranges.max():.2f}m")
print(f"  最小垂直范围: {vertical_ranges.min():.2f}m")
print(f"  垂直范围标准差: {vertical_ranges.std():.2f}m")

# ==================== 极简保存版本 ====================
print("\n" + "=" * 70)
print("保存ICESat-2数据")
print("=" * 70)

from datetime import datetime

# 生成带日期的文件名
# ==================== 使用 h5py 保存为 H5 格式 ====================
print("\n" + "=" * 70)
print("使用 h5py 保存 ICESat-2 数据为 H5 格式")
print("=" * 70)
current_date = datetime.now().strftime("%Y%m%d")
save_path = Path(r"D:\研究生\SanFrancisco\ICESatdata\processed")
save_path.mkdir(parents=True, exist_ok=True)

output_file = save_path / f"icesat2_photons_{current_date}.h5"

with h5py.File(output_file, 'w') as f:
    # 创建组
    photons_group = f.create_group('photons')

    # 保存每个列作为数据集
    for col in df.columns:
        if df[col].dtype == 'object':  # 字符串类型
            # 将字符串转换为字节字符串
            str_data = df[col].astype(str).values.astype('S')
            f.create_dataset(col, data=str_data, compression='gzip')
        else:# 数值类型
            f.create_dataset(col, data=df[col].values,
                             compression='gzip', compression_opts=9)
    # 保存元数据
    f.attrs['creation_date'] = current_date
    f.attrs['total_photons'] = len(df)
    f.attrs['signal_photons'] = int(df['is_signal'].sum())
    f.attrs['noise_photons'] = int((~df['is_signal']).sum())
    # 保存统计信息
    stats_group = f.create_group('statistics')
    stats_group.attrs['h_min'] = float(df['h'].min())
    stats_group.attrs['h_max'] = float(df['h'].max())
    stats_group.attrs['h_mean'] = float(df['h'].mean())
    stats_group.attrs['h_median'] = float(df['h'].median())
    stats_group.attrs['h_std'] = float(df['h'].std())

    # 保存源文件信息
    if 'file_source' in df.columns:
        sources = df['file_source'].unique()
        stats_group.attrs['source_files'] = ','.join(sources)
        stats_group.attrs['num_source_files'] = len(sources)

    # 保存波束信息
    if 'beam_source' in df.columns:
        beams = df['beam_source'].unique()
        stats_group.attrs['beams'] = ','.join(beams)
        stats_group.attrs['num_beams'] = len(beams)

    # 保存置信度分布
    conf_counts = df['conf'].value_counts().sort_index()
    stats_group.attrs['conf_distribution'] = str(conf_counts.to_dict())
    print(f"✓ 已保存到: {output_file}")
