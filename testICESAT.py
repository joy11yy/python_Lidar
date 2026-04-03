import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_ATL03_data(filepath, beam='/gt2r'):
    """
    读取ATL03数据，用signal_conf_ph去噪
    """
    with h5py.File(filepath, 'r') as f:
        print(f"文件中的波束: {list(f.keys())}")

        # 自动选择强波束
        if beam not in f:
            strong_beams = [k for k in f.keys() if k.startswith('gt') and k.endswith('r')]
            if strong_beams:
                beam = strong_beams[0]
            else:
                gt_beams = [k for k in f.keys() if k.startswith('gt')]
                beam = gt_beams[0] if gt_beams else beam
            print(f"改用波束: {beam}")

        heights = f[beam]['heights']

        # 读取基础数据
        h_ph = np.squeeze(heights['h_ph'][:])
        dist_ph_along = np.squeeze(heights['dist_ph_along'][:])
        lat_ph = np.squeeze(heights['lat_ph'][:])
        lon_ph = np.squeeze(heights['lon_ph'][:])

        n_photons = len(h_ph)
        print(f"  光子总数: {n_photons}")

        # 读取置信度（5维，取第0列）
        if 'signal_conf_ph' in heights:
            conf_raw = heights['signal_conf_ph'][:]
            conf_raw = np.squeeze(conf_raw)
            if conf_raw.ndim == 2:
                conf = conf_raw[:, 0]  # 取地表类型置信度
            else:
                conf = conf_raw
            print(f"  conf shape: {conf.shape}")
        else:
            conf = np.ones(n_photons) * 2
            print("  未找到signal_conf_ph，使用默认值2")

        # 确保长度匹配
        if len(conf) != n_photons:
            print(f"  警告: conf长度({len(conf)}) != 光子数({n_photons})")
            if len(conf) > n_photons:
                conf = conf[:n_photons]
            else:
                conf = np.resize(conf, n_photons)

        # 去除无效高程
        valid_mask = (h_ph > -900) & ~np.isnan(h_ph)

        # 去噪：只用 conf >= 2 的光子（中等以上置信度）
        signal_mask = valid_mask & (conf >= 2)

        df = pd.DataFrame({
            'h': h_ph[signal_mask],
            'dist': dist_ph_along[signal_mask],
            'lat': lat_ph[signal_mask],
            'lon': lon_ph[signal_mask],
            'conf': conf[signal_mask],
        })

        print(f"  去噪后光子数: {len(df)} ({100 * len(df) / n_photons:.1f}%)")

        return df, beam


def explore_all_beams(filepath):
    """探索所有波束的光子数量"""
    with h5py.File(filepath, 'r') as f:
        beams = [k for k in f.keys() if k.startswith('gt')]
        print("=" * 60)
        print("各波束光子数量统计:")
        print("-" * 60)

        beam_stats = []
        for beam in beams:
            heights = f[beam]['heights']
            h_ph = np.squeeze(heights['h_ph'][:])
            n_photons = len(h_ph)
            valid = (h_ph > -900) & ~np.isnan(h_ph)

            # 获取置信度信息
            if 'signal_conf_ph' in heights:
                conf_raw = heights['signal_conf_ph'][:]
                conf_raw = np.squeeze(conf_raw)
                if conf_raw.ndim == 2:
                    conf = conf_raw[:, 0]
                else:
                    conf = conf_raw
                if len(conf) != n_photons:
                    conf_status = f"维度不匹配({len(conf)})"
                else:
                    high_conf = (conf >= 3).sum()
                    conf_status = f"高置信度(>=3): {high_conf}"
            else:
                conf_status = "无置信度信息"

            beam_stats.append({
                'beam': beam,
                'total': n_photons,
                'valid': valid.sum(),
                'conf_status': conf_status
            })
            print(f"  {beam}: 总光子={n_photons}, 有效={valid.sum()}, {conf_status}")

        print("=" * 60)
        return beam_stats


def denoise_by_local_density(df, segment_length=100, radius_dist=25, radius_h=2, density_percentile=40):
    """
    基于局部密度的去噪算法（修复版）
    """
    df = df.copy()
    df['segment'] = (df['dist'] // segment_length).astype(int)
    df['is_signal'] = False

    for seg_id in df['segment'].unique():
        seg_mask = df['segment'] == seg_id
        seg_h = df.loc[seg_mask, 'h'].values
        seg_dist = df.loc[seg_mask, 'dist'].values
        seg_idx = np.where(seg_mask)[0]

        if len(seg_h) < 10:
            continue

        # 高程直方图粗去噪
        mu = np.mean(seg_h)
        sigma = np.std(seg_h)
        coarse_mask = (seg_h >= mu - 2 * sigma) & (seg_h <= mu + 2 * sigma)

        if coarse_mask.sum() < 5:
            continue

        seg_h_filtered = seg_h[coarse_mask]
        seg_dist_filtered = seg_dist[coarse_mask]
        seg_idx_filtered = seg_idx[coarse_mask]

        # 计算每个光子的局部密度
        densities = []
        for i in range(len(seg_h_filtered)):
            # 计算与其他光子的距离
            dist_diff = np.abs(seg_dist_filtered - seg_dist_filtered[i])
            h_diff = np.abs(seg_h_filtered - seg_h_filtered[i])
            neighbors = (dist_diff <= radius_dist) & (h_diff <= radius_h)
            densities.append(np.sum(neighbors))

        densities = np.array(densities)

        # 密度阈值
        if len(densities) > 0:
            density_threshold = np.percentile(densities, density_percentile)
            fine_mask = densities >= density_threshold

            signal_idx = seg_idx_filtered[fine_mask]
            df.loc[signal_idx, 'is_signal'] = True

    return df


def extract_ground_canopy(df, segment_length=50, ground_percentile=20):
    """
    提取地面和冠层光子
    """
    df = df.copy()
    df['segment'] = (df['dist'] // segment_length).astype(int)
    df['type'] = 'noise'

    for seg_id in df['segment'].unique():
        seg_mask = (df['segment'] == seg_id) & (df['is_signal'])
        seg_h = df.loc[seg_mask, 'h'].values
        seg_idx = np.where(seg_mask)[0]

        if len(seg_h) < 5:
            continue

        # 按高程排序，取最低的N%作为地面光子
        h_sorted_idx = np.argsort(seg_h)
        n_ground = max(1, int(len(seg_h) * ground_percentile / 100))
        ground_idx_local = h_sorted_idx[:n_ground]
        canopy_idx_local = h_sorted_idx[n_ground:]

        ground_idx_global = seg_idx[ground_idx_local]
        canopy_idx_global = seg_idx[canopy_idx_local]

        df.loc[ground_idx_global, 'type'] = 'ground'
        df.loc[canopy_idx_global, 'type'] = 'canopy'

    return df


def compute_features(df, window_size=50):
    """
    为每个光子提取特征（用于后续分类）
    """
    df = df.copy()
    df['rel_h'] = 0.0  # 相对高程
    df['density'] = 0  # 局部密度
    df['vertical_range'] = 0  # 垂直范围

    df['window'] = (df['dist'] // window_size).astype(int)

    for win_id in df['window'].unique():
        win_mask = df['window'] == win_id
        win_h = df.loc[win_mask, 'h'].values
        win_dist = df.loc[win_mask, 'dist'].values
        win_idx = np.where(win_mask)[0]

        if len(win_h) < 3:
            continue

        # 局部最低点作为地面参考
        ground_h = np.percentile(win_h, 10)

        # 垂直范围
        vertical_range = win_h.max() - win_h.min()

        # 计算每个光子的特征
        for i, (h_val, d_val, idx) in enumerate(zip(win_h, win_dist, win_idx)):
            # 相对高程
            df.loc[idx, 'rel_h'] = h_val - ground_h

            # 垂直范围
            df.loc[idx, 'vertical_range'] = vertical_range

            # 局部密度（半径10m内光子数）
            neighbors = (np.abs(win_dist - d_val) <= 10) & (np.abs(win_h - h_val) <= 2)
            df.loc[idx, 'density'] = np.sum(neighbors)

    return df


def plot_results(df, beam_name):
    """
    可视化结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 原始数据按置信度着色
    ax1 = axes[0, 0]
    colors = {0: 'gray', 1: 'lightgray', 2: 'blue', 3: 'green', 4: 'red'}
    for conf_val, color in colors.items():
        mask = df['conf'] == conf_val
        if mask.sum() > 0:
            ax1.scatter(df.loc[mask, 'dist'], df.loc[mask, 'h'],
                        c=color, s=1, alpha=0.5)
    ax1.set_xlabel('沿轨距离 (m)')
    ax1.set_ylabel('高程 (m)')
    ax1.set_title(f'{beam_name} - 按官方置信度')
    ax1.legend(['噪声(0)', '低(1)', '中(2)', '高(3)', '最高(4)'], loc='upper right', fontsize=8)

    # 2. 去噪结果
    ax2 = axes[0, 1]
    if 'is_signal' in df.columns:
        signal = df[df['is_signal']]
        noise = df[~df['is_signal']]
        ax2.scatter(noise['dist'], noise['h'], c='lightgray', s=1, alpha=0.3, label='噪声')
        ax2.scatter(signal['dist'], signal['h'], c='green', s=1, alpha=0.5, label='信号')
        ax2.set_title(f'{beam_name} - 去噪结果')
        ax2.legend()
    else:
        ax2.scatter(df['dist'], df['h'], c='blue', s=1, alpha=0.5)
        ax2.set_title(f'{beam_name} - 信号光子')
    ax2.set_xlabel('沿轨距离 (m)')
    ax2.set_ylabel('高程 (m)')

    # 3. 地面/冠层分类
    ax3 = axes[1, 0]
    if 'type' in df.columns:
        ground = df[df['type'] == 'ground']
        canopy = df[df['type'] == 'canopy']
        noise = df[df['type'] == 'noise']
        ax3.scatter(noise['dist'], noise['h'], c='lightgray', s=1, alpha=0.3, label='噪声')
        ax3.scatter(ground['dist'], ground['h'], c='brown', s=1, alpha=0.5, label='地面')
        ax3.scatter(canopy['dist'], canopy['h'], c='darkgreen', s=1, alpha=0.5, label='冠层')
        ax3.set_title(f'{beam_name} - 地面/冠层分类')
        ax3.legend()
    else:
        ax3.scatter(df['dist'], df['h'], c='blue', s=1, alpha=0.5)
        ax3.set_title(f'{beam_name} - 光子分布')
    ax3.set_xlabel('沿轨距离 (m)')
    ax3.set_ylabel('高程 (m)')

    # 4. 高程分布
    ax4 = axes[1, 1]
    ax4.hist(df['h'], bins=100, alpha=0.5, label='所有光子', color='blue')
    if 'is_signal' in df.columns:
        signal = df[df['is_signal']]
        if len(signal) > 0:
            ax4.hist(signal['h'], bins=100, alpha=0.5, label='信号光子', color='green')
    if 'type' in df.columns:
        ground = df[df['type'] == 'ground']
        canopy = df[df['type'] == 'canopy']
        if len(ground) > 0:
            ax4.hist(ground['h'], bins=100, alpha=0.5, label='地面光子', color='brown')
        if len(canopy) > 0:
            ax4.hist(canopy['h'], bins=100, alpha=0.5, label='冠层光子', color='darkgreen')
    ax4.set_xlabel('高程 (m)')
    ax4.set_ylabel('光子数量')
    ax4.set_title('高程分布')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'{beam_name}_analysis.png', dpi=150)
    plt.show()
    print(f"图片已保存: {beam_name}_analysis.png")


def main():
    filepath = r"D:\研究生\ICESAT\ATL03_20251110081832_08672906_007_01_subsetted.h5"

    print("步骤1: 探索所有波束")
    beam_stats = explore_all_beams(filepath)

    print("\n步骤2: 加载数据")
    # 选择有效光子最多的强波束
    best_beam = None
    best_count = 0
    for stat in beam_stats:
        if stat['beam'].endswith('r') and stat['valid'] > best_count:
            best_beam = stat['beam']
            best_count = stat['valid']

    if best_beam is None:
        best_beam = beam_stats[0]['beam']

    print(f"选择波束: {best_beam}")
    df, beam_name = load_ATL03_data(filepath, beam=best_beam)

    print("\n步骤3: 局部密度去噪")
    df = denoise_by_local_density(df, segment_length=100, radius_dist=25, radius_h=2, density_percentile=40)
    signal_cnt = df['is_signal'].sum()
    print(f"  信号光子: {signal_cnt} / {len(df)} ({100 * signal_cnt / len(df):.1f}%)")

    print("\n步骤4: 地面/冠层提取")
    df = extract_ground_canopy(df, segment_length=50, ground_percentile=20)
    ground_cnt = (df['type'] == 'ground').sum()
    canopy_cnt = (df['type'] == 'canopy').sum()
    print(f"  地面光子: {ground_cnt}")
    print(f"  冠层光子: {canopy_cnt}")

    print("\n步骤5: 特征提取")
    df = compute_features(df, window_size=50)
    print(f"  特征: 相对高程, 局部密度, 垂直范围")

    print("\n步骤6: 统计分析")
    print(f"  高程范围: {df['h'].min():.2f} ~ {df['h'].max():.2f} m")
    print(f"  沿轨距离范围: {df['dist'].min():.2f} ~ {df['dist'].max():.2f} m")

    print(f"\n  置信度分布:")
    for conf_val in sorted(df['conf'].unique()):
        count = (df['conf'] == conf_val).sum()
        print(f"    conf={conf_val}: {count} 光子 ({100 * count / len(df):.1f}%)")

    print("\n步骤7: 可视化")
    plot_results(df, beam_name)

    # 保存结果到CSV
    output_csv = f'{beam_name}_processed.csv'
    df.to_csv(output_csv, index=False)
    print(f"数据已保存: {output_csv}")

    return df


if __name__ == '__main__':
    df = main()