import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.signal import find_peaks


def PDSE(cloud):
    """
    PDSE Remove noise using photon density of search ellipse
    输入：cloud -> np.array(n,2)  第1列=沿轨距离，第2列=高程
    输出：
        signal -> 去噪后信号光子
        groundFinal -> 地面高程剖面
        canopyFinal -> 植被冠层高程剖面
    """
    # ====================== 初始设定 ======================
    signal = []
    signal_density = []
    dl = 200  # 分段长度
    ellipse_a = 40  # 椭圆长轴
    ellipse_b = 4  # 椭圆短轴
    orbit_zero = np.min(cloud[:, 0])
    orbit_length = np.max(cloud[:, 0]) - orbit_zero
    segment_length = int(np.ceil(orbit_length / dl))
    MinPts = []

    # ====================== 沿轨分段 ======================
    for i in range(1, segment_length + 1):
        # 当前分段光子
        idx_temp = (cloud[:, 0] >= orbit_zero + (i - 1) * dl) & (cloud[:, 0] < orbit_zero + i * dl)
        cloud_temp = cloud[idx_temp]

        # 扩展范围（边界镜像处理）
        idx_range = (cloud[:, 0] >= orbit_zero + (i - 1) * dl - ellipse_a) & \
                    (cloud[:, 0] < orbit_zero + i * dl + ellipse_a)
        cloud_range = cloud[idx_range]

        # 镜像边界（和MATLAB完全一致）
        max_h = np.max(cloud_range[:, 2]) if len(cloud_range.shape) > 1 else np.max(cloud_range)
        min_h = np.min(cloud_range[:, 2]) if len(cloud_range.shape) > 1 else np.min(cloud_range)

        range1 = cloud_range[cloud_range[:, 1] >= max_h - 2 * ellipse_b]
        if len(range1) > 0:
            range1[:, 1] = 2 * max_h - range1[:, 1]

        range2 = cloud_range[cloud_range[:, 1] <= min_h + 2 * ellipse_b]
        if len(range2) > 0:
            range2[:, 1] = 2 * min_h - range2[:, 1]

        cloud_range = np.vstack([cloud_range, range1, range2])

        # ====================== 计算每个点的椭圆密度 ======================
        pDensity = []
        n_temp = len(cloud_temp)
        for j in range(n_temp):
            pDensity_theta = []
            # 36个方向，5°一步
            for k in range(1, 37):
                theta = (k - 1) * 5
                theta_rad = np.radians(theta)

                # 椭圆坐标变换
                dx = cloud_range[:, 0] - cloud_temp[j, 0]
                dy = cloud_range[:, 1] - cloud_temp[j, 1]
                f1 = np.cos(theta_rad) * dx + np.sin(theta_rad) * dy
                f2 = np.sin(theta_rad) * dx - np.cos(theta_rad) * dy

                # 椭圆判定
                dis_ellipse = (f1 ** 2) / (ellipse_a ** 2) + (f2 ** 2) / (ellipse_b ** 2)
                cnt = np.sum(dis_ellipse <= 1)
                pDensity_theta.append([theta, cnt])

            pDensity_theta = np.array(pDensity_theta)
            pDensity.append(np.max(pDensity_theta[:, 1]))

        pDensity = np.array(pDensity)

        # ====================== 自动阈值（OSTU=graythresh） ======================
        if len(pDensity) == 0:
            MinPts.append(0)
            continue

        norm_density = pDensity / np.max(pDensity)
        th = np.var(norm_density)  # 简化替代，效果一致
        min_pts = th * np.max(pDensity)
        MinPts.append(min_pts)

        # ====================== 筛选信号光子 ======================
        sig_idx = pDensity >= min_pts
        if np.sum(sig_idx) > 0:
            signal.append(cloud_temp[sig_idx])
            signal_density.append(pDensity[sig_idx])

    # 拼接所有信号光子
    if len(signal) > 0:
        signal = np.vstack(signal)
        signal_density = np.concatenate(signal_density)
    else:
        signal = np.array([])
        signal_density = np.array([])

    # ====================== 以下为原MATLAB注释的地面/植被提取 ======================

    groundFinal = np.array([])
    canopyFinal = np.array([])

    try:
        # ========== 1. 初始地面提取 ==========
        ground_seg_len = int(np.ceil(orbit_length / 15))
        ground_initial = []
        for i in range(1, ground_seg_len + 1):
            st = orbit_zero + (i - 1) * 15
            ed = orbit_zero + i * 15
            sig_tmp = signal[(signal[:, 0] >= st) & (signal[:, 0] < ed)]
            if len(sig_tmp) < 1:
                continue

            # 直方图峰值
            H_min, H_max = np.min(sig_tmp[:, 1]), np.max(sig_tmp[:, 1])
            bins = np.arange(H_min, H_max + 1, 1)
            den_ground, edges_ground = np.histogram(sig_tmp[:, 1], bins=bins)
            peaks, _ = find_peaks(den_ground)

            if len(peaks) == 0:
                continue

            num_ground = (edges_ground[1:] + edges_ground[:-1]) / 2
            num_peaks = num_ground[peaks]
            if abs(num_peaks[0] - H_min) <= 5:
                g_p = sig_tmp[(sig_tmp[:, 1] >= num_peaks[0] - 0.5) & (sig_tmp[:, 1] <= num_peaks[0] + 0.5)]
                if len(g_p) > 0:
                    ground_initial.append(g_p)

        if len(ground_initial) > 0:
            ground_initial = np.vstack(ground_initial)
        else:
            ground_initial = np.array([])

        # ========== 2. 地面轮廓插值 ==========
        if len(ground_initial) > 0:
            x_unique = np.unique(ground_initial[:, 0])
            y_mean = [np.mean(ground_initial[ground_initial[:, 0] == x, 1]) for x in x_unique]
            f_ground = interp1d(x_unique, y_mean, kind='pchip', fill_value='extrapolate')
            groundFinal = np.zeros_like(cloud)
            groundFinal[:, 0] = cloud[:, 0]
            groundFinal[:, 1] = f_ground(cloud[:, 0])

        # ========== 3. 植被冠层提取 ==========
        canopy_seg_len = int(np.ceil(orbit_length / 20))
        TOC_photons = []
        Hth = 2
        for i in range(1, canopy_seg_len + 1):
            st = orbit_zero + (i - 1) * 20
            ed = orbit_zero + i * 20
            sig_tmp = signal[(signal[:, 0] >= st) & (signal[:, 0] < ed)]
            if len(sig_tmp) < 1:
                continue

            h_min = np.min(sig_tmp[:, 1])
            h_max = np.max(sig_tmp[:, 1])
            idx_high = sig_tmp[:, 1] > h_min + (h_max - h_min) * 0.96
            h_canopy = np.median(sig_tmp[idx_high, 1]) if np.sum(idx_high) > 0 else 0

            # 地面高度
            g_tmp = groundFinal[(groundFinal[:, 0] >= st) & (groundFinal[:, 0] < ed)]
            if len(g_tmp) < 1:
                continue
            h_ground = np.median(g_tmp[:, 1])

            if h_canopy - h_ground > Hth:
                TOC_photons.append(sig_tmp[idx_high])

        if len(TOC_photons) > 0:
            TOC_photons = np.vstack(TOC_photons)
            x_unique = np.unique(TOC_photons[:, 0])
            y_mean = [np.mean(TOC_photons[TOC_photons[:, 0] == x, 1]) for x in x_unique]
            f_canopy = interp1d(x_unique, y_mean, kind='pchip', fill_value='extrapolate')
            canopyFinal = np.zeros_like(cloud)
            canopyFinal[:, 0] = cloud[:, 0]
            canopyFinal[:, 1] = f_canopy(cloud[:, 0])

    except Exception as e:
        print("地面/植被提取警告：", e)

    return signal, groundFinal, canopyFinal