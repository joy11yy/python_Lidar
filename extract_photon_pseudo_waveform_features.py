import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm


def extract_photon_pseudo_waveform_features(gedi_file, beam_name, fp_idx,
                                            bin_width=1.0, min_photons=30):
    """
    提取 ICESat-2 光子的伪波形特征（高程分布）

    参数:
        gedi_file: 融合后的 H5 文件路径
        beam_name: 波束名称
        fp_idx: 足迹索引
        bin_width: 高程直方图 bin 宽度（米），默认 1m
        min_photons: 最少光子数，少于该值则返回 None
    返回:
        dict: 伪波形特征，若光子不足或发生错误返回 None
    """
    with h5py.File(gedi_file, 'r') as f:
        if beam_name not in f:
            return None
        beam = f[beam_name]
        fp_group_name = f'fp_{fp_idx:04d}'
        if fp_group_name not in beam or 'photons' not in beam[fp_group_name]:
            return None
        photons = beam[fp_group_name]['photons']
        h = photons['h'][:]  # 光子高程数组
        if len(h) < min_photons:
            return None

        # 生成高程直方图
        h_min, h_max = h.min(), h.max()
        bins = np.arange(h_min, h_max + bin_width, bin_width)
        hist, bin_edges = np.histogram(h, bins=bins)
        # 使用 bin 中心作为横坐标
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 三次样条插值，获得连续伪波形
        # 如果点数太少，直接用 hist 进行线性插值
        if len(hist) < 4:
            # 简单线性插值到更密的点
            x_dense = np.linspace(bin_centers[0], bin_centers[-1], 100)
            from scipy.interpolate import interp1d
            f_interp = interp1d(bin_centers, hist, kind='linear', fill_value=0, bounds_error=False)
            waveform = f_interp(x_dense)
            x_axis = x_dense
        else:
            cs = CubicSpline(bin_centers, hist, bc_type='natural')
            x_axis = np.linspace(bin_centers[0], bin_centers[-1], 200)
            waveform = cs(x_axis)
            waveform = np.maximum(waveform, 0)  # 移除负值

        # 归一化
        w_min, w_max = waveform.min(), waveform.max()
        if w_max == w_min:
            return None
        norm_wave = (waveform - w_min) / (w_max - w_min)

        # 提取特征
        # 1. 最大归一化强度
        max_intensity = np.max(norm_wave)
        # 2. 归一化强度标准差
        std_intensity = np.std(norm_wave)

        # 3. 首个强度峰值的下降沿宽度
        # 找到第一个峰值位置
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(norm_wave, height=0.1)
        if len(peaks) == 0:
            fall_width = 0.0
        else:
            first_peak_idx = peaks[0]
            # 寻找峰值右侧的第一个谷值（极小值）
            right_slice = norm_wave[first_peak_idx:]
            # 使用 find_peaks 找负方向？简单方法：找第一个低于左右两侧的点
            valley_idx = None
            for i in range(1, len(right_slice) - 1):
                if right_slice[i] <= right_slice[i - 1] and right_slice[i] <= right_slice[i + 1]:
                    valley_idx = i
                    break
            if valley_idx is None:
                # 没有明显谷值，使用波形终点
                fall_width = (x_axis[-1] - x_axis[first_peak_idx])
            else:
                fall_width = x_axis[first_peak_idx + valley_idx] - x_axis[first_peak_idx]

        # 4. 偏度
        skewness = skew(norm_wave)
        # 5. 峰度
        kurt = kurtosis(norm_wave)

        features = {
            'max_intensity': max_intensity,
            'std_intensity': std_intensity,
            'fall_width': fall_width,
            'skewness': skewness,
            'kurtosis': kurt,
            'photon_count': len(h)
        }
        return features

