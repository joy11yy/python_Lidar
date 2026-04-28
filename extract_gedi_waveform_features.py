import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm

# 导入已有的波形分解函数
from waveresolve import waveresolve  # 假设 waveresolve.py 在同一目录


def extract_gedi_waveform_features(gedi_file, beam_name, fp_idx, verbose=False):
    """
    从融合后的 GEDI H5 文件中提取指定足迹的波形特征（高斯分解）

    参数:
        gedi_file: 融合后的 H5 文件路径
        beam_name: 波束名称，如 'BEAM0000'
        fp_idx: 足迹索引（整数）
        verbose: 是否打印信息
    返回:
        dict: 波形特征，若失败返回 None
        提取的gedi波形包括高斯峰个数、最大峰值幅值、最小峰值幅值、平均脉宽、总能量
    """
    with h5py.File(gedi_file, 'r') as f:
        if beam_name not in f:
            return None
        beam = f[beam_name]
        # 波形数据存储在 wavedata/rxwaveform 中，每个足迹为一个变长数组
        if 'wavedata' not in beam or 'rxwaveform' not in beam['wavedata']:
            return None
        rxwaveforms = beam['wavedata']['rxwaveform']
        if fp_idx >= len(rxwaveforms):
            return None
        rxwave = rxwaveforms[fp_idx][:]  # 一维数组
        # 采样点 x 坐标（取索引）
        x = np.arange(1, len(rxwave) + 1)

        # 调用波形分解函数
        try:
            prfnl, prini = waveresolve(
                rx=x,
                ry=rxwave,
                filtwidth=4,  # 可根据数据调整
                noise_sigma=np.std(rxwave[:100]),  # 用前100点估计噪声
                txsigma=4,
                maxwavenum=6,
                display=0
            )
        except Exception as e:
            if verbose:
                print(f"波形分解失败: {e}")
            return None

        if len(prfnl) == 0:
            # 没有分解出有效高斯分量
            features = {
                'n_peaks': 0,
                'max_amplitude': 0.0,
                'min_amplitude': 0.0,
                'mean_width': 0.0,
                'total_energy': 0.0
            }
        else:
            amplitudes = prfnl[:, 0]
            widths = prfnl[:, 2]
            energies = amplitudes * widths
            features = {
                'n_peaks': len(prfnl),
                'max_amplitude': np.max(amplitudes),
                'min_amplitude': np.min(amplitudes),
                'mean_width': np.mean(widths),
                'total_energy': np.sum(energies)
            }
        return features

