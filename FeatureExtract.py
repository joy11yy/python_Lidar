import h5py
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import warnings
import waveresolve

warnings.filterwarnings('ignore')


def extract_waveform_features(h5_file_path, beam_name='BEAM1019', max_waveforms=None):
    """
    从GEDI匹配后的HDF5文件中提取波形特征

    参数:
    - h5_file_path: HDF5文件路径
    - beam_name: 波束名称
    - max_waveforms: 最大处理的波形数量（用于测试）
    """

    # 打开HDF5文件
    with h5py.File(h5_file_path, 'a') as f:  # 使用'a'模式以允许写入
        # 获取波形数据
        rxwaveform = f[f'{beam_name}/rxwaveform'][:]  # 接收波形
        txwaveform = f[f'{beam_name}/txwaveform'][:]  # 发射波形

        # 获取其他辅助数据
        quality_flag = f[f'{beam_name}/quality_flag'][:]
        degrade_flag = f[f'{beam_name}/degrade_flag'][:]
        shot_number = f[f'{beam_name}/shot_number'][:]

        # 波形采样点
        sample_points = np.arange(1, rxwaveform.shape[1] + 1)

        # 限制处理数量（用于测试）
        n_shots = len(rxwaveform)
        if max_waveforms and max_waveforms < n_shots:
            n_shots = max_waveforms
            print(f"限制处理波形数量: {n_shots}")

        # 初始化特征存储
        features = {
            'shot_number': shot_number[:n_shots],
            'n_peaks': np.zeros(n_shots, dtype=int),
            'max_peak_amplitude': np.zeros(n_shots, dtype=float),
            'min_peak_amplitude': np.zeros(n_shots, dtype=float),
            'mean_peak_amplitude': np.zeros(n_shots, dtype=float),
            'std_peak_amplitude': np.zeros(n_shots, dtype=float),
            'total_energy': np.zeros(n_shots, dtype=float),
            'max_peak_width': np.zeros(n_shots, dtype=float),
            'min_peak_width': np.zeros(n_shots, dtype=float),
            'mean_peak_width': np.zeros(n_shots, dtype=float),
            'max_peak_position': np.zeros(n_shots, dtype=float),
            'min_peak_position': np.zeros(n_shots, dtype=float),
            'first_peak_position': np.zeros(n_shots, dtype=float),
            'last_peak_position': np.zeros(n_shots, dtype=float),
            'waveform_length': np.zeros(n_shots, dtype=float),
            'waveform_snr': np.zeros(n_shots, dtype=float),
            'decomposition_success': np.zeros(n_shots, dtype=int),
            'quality_flag': quality_flag[:n_shots],
            'degrade_flag': degrade_flag[:n_shots]
        }

        # 处理每个波形
        print(f"开始处理 {n_shots} 个波形...")
        for i in range(n_shots):
            if i % 1000 == 0:
                print(f"处理进度: {i}/{n_shots}")

            try:
                # 获取当前波形
                rx = rxwaveform[i]
                tx = txwaveform[i] if i < len(txwaveform) else None

                # 计算噪声水平（使用波形前10个点和后10个点）
                noise_samples = np.concatenate([rx[:10], rx[-10:]])
                noise_sigma = np.std(noise_samples)

                # 波形能量
                features['total_energy'][i] = np.sum(rx)

                # 波形长度（信号范围）
                signal_mask = rx > noise_sigma * 3
                if np.any(signal_mask):
                    signal_indices = np.where(signal_mask)[0]
                    features['waveform_length'][i] = signal_indices[-1] - signal_indices[0]
                else:
                    features['waveform_length'][i] = 0

                # 计算信噪比
                signal_power = np.mean(rx[rx > noise_sigma * 3] ** 2) if np.any(rx > noise_sigma * 3) else 0
                noise_power = noise_sigma ** 2
                features['waveform_snr'][i] = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0

                # 跳过质量差的波形
                if quality_flag[i] != 1 or degrade_flag[i] != 0:
                    features['decomposition_success'][i] = -1
                    continue

                # 调用波形分解函数
                # 注意：需要根据您的数据调整参数
                try:
                    prfnl, prini = waveresolve(
                        rx=sample_points,
                        ry=rx,
                        filtwidth=4,  # 滤波宽度，可能需要调整
                        noise_sigma=noise_sigma,
                        txsigma=2,  # 发射波形sigma，可能需要调整
                        maxwavenum=6,  # 最大高斯分量数
                        display=0  # 不显示图形
                    )

                    # 检查是否成功分解出高斯分量
                    if len(prfnl) > 0:
                        features['decomposition_success'][i] = 1

                        # 提取峰值特征
                        amplitudes = prfnl[:, 0]  # 幅值
                        positions = prfnl[:, 1]  # 位置
                        widths = prfnl[:, 2]  # 脉宽

                        features['n_peaks'][i] = len(amplitudes)
                        features['max_peak_amplitude'][i] = np.max(amplitudes)
                        features['min_peak_amplitude'][i] = np.min(amplitudes)
                        features['mean_peak_amplitude'][i] = np.mean(amplitudes)
                        features['std_peak_amplitude'][i] = np.std(amplitudes) if len(amplitudes) > 1 else 0

                        features['max_peak_width'][i] = np.max(widths)
                        features['min_peak_width'][i] = np.min(widths)
                        features['mean_peak_width'][i] = np.mean(widths)

                        features['max_peak_position'][i] = np.max(positions)
                        features['min_peak_position'][i] = np.min(positions)

                        # 按位置排序后获取第一个和最后一个峰值
                        sorted_indices = np.argsort(positions)
                        features['first_peak_position'][i] = positions[sorted_indices[0]]
                        features['last_peak_position'][i] = positions[sorted_indices[-1]]

                    else:
                        features['decomposition_success'][i] = 0

                except Exception as e:
                    print(f"波形 {i} 分解失败: {str(e)}")
                    features['decomposition_success'][i] = -2

            except Exception as e:
                print(f"处理波形 {i} 时出错: {str(e)}")
                continue

        # 将特征转换为DataFrame以便查看
        df_features = pd.DataFrame(features)
        print("\n特征统计：")
        print(df_features.describe())

        # 创建新组来存储特征
        feature_group_name = f'{beam_name}/waveform_features'
        if feature_group_name in f:
            del f[feature_group_name]
        feature_group = f.create_group(feature_group_name)

        # 将每个特征保存为单独的dataset
        for feature_name, feature_data in features.items():
            feature_group.create_dataset(feature_name, data=feature_data, compression='gzip')

        print(f"\n特征已保存到: {feature_group_name}")

        # 可选：添加特征描述作为属性
        feature_descriptions = {
            'shot_number': 'GEDI shot number',
            'n_peaks': 'Number of detected Gaussian peaks',
            'max_peak_amplitude': 'Maximum amplitude among peaks',
            'min_peak_amplitude': 'Minimum amplitude among peaks',
            'mean_peak_amplitude': 'Mean amplitude of peaks',
            'std_peak_amplitude': 'Standard deviation of peak amplitudes',
            'total_energy': 'Total waveform energy (sum of amplitudes)',
            'max_peak_width': 'Maximum peak width (sigma)',
            'min_peak_width': 'Minimum peak width (sigma)',
            'mean_peak_width': 'Mean peak width',
            'max_peak_position': 'Maximum peak position (bin number)',
            'min_peak_position': 'Minimum peak position (bin number)',
            'first_peak_position': 'Position of the first peak',
            'last_peak_position': 'Position of the last peak',
            'waveform_length': 'Length of signal extent (bins)',
            'waveform_snr': 'Waveform Signal-to-Noise Ratio (dB)',
            'decomposition_success': '1=success, 0=no peaks, -1=low quality, -2=decomposition error',
            'quality_flag': 'Original GEDI quality flag',
            'degrade_flag': 'Original GEDI degrade flag'
        }

        for feat_name, desc in feature_descriptions.items():
            if feat_name in feature_group:
                feature_group[feat_name].attrs['description'] = desc

        return df_features


def add_features_to_original_h5(original_h5_path, features_df, beam_name='BEAM1019'):
    """
    将特征添加到原始的HDF5文件中（另一种方法，直接修改原文件）
    """
    with h5py.File(original_h5_path, 'a') as f:
        # 检查是否存在特征组
        feature_group_name = f'{beam_name}/waveform_features'
        if feature_group_name in f:
            print(f"警告: {feature_group_name} 已存在，将被覆盖")
            del f[feature_group_name]

        # 创建特征组
        feature_group = f.create_group(feature_group_name)

        # 添加每个特征
        for column in features_df.columns:
            feature_group.create_dataset(column, data=features_df[column].values, compression='gzip')

        print(f"特征已添加到 {feature_group_name}")


# 使用示例
if __name__ == "__main__":
    h5_file = r"D:\研究生\SanFrancisco\GEDI_Matched_Compact_SF_Land30.h5"

    # 提取特征（可以先测试少量波形）
    features_df = extract_waveform_features(
        h5_file_path=h5_file,
        beam_name='BEAM1019',
        max_waveforms=1000  # 先测试1000个波形
    )

    # 如果需要保存特征的单独副本
    features_df.to_csv('waveform_features.csv', index=False)
    print("\n特征已保存到 waveform_features.csv")

    # 查看成功分解的波形统计
    success_mask = features_df['decomposition_success'] == 1
    print(f"\n成功分解的波形数量: {success_mask.sum()}")
    if success_mask.any():
        print("\n成功分解波形的峰值统计:")
        print(features_df[success_mask][['n_peaks', 'max_peak_amplitude', 'mean_peak_width']].describe())



        