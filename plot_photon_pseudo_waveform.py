import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

from explore_gedi_file import gedi_file

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_photon_pseudo_waveform(gedi_file, beam_name, fp_idx, bin_width=1.0):
    """
    绘制指定足迹的光子高程直方图和伪波形（论文标准方法）
    包含：高程直方图、三次样条拟合的伪波形、归一化伪波形、特征提取
    """
    with h5py.File(gedi_file, 'r') as f:
        if beam_name not in f:
            print(f"波束 {beam_name} 不存在")
            return
        beam = f[beam_name]
        fp_group_name = f'fp_{fp_idx:04d}'
        if fp_group_name not in beam or 'photons' not in beam[fp_group_name]:
            print(f"足迹 {fp_idx} 没有光子数据")
            return

        photons = beam[fp_group_name]['photons']
        h = photons['h'][:]
        print(f"光子数量: {len(h)}")
        print(f"高程范围: {h.min():.2f} ~ {h.max():.2f} m")

        # === 步骤1: 生成高程频数直方图 ===
        h_min, h_max = h.min(), h.max()
        bins = np.arange(h_min, h_max + bin_width, bin_width)
        hist, bin_edges = np.histogram(h, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # === 步骤2: 三次样条曲线拟合（生成连续伪波形）===
        cs = CubicSpline(bin_centers, hist, bc_type='natural')
        x_dense = np.linspace(bin_centers[0], bin_centers[-1], 500)
        waveform = cs(x_dense)
        waveform = np.maximum(waveform, 0)  # 移除负值（样条可能产生负值）

        # === 步骤3: 归一化处理 ===
        w_min, w_max = waveform.min(), waveform.max()
        if w_max > w_min:
            norm_wave = (waveform - w_min) / (w_max - w_min)
        else:
            norm_wave = waveform

        # === 步骤4: 提取波形特征 ===
        max_intensity = np.max(norm_wave)
        std_intensity = np.std(norm_wave)

        # 检测峰值
        peaks, peak_props = find_peaks(norm_wave, height=0.1, prominence=0.05)

        if len(peaks) == 0:
            fall_width = 0.0
            first_peak_x = None
            first_peak_height = 0
        else:
            first_peak_idx = peaks[0]
            first_peak_x = x_dense[first_peak_idx]
            first_peak_height = norm_wave[first_peak_idx]

            # 计算下降沿宽度：首个峰值右侧到第一个谷值的距离
            right_slice = norm_wave[first_peak_idx:]
            # 寻找谷值（极小值点）
            valleys, _ = find_peaks(-right_slice)  # 找负峰值即谷值
            if len(valleys) == 0:
                fall_width = x_dense[-1] - first_peak_x
            else:
                first_valley_idx = valleys[0]
                valley_x = x_dense[first_peak_idx + first_valley_idx]
                fall_width = valley_x - first_peak_x

        skewness = skew(norm_wave)
        kurt_val = kurtosis(norm_wave)

        # === 绘图：4个子图 ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 图1: 光子高程直方图（原始数据分布）
        ax1 = axes[0, 0]
        ax1.bar(bin_centers, hist, width=bin_width * 0.8, alpha=0.7,
                color='steelblue', edgecolor='black', label='光子频数')
        ax1.set_xlabel('高程 (m)', fontsize=12)
        ax1.set_ylabel('光子计数', fontsize=12)
        ax1.set_title(f'① 光子高程频数直方图 (bin={bin_width}m)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 图2: 三次样条拟合的伪波形（论文图5.3样式）
        ax2 = axes[0, 1]
        ax2.plot(x_dense, waveform, 'b-', linewidth=2, label='三次样条拟合伪波形')
        ax2.fill_between(x_dense, 0, waveform, alpha=0.3, color='steelblue')
        ax2.scatter(bin_centers, hist, c='red', s=20, alpha=0.6, label='原始直方图', zorder=5)
        ax2.set_xlabel('高程 (m)', fontsize=12)
        ax2.set_ylabel('光子频数', fontsize=12)
        ax2.set_title('② 三次样条曲线拟合伪波形', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 图3: 归一化伪波形 + 特征标注（论文图5.4样式）
        ax3 = axes[1, 0]
        ax3.plot(x_dense, norm_wave, 'g-', linewidth=2, label='归一化伪波形')
        ax3.fill_between(x_dense, 0, norm_wave, alpha=0.3, color='green')

        if first_peak_x is not None:
            # 标记首个峰值
            ax3.plot(first_peak_x, first_peak_height, 'ro', markersize=8,
                     label=f'首个峰值 @ {first_peak_x:.1f}m')
            # 标记下降沿宽度
            ax3.annotate(f'下降沿宽度 = {fall_width:.1f}m',
                         xy=(first_peak_x, first_peak_height),
                         xytext=(first_peak_x + fall_width / 2, first_peak_height * 0.6),
                         arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                         fontsize=10, color='orange')
            ax3.axvline(x=first_peak_x + fall_width, color='orange', linestyle='--',
                        linewidth=1.5, label=f'下降沿终点 @ {first_peak_x + fall_width:.1f}m')

        ax3.set_xlabel('高程 (m)', fontsize=12)
        ax3.set_ylabel('归一化强度', fontsize=12)
        ax3.set_title('③ 归一化伪波形及特征）', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 图4: 波形特征统计
        ax4 = axes[1, 1]
        feature_names = ['最大强度\nMax', '强度标准差\nSD', '下降沿宽度\nLR', '偏度\nSk', '峰度\nKu', '光子数\nN']
        feature_values = [max_intensity, std_intensity, fall_width, skewness, kurt_val, len(h)]

        bars = ax4.bar(feature_names, feature_values, color='coral', alpha=0.7, edgecolor='black')
        ax4.set_ylabel('数值', fontsize=12)
        ax4.set_title('④ 伪波形特征统计', fontsize=14)
        ax4.tick_params(axis='x', labelsize=10)

        for bar, val in zip(bars, feature_values):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.3f}' if val < 10 else f'{val:.1f}',
                     ha='center', va='bottom', fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'伪波形分析 - {beam_name} / footprint_{fp_idx:04d}', fontsize=16)
        plt.tight_layout()
        plt.show()

        # 输出特征到控制台
        print("\n=== 提取的伪波形特征（论文5.2.2节）===")
        print(f"  最大归一化强度 (Max):        {max_intensity:.4f}")
        print(f"  归一化强度标准差 (SD):       {std_intensity:.4f}")
        print(f"  首个峰值下降沿宽度 (LR):     {fall_width:.2f} m")
        print(f"  偏度 (Sk):                   {skewness:.4f}")
        print(f"  峰度 (Ku):                   {kurt_val:.4f}")
        print(f"  光子数:                      {len(h)}")

        return {
            'max_intensity': max_intensity,
            'std_intensity': std_intensity,
            'fall_width': fall_width,
            'skewness': skewness,
            'kurtosis': kurt_val,
            'photon_count': len(h)
        }


# 使用示例
if __name__ == "__main__":
    gedi_file = r"D:\研究生\SanFrancisco\GEDIdata\merged_gedi_data_20260411_with_photons_15m20260427.h5"
    #gedi_file=r"D:\研究生\SanFrancisco\GEDIdata\merged_gedi_data_20260411_with_photons_20260427.h5"
    # 绘制有效足迹（如 BEAM0000 的 fp_0006）
    features = plot_photon_pseudo_waveform(
        gedi_file=gedi_file,
        beam_name='BEAM0000',
        fp_idx=38,
        bin_width=1.0
    )