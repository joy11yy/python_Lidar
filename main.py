import numpy as np

from Match_GEDI_Land30 import match_gedi_to_landcover,match_gedi_to_landcover_multi
from ReadData import ReadGEDI_L1B
from Save_Match_Data import save_matched_data
from waveform_read import print_data_summary, draw_wave
import os
from waveresolve import waveresolve
import matplotlib.pyplot as plt
from Load_filtered_data import load_filtered_gedi_data,draw_wave,print_data_summary
import matplotlib
import Save_Match_Data
import Match_GEDI_Land30
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
if __name__ == "__main__":
    # 直接读取筛选后的文件
    gedi_filtered_file = r"D:\研究生\SanFrancisco\GEDIdata\GEDI_filtered_2025032182236_O34785_02_T02894_02_006_02_V002.h5"

    if os.path.exists(gedi_filtered_file):
        print("开始读取筛选后的GEDI数据...")
        GEDIdata = load_filtered_gedi_data(gedi_filtered_file)

        # 打印数据概览
        print_data_summary(GEDIdata)

        # 绘制波形示例
        # beam_channel = '0001'  # 选择波束
        # point_idx = 10  # 选择第10个点
        # draw_wave(GEDIdata, beam_channel, point_idx)

        # 波形分解示例
        #print(f"\n处理波束 {beam_channel} 的第 {point_idx} 个波形")

        # 找到对应的beam_idx
    #     beam_idx = None
    #     for idx, data in GEDIdata.items():
    #         if data['channel'] == beam_channel:
    #             beam_idx = idx
    #             break
    #
    #     if beam_idx is not None:
    #         beam_data = GEDIdata[beam_idx]
    #
    #         # 提取波形数据
    #         rxwave_list = beam_data['wavedata']['rxwaveform']
    #         rxwave_original = rxwave_list[point_idx].copy()
    #         noise_std = beam_data['noisedata']['noise_std'][point_idx]
    #         noise_mean = beam_data['noisedata']['noise_mean'][point_idx]
    #
    #         print(f"\n=== 原始数据信息 ===")
    #         print(f"原始波形长度: {len(rxwave_original)}")
    #         print(f"噪声标准差: {noise_std:.4f}")
    #         print(f"噪声均值: {noise_mean:.4f}")
    #         print(f"原始波形最大值: {np.max(rxwave_original):.4f}")
    #         print(f"原始波形最小值: {np.min(rxwave_original):.4f}")
    #         print(f"原始波形均值: {np.mean(rxwave_original):.4f}")
    #
    #         # 查看原始波形的统计信息
    #         non_zero = rxwave_original[rxwave_original != 0]
    #         print(f"\n非0值统计:")
    #         print(f"  非0值个数: {len(non_zero)}")
    #         if len(non_zero) > 0:
    #             print(f"  非0值最大值: {np.max(non_zero):.4f}")
    #             print(f"  非0值最小值: {np.min(non_zero):.4f}")
    #             print(f"  非0值均值: {np.mean(non_zero):.4f}")
    #
    #         # 预处理
    #         rxwave = rxwave_original[rxwave_original != 0]  # 移除0值
    #         rxwave = rxwave - noise_mean  # 减去噪声均值
    #
    #         print(f"\n=== 预处理后数据信息 ===")
    #         print(f"预处理后长度: {len(rxwave)}")
    #         if len(rxwave) > 0:
    #             print(f"预处理后最大值: {np.max(rxwave):.4f}")
    #             print(f"预处理后最小值: {np.min(rxwave):.4f}")
    #             print(f"预处理后均值: {np.mean(rxwave):.4f}")
    #
    #         # 计算阈值
    #         threshold = noise_std * 3
    #         print(f"\n噪声阈值 (3*sigma): {threshold:.4f}")
    #
    #         # 检查有多少点超过阈值
    #         if len(rxwave) > 0:
    #             above_threshold = rxwave[rxwave > threshold]
    #             print(f"超过阈值的点数: {len(above_threshold)}")
    #             if len(above_threshold) > 0:
    #                 print(f"超过阈值的最大值: {np.max(above_threshold):.4f}")
    #
    #         # 绘制原始波形和预处理后的波形
    #         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    #
    #         ax1.plot(rxwave_original, 'b-', linewidth=1)
    #         ax1.axhline(y=noise_mean, color='r', linestyle='--', label=f'噪声均值={noise_mean:.2f}')
    #         ax1.axhline(y=noise_mean + threshold, color='g', linestyle='--', label=f'阈值={threshold:.2f}')
    #         ax1.set_title('原始接收波形')
    #         ax1.set_xlabel('采样点')
    #         ax1.set_ylabel('信号强度')
    #         ax1.legend()
    #         ax1.grid(True, alpha=0.3)
    #
    #         ax2.plot(rxwave, 'r-', linewidth=1)
    #         ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    #         ax2.axhline(y=threshold, color='g', linestyle='--', label=f'阈值={threshold:.2f}')
    #         ax2.set_title('预处理后波形（移除0值并减去噪声均值）')
    #         ax2.set_xlabel('采样点')
    #         ax2.set_ylabel('信号强度')
    #         ax2.legend()
    #         ax2.grid(True, alpha=0.3)
    #
    #         plt.tight_layout()
    #         plt.show()
    #
    #         # 如果波形太弱，尝试降低阈值
    #         if len(rxwave) > 0 and np.max(rxwave) < threshold:
    #             print("\n警告: 波形最大值低于阈值，尝试降低阈值...")
    #             custom_threshold = noise_std * 1.5
    #             print(f"使用自定义阈值: {custom_threshold:.4f}")
    #
    #             if np.max(rxwave) > custom_threshold:
    #                 print(f"波形最大值 ({np.max(rxwave):.4f}) 高于自定义阈值，可以尝试分解")
    #             else:
    #                 print("波形仍然太弱，无法分解")
    #
    #         # 调用waveresolve
    #         print("\n开始波形分解...")
    #         try:
    #             prfnl, prini = waveresolve(
    #                 rx=np.arange(1, len(rxwave) + 1),
    #                 ry=rxwave,
    #                 filtwidth=15.6 / 2.355,
    #                 signalextent=None,
    #                 noise_sigma=noise_std,
    #                 txsigma=15.6 / 2.355,
    #                 maxwavenum=20,
    #                 display=1
    #             )
    #
    #             print(f"\nprfnl 形状: {prfnl.shape}")
    #             print(f"prini 形状: {prini.shape}")
    #
    #             if len(prfnl) == 0:
    #                 print("\n未检测到高斯分量，可能原因:")
    #                 print("1. 波形幅度太小，低于噪声阈值")
    #                 print("2. 波形中有效信号太少")
    #                 print("3. 拐点检测失败")
    #
    #                 # 尝试手动检查拐点
    #                 print("\n尝试手动检查波形特征...")
    #                 from scipy.signal import find_peaks
    #
    #                 peaks, properties = find_peaks(rxwave, height=threshold)
    #                 print(f"检测到 {len(peaks)} 个峰值")
    #                 if len(peaks) > 0:
    #                     print(f"峰值位置: {peaks}")
    #                     print(f"峰值高度: {properties['peak_heights']}")
    #
    #         except Exception as e:
    #             print(f"波形分解出错: {e}")
    #             import traceback
    #
    #             traceback.print_exc()
    #
    # else:
    #     print(f"文件不存在: {gedi_filtered_file}")


    #匹配数据
    # landcover_tif=r"D:\研究生\SanFrancisco\GLC_FCS30_2020_W125N40.tif"
    # output_file=r"D:\研究生\SanFrancisco\GEDI_Matched_Compact_SF_Land30.h5"
    # GEDIdata_match=match_gedi_to_landcover(gedi_filtered_file,landcover_tif)

    #多个TIF匹配
    landcover_tif_list=[r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W110N25.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W110N30.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W110N35.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W110N40.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W115N25.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W120N35.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W120N40.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W125N35.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W125N40.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W115N30.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W115N35.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W115N40.tif",
                        r"D:\研究生\SanFrancisco\SanTIF\GLC_FCS30_2020_W120N30.tif"

    ]
    output_file = r"D:\研究生\SanFrancisco\GEDIdata\GEDI_Matched_MultiTIF_SF.h5"
    GEDIdata_match=match_gedi_to_landcover_multi(gedi_filtered_file,landcover_tif_list,output_file)

    print(f"\n[2/2] 正在保存精简版数据到: {output_file}")
    try:
        saved_path=save_matched_data(
            GEDIdata_match,
            output_file)
        print("保存成功")
    except Exception as e:
        print(f"保存失败：{e}")
        import traceback
        traceback.print_exc()
