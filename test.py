import h5py
import numpy as np


def check_quality_sensitivity(GEDI_L2A_file):
    """
    查看L2A中quality_flag和sensitivity分布（直接从根目录读取）
    """
    BEAMNAME = ['/BEAM0000', '/BEAM0001', '/BEAM0010', '/BEAM0011',
                '/BEAM0101', '/BEAM0110', '/BEAM1000', '/BEAM1011']

    with h5py.File(GEDI_L2A_file, 'r') as f2:
        for beam in BEAMNAME:
            beam_name = beam[1:]
            if beam_name not in f2:
                continue

            beam_group = f2[beam_name]

            print(f"\n{'=' * 60}")
            print(f"波束 {beam_name}")
            print(f"{'=' * 60}")

            # 直接从根目录读取质量标志
            if 'quality_flag' in beam_group:
                quality_flag = beam_group['quality_flag'][:]
                print(f"\n--- 质量标志 ---")
                print(f"  quality_flag 分布:")
                flag_counts = np.bincount(quality_flag)
                for flag_val, count in enumerate(flag_counts):
                    if count > 0:
                        percentage = count / len(quality_flag) * 100
                        print(f"    flag={flag_val}: {count:6d} 点 ({percentage:5.2f}%)")

            if 'sensitivity' in beam_group:
                sensitivity = beam_group['sensitivity'][:]
                print(f"\n--- 灵敏度 ---")
                print(f"  sensitivity 统计:")
                print(f"    最小值: {np.min(sensitivity):10.4f}")
                print(f"    最大值: {np.max(sensitivity):10.4f}")
                print(f"    平均值: {np.mean(sensitivity):10.4f}")
                print(f"    中位数: {np.median(sensitivity):10.4f}")

                # sensitivity在0-1范围内的比例
                in_range = np.sum((sensitivity >= 0) & (sensitivity <= 1))
                print(f"    0-1范围内: {in_range:6d}/{len(sensitivity)} ({in_range / len(sensitivity) * 100:5.2f}%)")

            # 其他质量标志
            if 'degrade_flag' in beam_group:
                degrade = beam_group['degrade_flag'][:]
                deg_counts = np.bincount(degrade)
                print(f"\n--- 其他标志 ---")
                print(f"  degrade_flag 分布:")
                for val, count in enumerate(deg_counts):
                    if count > 0:
                        print(f"    {val}: {count} 点")

            if 'surface_flag' in beam_group:
                surface = beam_group['surface_flag'][:]
                surf_counts = np.bincount(surface)
                print(f"  surface_flag 分布:")
                for val, count in enumerate(surf_counts):
                    if count > 0:
                        print(f"    {val}: {count} 点")

            if 'selected_algorithm' in beam_group:
                selected = beam_group['selected_algorithm'][:]
                sel_counts = np.bincount(selected)
                print(f"  selected_algorithm 分布:")
                for val, count in enumerate(sel_counts):
                    if count > 0:
                        print(f"    {val}: {count} 点")


# 运行
gedi_l2a = r"D:\研究生\SanFrancisco\GEDI02_A_2025032182236_O34785_02_T02894_02_004_02_V002.h5"
check_quality_sensitivity(gedi_l2a)

