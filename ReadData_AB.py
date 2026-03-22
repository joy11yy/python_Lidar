import h5py
import numpy as np
import os
from waveform_read import waveform_read


def ReadGEDI_L1B_L2A(GEDI_L1B_file, GEDI_L2A_file):
    """
    读取GEDI L1B数据，并加入L2A质量筛选

    参数：
    ----------
    GEDI_L1B_file : str
        L1B文件路径
    GEDI_L2A_file : str
        L2A文件路径

    返回：
    ----------
    dict
        包含质量筛选后的数据
    """
    BEAMNAME = ['/BEAM0000', '/BEAM0001', '/BEAM0010', '/BEAM0011',
                '/BEAM0101', '/BEAM0110', '/BEAM1000', '/BEAM1011']

    # 读取L2A的质量数据
    l2a_quality = {}  # 格式: {beam_name: {shot_number: quality_info}}

    if os.path.exists(GEDI_L2A_file):
        print(f"读取L2A质量数据: {GEDI_L2A_file}")
        with h5py.File(GEDI_L2A_file, 'r') as f2:
            # L2A也是分波束的，每个波束在根目录下有数据
            for beam in BEAMNAME:
                beam_name = beam[1:]  # 去掉开头的'/'
                if beam_name in f2:
                    l2a_quality[beam_name] = {}

                    # 获取该波束的数据（直接从根目录读取）
                    beam_group = f2[beam_name]

                    # 读取shot_number和质量标志（都在根目录下）
                    shot_number = beam_group['shot_number'][:]
                    quality_flag = beam_group['quality_flag'][:]  # 注意：这里是 quality_flag 不是 quality_flag_a1
                    sensitivity = beam_group['sensitivity'][:]    # 注意：这里是 sensitivity 不是 sensitivity_a1
                    selected_algorithm = beam_group['selected_algorithm'][:]
                    degrade_flag = beam_group['degrade_flag'][:]
                    surface_flag = beam_group['surface_flag'][:]
                    lat_lowestmode=beam_group['lat_lowestmode'][:]
                    lon_lowestmode=beam_group['lon_lowestmode'][:]

                    print(f"  波束 {beam_name}: {len(shot_number)} 个shot")
                    print(f"    quality_flag 为0和1的数据: {np.bincount(quality_flag)}")
                    print(f"    sensitivity 范围: [{np.min(sensitivity):.3f}, {np.max(sensitivity):.3f}]")

                    # 建立映射
                    for i, shot in enumerate(shot_number):
                        l2a_quality[beam_name][shot] = {
                            'quality_flag': quality_flag[i],
                            'sensitivity': sensitivity[i],
                            'selected_algorithm': selected_algorithm[i],
                            'degrade_flag': degrade_flag[i],
                            'surface_flag': surface_flag[i],
                            'lat_lowestmode': lat_lowestmode[i],
                            'lon_lowestmode': lon_lowestmode[i]
                        }

        # 统计总加载数
        total_shots = sum(len(shots) for shots in l2a_quality.values())
        print(f"  总共加载了 {total_shots} 个shot的质量数据")
    else:
        print(f"L2A文件不存在: {GEDI_L2A_file}")
        return None

    GEDIdata = {}

    with h5py.File(GEDI_L1B_file, 'r') as f:
        for nameindex, BEAM in enumerate(BEAMNAME):
            beam_idx = nameindex
            BEAM2 = BEAM[5:]  # 得到 '0000', '0001'等
            BEAM_full = BEAM[1:]  # 得到 'BEAM0000'等
            print(f"正在处理波束{BEAM2}...")

            # 检查波束是否存在
            if BEAM not in f:
                print(f"  波束{BEAM2}不存在，跳过")
                continue

            # 获取该波束的shot_number
            shot_number = f[f"{BEAM}/shot_number"][:].flatten()

            # 初始化有效点列表
            valid_indices_list = []
            valid_quality_info = {
                'quality_flag': [],
                'sensitivity': [],
                'selected_algorithm': [],
                'degrade_flag': [],
                'surface_flag': [],
                'is_valid': [],
                'lat_lowestmode': [],
                'lon_lowestmode': []
            }

            # 进行质量筛选
            valid_count = 0
            if BEAM_full in l2a_quality:
                beam_quality = l2a_quality[BEAM_full]

                for i, shot in enumerate(shot_number):
                    if shot in beam_quality:
                        q = beam_quality[shot]

                        # 质量筛选条件（根据你的数据调整）
                        # quality_flag=0 是正常的，sensitivity 需要 >0 且在合理范围内
                        if (q['sensitivity'] > 0.8) and (q['sensitivity'] <= 1) and (q['degrade_flag'] == 0) and (q['quality_flag']==1):
                            valid_indices_list.append(i)
                            valid_quality_info['quality_flag'].append(q['quality_flag'])
                            valid_quality_info['sensitivity'].append(q['sensitivity'])
                            valid_quality_info['selected_algorithm'].append(q['selected_algorithm'])
                            valid_quality_info['degrade_flag'].append(q['degrade_flag'])
                            valid_quality_info['surface_flag'].append(q['surface_flag'])
                            valid_quality_info['is_valid'].append(True)
                            valid_quality_info['lat_lowestmode'].append(q['lat_lowestmode'])
                            valid_quality_info['lon_lowestmode'].append(q['lon_lowestmode'])
                            valid_count += 1

            print(f"  质量筛选: {valid_count}/{len(shot_number)} 个有效点")

            if valid_count == 0:
                print(f"  波束{BEAM2}没有有效点，跳过")
                continue

            # 转换为numpy数组
            valid_indices = np.array(valid_indices_list)
            len_data = len(valid_indices)

            # 初始化数据结构
            GEDIdata[beam_idx] = {
                'channel': BEAM2,
                'pointnum': len_data,
                'vailddata': len_data,
                'deltatime': None,
                'wavedata': {},
                'fpdata': {},
                'noisedata': {},
                'quality': {
                    'quality_flag': np.array(valid_quality_info['quality_flag'], dtype=np.uint8),
                    'sensitivity': np.array(valid_quality_info['sensitivity'], dtype=np.float32),
                    'selected_algorithm': np.array(valid_quality_info['selected_algorithm'], dtype=np.uint8),
                    'degrade_flag': np.array(valid_quality_info['degrade_flag'], dtype=np.uint8),
                    'surface_flag': np.array(valid_quality_info['surface_flag'], dtype=np.uint8),
                    'is_valid': np.array(valid_quality_info['is_valid'], dtype=bool)
                },
                'shot_number': shot_number[valid_indices]
            }

            try:
                # 读取所有数据，只取有效点
                delta_time = f[f"{BEAM}/delta_time"][:].flatten()[valid_indices]

                # 波形参数
                tx_sample_count = f[f"{BEAM}/tx_sample_count"][:].flatten()[valid_indices]
                tx_sample_start_index = f[f"{BEAM}/tx_sample_start_index"][:].flatten()[valid_indices]

                rx_sample_count = f[f"{BEAM}/rx_sample_count"][:].flatten()[valid_indices]
                rx_sample_start_index_orig = f[f"{BEAM}/rx_sample_start_index"][:].flatten()
                rx_energy = f[f"{BEAM}/rx_energy"][:].flatten()[valid_indices]

                # 定位数据
                ins_lat = f[f"{BEAM}/geolocation/latitude_instrument"][:].flatten()[valid_indices]
                ins_lon = f[f"{BEAM}/geolocation/longitude_instrument"][:].flatten()[valid_indices]
                ins_alt = f[f"{BEAM}/geolocation/altitude_instrument"][:].flatten()[valid_indices]

                elev_bin0 = f[f"{BEAM}/geolocation/elevation_bin0"][:].flatten()[valid_indices]
                elev_lastbin = f[f"{BEAM}/geolocation/elevation_lastbin"][:].flatten()[valid_indices]
                lat_bin0 = f[f"{BEAM}/geolocation/latitude_bin0"][:].flatten()[valid_indices]
                lat_lastbin = f[f"{BEAM}/geolocation/latitude_lastbin"][:].flatten()[valid_indices]
                lon_bin0 = f[f"{BEAM}/geolocation/longitude_bin0"][:].flatten()[valid_indices]
                lon_lastbin = f[f"{BEAM}/geolocation/longitude_lastbin"][:].flatten()[valid_indices]




                # 噪声数据
                noise_mean = f[f"{BEAM}/noise_mean_corrected"][:].flatten()[valid_indices]
                noise_std = f[f"{BEAM}/noise_stddev_corrected"][:].flatten()[valid_indices]
                solar_elev = f[f"{BEAM}/geolocation/solar_elevation"][:].flatten()[valid_indices]
                solar_azimuth = f[f"{BEAM}/geolocation/solar_azimuth"][:].flatten()[valid_indices]

                # 大地水准面校正
                elev_EGM2008_corr = f[f"{BEAM}/geophys_corr/geoid"][:].flatten()[valid_indices]

                # 波形数据读取
                txwaveformdata = f[f"{BEAM}/txwaveform"][:].flatten()
                txwaveform = waveform_read(txwaveformdata, tx_sample_start_index, tx_sample_count,
                                           np.arange(1, len_data + 1))

                rxwaveformdata = f[f"{BEAM}/rxwaveform"][:].flatten()
                rx_sample_start_index = rx_sample_start_index_orig[valid_indices]
                rxwaveform = waveform_read(rxwaveformdata, rx_sample_start_index, rx_sample_count,
                                           np.arange(1, len_data + 1))

                # 保存数据
                GEDIdata[beam_idx]['wavedata'] = {
                    'txwaveform': txwaveform,
                    'rxwaveform': rxwaveform,
                    'rx_sample_count': rx_sample_count,
                    'rx_energy': rx_energy
                }

                GEDIdata[beam_idx]['fpdata'] = {
                    'ins_lat': ins_lat,
                    'ins_lon': ins_lon,
                    'ins_alt': ins_alt,
                    'elev_bin0': elev_bin0,
                    'elev_lastbin': elev_lastbin,
                    'lat_bin0': lat_bin0,
                    'lat_lastbin': lat_lastbin,
                    'lon_bin0': lon_bin0,
                    'lon_lastbin': lon_lastbin,
                    'elev_EGM2008_corr': elev_EGM2008_corr,
                    'lat_lowestmode': np.array(valid_quality_info['lat_lowestmode'], dtype=np.float64),
                    'lon_lowestmode': np.array(valid_quality_info['lon_lowestmode'], dtype=np.float64)

                }

                GEDIdata[beam_idx]['noisedata'] = {
                    'noise_mean': noise_mean,
                    'noise_std': noise_std,
                    'solar_elev': solar_elev,
                    'solar_azimuth': solar_azimuth
                }

                GEDIdata[beam_idx]['deltatime'] = delta_time

                print(f"  波束 {BEAM2} 读取完成，包含 {len_data} 个有效脉冲")

            except Exception as e:
                print(f"  处理波束 {BEAM2} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    return GEDIdata

# 使用示例
if __name__ == "__main__":
    # gedi_l1b = r"D:\研究生\SanFrancisco\GEDI01_B_2025032182236_O34785_02_T02894_02_006_02_V002.h5"
    # gedi_l2a = r"D:\研究生\SanFrancisco\GEDI02_A_2025032182236_O34785_02_T02894_02_004_02_V002.h5"

    gedi_l1b = r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_2025009102237_O34423_03_T04153_02_006_02_V002.h5"
    gedi_l2a = r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_2025009102237_O34423_03_T04153_02_004_02_V002.h5"

    # gedi_l1b=r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_2024360161006_O34194_03_T07611_02_006_02_V002_subsetted.h5"
    # gedi_l2a=r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_2024360161006_O34194_03_T07611_02_004_02_V002_subsetted.h5"
    # 读取带质量筛选的数据
    GEDIdata = ReadGEDI_L1B_L2A(gedi_l1b, gedi_l2a)

    if GEDIdata:
        # 统计总有效点数
        total_valid = 0
        for beam_idx, beam_data in GEDIdata.items():
            print(f"\n波束 {beam_data['channel']}:")
            print(f"  有效点数: {beam_data['pointnum']}")

            # 查看质量信息
            if 'quality' in beam_data:
                q = beam_data['quality']
                if len(q['sensitivity']) > 0:
                    print(f"  平均灵敏度: {np.mean(q['sensitivity']):.3f}")
                    print(f"  质量良好比例: {np.mean(q['quality_flag']) * 100:.1f}%")

            total_valid += beam_data['pointnum']

        print(f"\n总共有效点数: {total_valid}")