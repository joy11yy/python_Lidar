""""
本文件包含了合并多个文件数据的函数MergedGEDIdata
将合并后的数据保存为新的h5文件

"""

import datetime
import ReadData_L1B_L2A
import os
import numpy as np
import h5py
def MergeGEDIData(file_pairs):
    """"合并GEDI数据，保留原始波形长度"""
    """
       合并多个GEDI文件的GEDIdata数据

       参数：
       ----------
       file_pairs : list of tuple
           包含文件路径对的列表，每个元素是 (L1B文件路径, L2A文件路径)
           例如: [('L1B1.h5', 'L2A1.h5'), ('L1B2.h5', 'L2A2.h5'), ...]

       返回：
       ----------
       dict
           合并后的GEDIdata，格式与ReadData_AB返回的相同
           
       """
    merged_data = {}
    for i, (l1b_path, l2a_path) in enumerate(file_pairs):
        print(f"\n正在处理第 {i + 1} 个文件: {os.path.basename(l1b_path)}")

        # 调用现有的读取函数
        gedi_data = ReadData_L1B_L2A.ReadGEDI_L1B_L2A(l1b_path, l2a_path)

        if gedi_data is None:
            print(f"  警告: 文件 {l1b_path} 读取失败，跳过")
            continue

        # 遍历每个波束
        for beam_idx, beam_data in gedi_data.items():
            if beam_idx not in merged_data:
                # 第一次遇到这个波束，直接复制
                merged_data[beam_idx] = beam_data.copy()
            else:
                # 已经存在，需要拼接
                existing = merged_data[beam_idx]
                new = beam_data

                # 更新点数量
                existing['pointnum'] += new['pointnum']
                existing['vailddata'] += new['vailddata']

                # 拼接一维数组字段
                # deltatime
                if existing['deltatime'] is not None and new['deltatime'] is not None:
                    existing['deltatime'] = np.concatenate([existing['deltatime'], new['deltatime']])

                # shot_number
                if 'shot_number' in existing and 'shot_number' in new:
                    existing['shot_number'] = np.concatenate([existing['shot_number'], new['shot_number']])

                # quality 字典
                for qual_key in existing['quality'].keys():
                    if qual_key in new['quality']:
                        existing['quality'][qual_key] = np.concatenate(
                            [existing['quality'][qual_key], new['quality'][qual_key]]
                        )

                # wavedata 字典
                for wave_key in existing['wavedata'].keys():
                    if wave_key in new['wavedata'] and wave_key != 'rxwaveform' and wave_key != 'txwaveform':
                        # 普通数组直接拼接
                        existing['wavedata'][wave_key] = np.concatenate(
                            [existing['wavedata'][wave_key], new['wavedata'][wave_key]]
                        )
                    elif wave_key in ['rxwaveform', 'txwaveform']:
                        # 波形数据是列表形式，需要逐元素扩展
                        existing['wavedata'][wave_key].extend(new['wavedata'][wave_key])

                # fpdata 字典
                for fp_key in existing['fpdata'].keys():
                    if fp_key in new['fpdata']:
                        existing['fpdata'][fp_key] = np.concatenate(
                            [existing['fpdata'][fp_key], new['fpdata'][fp_key]]
                        )

                # noisedata 字典
                for noise_key in existing['noisedata'].keys():
                    if noise_key in new['noisedata']:
                        existing['noisedata'][noise_key] = np.concatenate(
                            [existing['noisedata'][noise_key], new['noisedata'][noise_key]]
                        )

        print(f"  第 {i + 1} 个文件处理完成，当前总数据量: {sum(merged_data[b]['pointnum'] for b in merged_data)} 个有效点")

    return merged_data


def save_merged_data(merged_data, output_path):
    """
    保存合并后的GEDI数据到h5文件
    """
    print(f"\n正在保存合并后的数据到: {output_path}")

    with h5py.File(output_path, 'w') as f:
        # 保存元数据
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['total_shots'] = sum(beam['pointnum'] for beam in merged_data.values())

        # 遍历每个波束
        for beam_idx, beam_data in merged_data.items():
            beam_name = f"BEAM{beam_data['channel']}"
            print(f"  保存波束 {beam_name} ({beam_data['pointnum']} shots)...")

            # 创建波束组
            grp = f.create_group(beam_name)
            grp.attrs['pointnum'] = beam_data['pointnum']

            # 保存基本数据
            grp.create_dataset('shot_number', data=beam_data['shot_number'], compression='gzip')
            if beam_data['deltatime'] is not None:
                grp.create_dataset('delta_time', data=beam_data['deltatime'], compression='gzip')

            # 保存quality
            q_grp = grp.create_group('quality')
            for key, val in beam_data['quality'].items():
                q_grp.create_dataset(key, data=val, compression='gzip')

            # 保存wavedata（波形是变长的，需要特殊处理）
            w_grp = grp.create_group('wavedata')
            for key in ['rx_sample_count', 'rx_energy']:
                if key in beam_data['wavedata']:
                    w_grp.create_dataset(key, data=beam_data['wavedata'][key], compression='gzip')

            # 处理变长波形
            for wave_key in ['txwaveform', 'rxwaveform']:
                if wave_key in beam_data['wavedata']:
                    waveforms = beam_data['wavedata'][wave_key]
                    dt = h5py.vlen_dtype(np.dtype('float32'))
                    ds = w_grp.create_dataset(wave_key, (len(waveforms),), dtype=dt, compression='gzip')
                    for i, w in enumerate(waveforms):
                        ds[i] = np.array(w, dtype=np.float32)

            # 保存fpdata
            fp_grp = grp.create_group('fpdata')
            for key, val in beam_data['fpdata'].items():
                fp_grp.create_dataset(key, data=val, compression='gzip')

            # 保存noisedata
            n_grp = grp.create_group('noisedata')
            for key, val in beam_data['noisedata'].items():
                n_grp.create_dataset(key, data=val, compression='gzip')

    file_size = os.path.getsize(output_path) / 1024 / 1024
    return output_path

if __name__ == "__main__":
    # 方式1：传入文件对列表
    file_pairs = [
        (r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_002-20260405_144930\GEDI01_B_2024243075949_O32374_02_T06551_02_006_04_V002_subsetted.h5",r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_002-20260405_143457\GEDI02_A_2024243075949_O32374_02_T06551_02_004_04_V002_subsetted.h5"),
        (r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_002-20260405_144930\GEDI01_B_2024247062601_O32435_02_T09397_02_006_04_V002_subsetted.h5",r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_002-20260405_143457\GEDI02_A_2024247062601_O32435_02_T09397_02_004_04_V002_subsetted.h5"),
        (r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_002-20260405_144930\GEDI01_B_2024270040236_O32790_03_T04612_02_006_02_V002_subsetted.h5",r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_002-20260405_143457\GEDI02_A_2024270040236_O32790_03_T04612_02_004_02_V002_subsetted.h5"),
        (r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_002-20260405_144930\GEDI01_B_2024270210406_O32801_02_T03858_02_006_02_V002_subsetted.h5",r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_002-20260405_143457\GEDI02_A_2024270210406_O32801_02_T03858_02_004_02_V002_subsetted.h5"),
        (r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_002-20260405_144930\GEDI01_B_2024155183808_O31016_02_T08280_02_006_02_V002_subsetted.h5",r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_002-20260405_143457\GEDI02_A_2024155183808_O31016_02_T08280_02_004_02_V002_subsetted.h5"),
        (r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_002-20260405_144930\GEDI01_B_2024181150536_O31417_03_T00190_02_006_03_V002_subsetted.h5",r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_002-20260405_143457\GEDI02_A_2024181150536_O31417_03_T00190_02_004_03_V002_subsetted.h5"),
        (r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_002-20260405_144930\GEDI01_B_2024182080705_O31428_02_T08280_02_006_03_V002_subsetted.h5",r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_002-20260405_143457\GEDI02_A_2024182080705_O31428_02_T08280_02_004_03_V002_subsetted.h5"),
        (r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_002-20260405_144930\GEDI01_B_2024185133041_O31478_03_T01919_02_006_03_V002_subsetted.h5",r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_002-20260405_143457\GEDI02_A_2024185133041_O31478_03_T01919_02_004_03_V002_subsetted.h5"),
        (r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_002-20260405_144930\GEDI01_B_2024216012631_O31951_03_T00343_02_006_03_V002_subsetted.h5",r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_002-20260405_143457\GEDI02_A_2024216012631_O31951_03_T00343_02_004_03_V002_subsetted.h5")

    ]

    merged_data = MergeGEDIData(file_pairs)

    if merged_data:
        print("\n=== 合并完成 ===")
        for beam_idx, beam_data in merged_data.items():
            print(f"波束 {beam_data['channel']}: {beam_data['pointnum']} 个有效点")
    # 动态生成带时间戳的文件名,文件保存名字为年月日
    from datetime import datetime

    output_filename = f"merged_gedi_data_{datetime.now().strftime('%Y%m%d')}.h5"
    output_path = os.path.join(r"D:\研究生\SanFrancisco\GEDIdata", output_filename)
    save_merged_data(merged_data, output_path)