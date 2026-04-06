import h5py
import numpy as np
import os
from waveform_read import waveform_read
from datetime import datetime
from ReadData_L1B_L2A import ReadGEDI_L1B_L2A


def save_filtered_gedi(GEDIdata, original_l1b_file):
    """
    将筛选后的GEDI数据保存到原始文件同目录

    参数：
    ----------
    GEDIdata : dict
        筛选后的数据字典
    original_l1b_file : str
        原始L1B文件路径，用于确定保存目录和生成文件名

    返回：
    ----------
    str
        保存的文件路径
    """
    # 生成输出文件名：在原文件名基础上加_filtered
    base_name = os.path.basename(original_l1b_file)
    # 将GEDI01_B替换为GEDI_filtered，或者直接在原名上加_filtered
    if base_name.startswith('GEDI01_B'):
        output_name = base_name.replace('GEDI01_B', 'GEDI_filtered')
    else:
        # 如果不是标准命名，就在末尾加_filtered
        name_parts = base_name.split('.')
        output_name = name_parts[0] + '_filtered1.' + name_parts[1]

    # 输出到同一个目录
    output_file = os.path.join(os.path.dirname(original_l1b_file), output_name)

    print(f"\n正在保存筛选后的数据到: {output_file}")

    with h5py.File(output_file, 'w') as f:
        # 添加一些元数据
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['source_file'] = base_name
        f.attrs['total_valid_shots'] = sum(beam['pointnum'] for beam in GEDIdata.values())

        # 保存每个波束的数据
        for beam_idx, beam_data in GEDIdata.items():
            beam_name = f"BEAM{beam_data['channel']}"
            print(f"  保存波束 {beam_name}...")

            # 创建波束组
            beam_group = f.create_group(beam_name)

            # 保存基本属性
            beam_group.attrs['pointnum'] = beam_data['pointnum']

            # 保存基本数据
            beam_group.create_dataset('shot_number', data=beam_data['shot_number'], compression='gzip')
            beam_group.create_dataset('delta_time', data=beam_data['deltatime'], compression='gzip')

            # 保存质量信息
            quality_group = beam_group.create_group('quality')
            for key, value in beam_data['quality'].items():
                quality_group.create_dataset(key, data=value, compression='gzip')

            # 保存波形数据
            wavedata_group = beam_group.create_group('wavedata')

            # 处理变长波形
            for wave_key in ['txwaveform', 'rxwaveform']:
                if wave_key in beam_data['wavedata']:
                    waveforms = beam_data['wavedata'][wave_key]
                    # 使用变长数据类型
                    dt = h5py.vlen_dtype(np.dtype('float32'))
                    wave_dataset = wavedata_group.create_dataset(wave_key, (len(waveforms),), dtype=dt,
                                                                 compression='gzip')
                    for i, w in enumerate(waveforms):
                        wave_dataset[i] = np.array(w, dtype=np.float32)

            # 保存其他波形数据
            for key in ['rx_sample_count', 'rx_energy']:
                if key in beam_data['wavedata']:
                    wavedata_group.create_dataset(key, data=beam_data['wavedata'][key], compression='gzip')

            # 保存定位数据
            fpdata_group = beam_group.create_group('fpdata')
            for key, value in beam_data['fpdata'].items():
                fpdata_group.create_dataset(key, data=value, compression='gzip')

            # 保存噪声数据
            noisedata_group = beam_group.create_group('noisedata')
            for key, value in beam_data['noisedata'].items():
                noisedata_group.create_dataset(key, data=value, compression='gzip')

    # 显示文件大小
    file_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"保存完成！文件大小: {file_size:.2f} MB")

    return output_file


def load_filtered_gedi(filtered_file):
    """
    加载之前保存的筛选后GEDI数据

    参数：
    ----------
    filtered_file : str
        筛选后的数据文件路径

    返回：
    ----------
    dict
        与ReadGEDI_L1B_L2A返回格式相同的字典
    """
    print(f"\n加载筛选后的数据从: {filtered_file}")

    GEDIdata = {}

    with h5py.File(filtered_file, 'r') as f:
        print(f"数据创建时间: {f.attrs.get('creation_date', 'Unknown')}")
        print(f"源文件: {f.attrs.get('source_file', 'Unknown')}")
        print(f"总有效shots: {f.attrs.get('total_valid_shots', 0)}")

        beam_idx = 0
        for beam_name in f.keys():
            if beam_name.startswith('BEAM'):
                beam_group = f[beam_name]
                # 重建数据结构
                beam_data = {
                    'channel': beam_name[4:],  # 从BEAM0000中提取0000
                    'pointnum': beam_group.attrs['pointnum'],
                    'vailddata': beam_group.attrs['pointnum'],
                    'shot_number': beam_group['shot_number'][:],
                    'deltatime': beam_group['delta_time'][:],
                }

                # 加载质量信息
                beam_data['quality'] = {}
                if 'quality' in beam_group:
                    for key in beam_group['quality'].keys():
                        beam_data['quality'][key] = beam_group['quality'][key][:]

                # 加载波形数据
                beam_data['wavedata'] = {}
                if 'wavedata' in beam_group:
                    wavedata_group = beam_group['wavedata']
                    for key in wavedata_group.keys():
                        if key in ['txwaveform', 'rxwaveform']:
                            # 变长数组
                            waveforms = [w[:] for w in wavedata_group[key]]
                            beam_data['wavedata'][key] = waveforms
                        else:
                            beam_data['wavedata'][key] = wavedata_group[key][:]

                # 加载定位数据
                beam_data['fpdata'] = {}
                if 'fpdata' in beam_group:
                    for key in beam_group['fpdata'].keys():
                        beam_data['fpdata'][key] = beam_group['fpdata'][key][:]

                # 加载噪声数据
                beam_data['noisedata'] = {}
                if 'noisedata' in beam_group:
                    for key in beam_group['noisedata'].keys():
                        beam_data['noisedata'][key] = beam_group['noisedata'][key][:]

                GEDIdata[beam_idx] = beam_data
                beam_idx += 1

                print(f"  加载波束 {beam_data['channel']}: {beam_data['pointnum']} shots")

    return GEDIdata


# 修改主程序
if __name__ == "__main__":
    # gedi_l1b = r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_2025009102237_O34423_03_T04153_02_006_02_V002.h5"
    # gedi_l2a = r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_2025009102237_O34423_03_T04153_02_004_02_V002.h5"
    gedi_l1b = r"D:\研究生\SanFrancisco\GEDIdata\GEDI01_B_2025009102237_O34423_03_T04153_02_006_02_V002.h5"
    gedi_l2a = r"D:\研究生\SanFrancisco\GEDIdata\GEDI02_A_2025009102237_O34423_03_T04153_02_004_02_V002.h5"

    # 读取带质量筛选的数据
    GEDIdata = ReadGEDI_L1B_L2A(gedi_l1b, gedi_l2a)

    if GEDIdata:
        # 保存筛选后的数据
        saved_file = save_filtered_gedi(GEDIdata, gedi_l1b)

        # 统计总有效点数
        total_valid = 0
        for beam_idx, beam_data in GEDIdata.items():
            print(f"\n波束 {beam_data['channel']}:")
            print(f"  有效点数: {beam_data['pointnum']}")

            if 'quality' in beam_data:
                q = beam_data['quality']
                if len(q['sensitivity']) > 0:
                    print(f"  平均灵敏度: {np.mean(q['sensitivity']):.3f}")
                    print(f"  质量良好比例: {np.mean(q['quality_flag']) * 100:.1f}%")

            total_valid += beam_data['pointnum']

        print(f"\n总共有效点数: {total_valid}")
        print(f"\n筛选后的数据已保存到: {saved_file}")

