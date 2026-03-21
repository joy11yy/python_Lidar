import h5py
import numpy as np
import os
from waveresolve import waveresolve
import matplotlib.pyplot as plt



def load_filtered_gedi_data(filtered_file):
    """
    加载之前保存的筛选后GEDI数据

    参数：
    ----------
    filtered_file : str
        筛选后的数据文件路径

    返回：
    ----------
    dict
        与原始读取函数返回格式相同的字典
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
                if 'quality' in beam_group:
                    beam_data['quality'] = {}
                    for key in beam_group['quality'].keys():
                        beam_data['quality'][key] = beam_group['quality'][key][:]

                # 加载波形数据
                beam_data['wavedata'] = {}
                if 'wavedata' in beam_group:
                    wavedata_group = beam_group['wavedata']
                    for key in wavedata_group.keys():
                        if key in ['txwaveform', 'rxwaveform']:
                            # 变长数组，需要逐个读取
                            waveforms = []
                            for waveform_vlen in wavedata_group[key]:
                                waveforms.append(waveform_vlen[:])
                            beam_data['wavedata'][key] = waveforms
                        else:
                            beam_data['wavedata'][key] = wavedata_group[key][:]

                # 加载定位数据
                beam_data['fpdata'] = {}
                if 'fpdata' in beam_group:
                    fp_group=beam_group['fpdata']
                    first_lon = fp_group['ins_lon'][0]
                    first_lat = fp_group['ins_lat'][0]

                    # 简单的合理性检查
                    # 旧金山经度应该在 -123 左右，纬度在 37 左右
                    is_suspicious = False
                    if -90 < first_lon < 90 and (first_lat < -90 or first_lat > 90):
                        print(f"   ⚠️ [警告] 波束 {beam_name} 数据疑似经纬度颠倒！")
                        print(f"      读取到的 ins_lon[0] = {first_lon} (看起来像纬度)")
                        print(f"      读取到的 ins_lat[0] = {first_lat} (看起来像经度)")
                        is_suspicious = True

                    if not is_suspicious:
                        print(f"   [检查] 波束 {beam_name} 坐标样例 -> Lon: {first_lon:.4f}, Lat: {first_lat:.4f}")

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


def print_data_summary(GEDIdata):
    """打印数据概览"""
    print("\n" + "=" * 60)
    print("GEDI 数据概览")
    print("=" * 60)

    total_shots = 0
    for beam_idx, beam_data in GEDIdata.items():
        print(f"\n波束 {beam_data['channel']}:")
        print(f"  有效点数: {beam_data['pointnum']}")

        if 'quality' in beam_data and len(beam_data['quality']['sensitivity']) > 0:
            q = beam_data['quality']
            print(f"  平均灵敏度: {np.mean(q['sensitivity']):.3f}")
            print(f"  质量良好比例: {np.mean(q['quality_flag']) * 100:.1f}%")

        total_shots += beam_data['pointnum']

    print(f"\n总共有效点数: {total_shots}")
    print("=" * 60)


def draw_wave(GEDIdata, beam_channel, point_idx):
    """绘制指定波束和点的波形"""
    # 找到对应的beam_idx
    beam_idx = None
    for idx, data in GEDIdata.items():
        if data['channel'] == beam_channel:
            beam_idx = idx
            break

    if beam_idx is None:
        print(f"未找到波束 {beam_channel}")
        return

    beam_data = GEDIdata[beam_idx]

    if point_idx >= beam_data['pointnum']:
        print(f"点索引 {point_idx} 超出范围 (最大: {beam_data['pointnum'] - 1})")
        return

    # 提取数据
    txwave = beam_data['wavedata']['txwaveform'][point_idx]
    rxwave = beam_data['wavedata']['rxwaveform'][point_idx]

    # 绘制
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(txwave, 'b-', linewidth=1)
    ax1.set_title(f'波束 {beam_channel} 第 {point_idx} 个发射波形')
    ax1.set_xlabel('采样点')
    ax1.set_ylabel('信号强度')
    ax1.grid(True, alpha=0.3)

    ax2.plot(rxwave, 'r-', linewidth=1)
    ax2.set_title(f'波束 {beam_channel} 第 {point_idx} 个接收波形')
    ax2.set_xlabel('采样点')
    ax2.set_ylabel('信号强度')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


