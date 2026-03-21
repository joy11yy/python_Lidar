import h5py
import numpy as np
import pandas as pd
import os
from waveform_read import  waveform_read
def ReadGEDI_L1B(GEDI_L1B):
    """"
    读取GEDI L1B数据
    返回包含8个波束的数据字典
    波形数据：发射波形txwaveform 接收波形rxwaveform
    GEDIdata[beam_idx] = {
    'channel': BEAM2,           # 字符串，如 '0000', '0001', ...
    'pointnum': len_data,       # 整数，脉冲数量
    'vailddata': len_data,      # 整数
    'deltatime': array,         # numpy数组，形状 (pointnum,)
    'wavedata': {                # 字典
        'txwaveform': array,     # numpy数组，形状 (pointnum, 采样点数)
        'rxwaveform': array,     # numpy数组，形状 (pointnum, 采样点数)
        'rx_sample_count': array,# numpy数组，形状 (pointnum,)
        'rx_energy': array       # numpy数组，形状 (pointnum,)
    },
    'fpdata': {                  # 字典，所有都是numpy数组，形状 (pointnum,)
        'ins_lat': array,
        'ins_lon': array,
        'ins_alt': array,
        'elev_bin0': array,
        'elev_lastbin': array,
        'lat_bin0': array,
        'lat_lastbin': array,
        'lon_bin0': array,
        'lon_lastbin': array,
        'beam_azimuth': array,
        'beam_azimuth_error': array,
        'beam_elevation': array,
        'beam_elevation_error': array,
        'lat_lastbin_error': array,
        'lon_lastbin_error': array,
        'elev_EGM2008_corr': array
    },
    'noisedata': {               # 字典，所有都是numpy数组，形状 (pointnum,)
        'noise_mean': array,
        'noise_std': array,
        'solar_elev': array,
        'solar_azimuth': array
    }
}
    """
    BEAMNAME=['/BEAM0000', '/BEAM0001', '/BEAM0010', '/BEAM0011',
        '/BEAM0101', '/BEAM0110', '/BEAM1000', '/BEAM1011']
    GEDIdata={}
    with h5py.File(GEDI_L1B,'r') as f:
        for nameindex,BEAM in enumerate(BEAMNAME):
            beam_idx=nameindex
            BEAM2=BEAM[5:]
            print(f"正在处理波束{BEAM2}...")

            GEDIdata[beam_idx] = {
                'channel': BEAM2,
                'pointnum': 0,
                'vailddata': 0,
                'deltatime': None,
                'wavedata': {},
                'fpdata': {},
                'noisedata': {}
            }
            try:
                #delta是代表激光脚点相对参考时间的偏移量
                delta_time_path=f"{BEAM}/delta_time"
                if delta_time_path not in f:
                    print(f"波束{BEAM2}中没有delta_time 数据 跳过")
                    GEDIdata[beam_idx]['pointnum']=0
                    GEDIdata[beam_idx]['vailddata']=0
                    continue

                gede01_deltatime = f[delta_time_path][:].flatten()
                gedi01_index = np.arange(len(gede01_deltatime))

                # 初始化数据存储
                wavedata = {}
                fpdata = {}
                noisedata = {}

                len_data = len(gedi01_index)

                # 发射波形参数
                tx_sample_count = f[f"{BEAM}/tx_sample_count"][:].flatten()[gedi01_index]
                tx_sample_start_index = f[f"{BEAM}/tx_sample_start_index"][:].flatten()[gedi01_index]

                # 接收波形参数
                rx_sample_count = f[f"{BEAM}/rx_sample_count"][:].flatten()[gedi01_index]
                rx_sample_start_index_orig = f[f"{BEAM}/rx_sample_start_index"][:].flatten()
                rx_energy = f[f"{BEAM}/rx_energy"][:].flatten()[gedi01_index]

                # ========== L1A数据 - 定位 ==========
                # 仪器位置
                ins_lat = f[f"{BEAM}/geolocation/latitude_instrument"][:].flatten()[gedi01_index]
                ins_lon = f[f"{BEAM}/geolocation/longitude_instrument"][:].flatten()[gedi01_index]
                ins_alt = f[f"{BEAM}/geolocation/altitude_instrument"][:].flatten()[gedi01_index]

                # 波形bin的位置
                elev_bin0 = f[f"{BEAM}/geolocation/elevation_bin0"][:].flatten()[gedi01_index]
                elev_lastbin = f[f"{BEAM}/geolocation/elevation_lastbin"][:].flatten()[gedi01_index]
                lat_bin0 = f[f"{BEAM}/geolocation/latitude_bin0"][:].flatten()[gedi01_index]
                lat_lastbin = f[f"{BEAM}/geolocation/latitude_lastbin"][:].flatten()[gedi01_index]
                lon_bin0 = f[f"{BEAM}/geolocation/longitude_bin0"][:].flatten()[gedi01_index]
                lon_lastbin = f[f"{BEAM}/geolocation/longitude_lastbin"][:].flatten()[gedi01_index]

                # 波束角度
                beam_azimuth = f[f"{BEAM}/geolocation/local_beam_azimuth"][:].flatten()[gedi01_index]
                beam_azimuth_error = f[f"{BEAM}/geolocation/local_beam_azimuth_error"][:].flatten()[gedi01_index]
                beam_elevation = f[f"{BEAM}/geolocation/local_beam_elevation"][:].flatten()[gedi01_index]
                beam_elevation_error = f[f"{BEAM}/geolocation/local_beam_elevation_error"][:].flatten()[gedi01_index]

                # 位置误差
                lat_lastbin_error = f[f"{BEAM}/geolocation/latitude_lastbin_error"][:].flatten()[gedi01_index]
                lon_lastbin_error = f[f"{BEAM}/geolocation/longitude_lastbin_error"][:].flatten()[gedi01_index]

                # ========== L1A数据 - 噪声 ==========
                noise_mean = f[f"{BEAM}/noise_mean_corrected"][:].flatten()[gedi01_index]
                noise_std = f[f"{BEAM}/noise_stddev_corrected"][:].flatten()[gedi01_index]
                solar_elev = f[f"{BEAM}/geolocation/solar_elevation"][:].flatten()[gedi01_index]
                solar_azimuth = f[f"{BEAM}/geolocation/solar_azimuth"][:].flatten()[gedi01_index]

                # ========== 大地水准面校正 ==========
                elev_EGM2008_corr = f[f"{BEAM}/geophys_corr/geoid"][:].flatten()[gedi01_index]

                # ========== 波形数据读取 ==========
                # 发射波形
                txwaveformdata = f[f"{BEAM}/txwaveform"][:].flatten()
                txwaveform = waveform_read(txwaveformdata, tx_sample_start_index, tx_sample_count,
                                           np.arange(1, len_data + 1))

                # 接收波形（需要特殊处理）
                rxwaveformdata = f[f"{BEAM}/rxwaveform"][:].flatten()
                rxwaveformdata_len = len(rxwaveformdata)
                rx_sample_start_index = rx_sample_start_index_orig.copy()

                # GEDI接收波形的四种情况的处理
                #情况1：最后一个脉冲的结束位置<波形数组长度-1000 说明有1000个采样点是空值
                #情况2：其实索引超出波形数组长度
                if rx_sample_start_index_orig[-1] + rx_sample_count[-1] < rxwaveformdata_len - 1000:
                    # rxwaveform额外加0
                    rxwaveformdata = rxwaveformdata[rxwaveformdata != 0]
                elif rx_sample_start_index_orig[-1] > rxwaveformdata_len:
                    # rx_sample_start_index与rx_sample_count额外标注加0
                    for sample_start_index in range(1, len(rx_sample_start_index_orig)):
                        rx_sample_start_index[sample_start_index] = (
                                rx_sample_start_index[sample_start_index - 1] +
                                rx_sample_count[sample_start_index - 1]
                        )

                rx_sample_start_index = rx_sample_start_index[gedi01_index]
                rxwaveform = waveform_read(rxwaveformdata, rx_sample_start_index, rx_sample_count,
                                           np.arange(1, len_data + 1))

                # ========== 数据记录 ==========
                # 波形数据
                wavedata['txwaveform'] = txwaveform
                wavedata['rxwaveform'] = rxwaveform
                wavedata['rx_sample_count'] = rx_sample_count
                wavedata['rx_energy'] = rx_energy

                # 定位数据
                fpdata['ins_lat'] = ins_lat
                fpdata['ins_lon'] = ins_lon
                fpdata['ins_alt'] = ins_alt
                fpdata['elev_bin0'] = elev_bin0
                fpdata['elev_lastbin'] = elev_lastbin
                fpdata['lat_bin0'] = lat_bin0
                fpdata['lat_lastbin'] = lat_lastbin
                fpdata['lon_bin0'] = lon_bin0
                fpdata['lon_lastbin'] = lon_lastbin
                fpdata['beam_azimuth'] = beam_azimuth
                fpdata['beam_azimuth_error'] = beam_azimuth_error
                fpdata['beam_elevation'] = beam_elevation
                fpdata['beam_elevation_error'] = beam_elevation_error
                fpdata['lat_lastbin_error'] = lat_lastbin_error
                fpdata['lon_lastbin_error'] = lon_lastbin_error
                fpdata['elev_EGM2008_corr'] = elev_EGM2008_corr

                # 噪声数据
                noisedata['noise_mean'] = noise_mean
                noisedata['noise_std'] = noise_std
                noisedata['solar_elev'] = solar_elev
                noisedata['solar_azimuth'] = solar_azimuth

                # 保存到GEDIdata
                GEDIdata[beam_idx]['pointnum'] = len_data
                GEDIdata[beam_idx]['vailddata'] = len_data  # 这里简化处理，实际可能需要根据质量标识计算
                GEDIdata[beam_idx]['deltatime'] = gede01_deltatime[gedi01_index]
                GEDIdata[beam_idx]['wavedata'] = wavedata
                GEDIdata[beam_idx]['fpdata'] = fpdata
                GEDIdata[beam_idx]['noisedata'] = noisedata
                print(f"  波束 {BEAM2} 读取完成，包含 {len_data} 个脉冲")

            except Exception as e:
                print(f"  处理波束 {BEAM2} 时出错: {e}")
                GEDIdata[beam_idx]['pointnum'] = 0
                GEDIdata[beam_idx]['vailddata'] = 0
                continue

    return GEDIdata


