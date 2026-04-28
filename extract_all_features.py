import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm
from extract_photon_pseudo_waveform_features import extract_photon_pseudo_waveform_features
from extract_gedi_waveform_features import extract_gedi_waveform_features

def extract_all_features(gedi_file, output_file, bin_width=1.0, min_photons=30):
    """
       遍历融合文件中所有有光子的足迹，提取 GEDI 波形特征和光子伪波形特征，
       保存到新的 H5 文件中，分为 /gedi 和 /photon 两个组。
       """
    print("=" * 70)
    print("提取 GEDI 波形特征 + ICESat-2 伪波形特征")
    print("=" * 70)

    gedi_records = []  # 存储 GEDI 特征 + 元数据
    photon_records = []  # 存储光子特征

    with h5py.File(gedi_file, 'r') as f:
        beams = [k for k in f.keys() if k.startswith('BEAM')]
        sample_id = 0
        for beam_name in tqdm(beams, desc="处理波束"):
            beam = f[beam_name]
            fp_groups = [k for k in beam.keys() if k.startswith('fp_') and 'photons' in beam[k]]
            for fp_group in fp_groups:
                fp_idx = int(fp_group.split('_')[1])

                # 提取 GEDI 特征
                wf = extract_gedi_waveform_features(gedi_file, beam_name, fp_idx)
                if wf is None:
                    continue
                # 提取光子特征
                pwf = extract_photon_pseudo_waveform_features(gedi_file, beam_name, fp_idx,
                                                              bin_width, min_photons)
                if pwf is None:
                    continue

                # 获取经纬度
                lat, lon = None, None
                if 'fpdata' in beam and 'lat_lowestmode' in beam['fpdata']:
                    lat = beam['fpdata']['lat_lowestmode'][fp_idx]
                    lon = beam['fpdata']['lon_lowestmode'][fp_idx]

                gedi_records.append({
                    'sample_id': sample_id,
                    'beam': beam_name,
                    'footprint_idx': fp_idx,
                    'lat': lat,
                    'lon': lon,
                    **wf
                })
                photon_records.append({
                    'sample_id': sample_id,
                    **pwf
                })
                sample_id += 1

    if len(gedi_records) == 0:
        print("没有提取到任何有效特征！")
        return

    # 保存为 H5，分两个组
    with h5py.File(output_file, 'w') as h5:
        # GEDI 组
        gedi_grp = h5.create_group('gedi')
        for col in gedi_records[0].keys():
            data = np.array([rec[col] for rec in gedi_records])
            if data.dtype.kind in ['U', 'S']:
                data = data.astype('S')
            gedi_grp.create_dataset(col, data=data, compression='gzip')

        # 光子组
        photon_grp = h5.create_group('photon')
        for col in photon_records[0].keys():
            data = np.array([rec[col] for rec in photon_records])
            photon_grp.create_dataset(col, data=data, compression='gzip')

        # 全局属性
        h5.attrs['bin_width_m'] = bin_width
        h5.attrs['min_photons'] = min_photons
        h5.attrs['num_samples'] = len(gedi_records)

    print(f"\n特征提取完成！共 {len(gedi_records)} 个样本。")
    print(f"GEDI 特征组列: {list(gedi_records[0].keys())}")
    print(f"光子特征组列: {list(photon_records[0].keys())}")
    print(f"输出文件: {output_file}")


# ==================== 主程序示例 ====================
if __name__ == "__main__":
    from datetime import  datetime
    # 融合后的 GEDI 文件路径
    #fused_gedi = r"D:\研究生\SanFrancisco\GEDIdata\merged_gedi_data_20260411_with_photons_v2.h5"
    fused_gedi = r"D:\研究生\SanFrancisco\GEDIdata\merged_gedi_data_20260411_with_photons_20260427.h5"
    # 获取当前日期，格式：YYYYMMDD
    today = datetime.now().strftime("%Y%m%d")
    # 输出特征文件，带上日期
    output_h5 = rf"D:\研究生\SanFrancisco\gedi_icesat2_features_v2{today}.h5"

    extract_all_features(fused_gedi, output_h5, bin_width=1.0, min_photons=30)

    # 可选：快速查看提取的特征
    with h5py.File(output_h5, 'r') as f:
        print("\n数据集内容预览：")
        for key in f.keys():
            data = f[key][:5]  # 前5行
            print(f"{key}: {data}")

