import h5py
import matplotlib.pyplot as plt
import numpy as np


# def plot_gedi_vs_icesat2(gedi_file, icesat2_file, beam_name='BEAM0000', sample=1000):
#     """绘制GEDI足迹和ICESat-2光子的空间分布"""
#
#     # 读取GEDI足迹经纬度
#     with h5py.File(gedi_file, 'r') as f:
#         beam = f[beam_name]
#         fpdata = beam['fpdata']
#         gedi_lats = fpdata['lat_lowestmode'][:]
#         gedi_lons = fpdata['lon_lowestmode'][:]
#
#     # 读取ICESat-2光子（采样部分以便绘图）
#     with h5py.File(icesat2_file, 'r') as f:
#         ph_lats = f['lat'][::sample]  # 采样加速绘图
#         ph_lons = f['lon'][::sample]
#
#     plt.figure(figsize=(12, 8))
#     plt.scatter(ph_lons, ph_lats, s=1, c='blue', alpha=0.5, label='ICESat-2 photons')
#     plt.scatter(gedi_lons, gedi_lats, s=20, c='red', marker='x', label='GEDI footprints')
#
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title(f'{beam_name}: GEDI vs ICESat-2 spatial distribution')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.axis('equal')
#     plt.show()
def plot_all_gedi_vs_icesat2(gedi_file, icesat2_file, sample=1000):
    # 收集所有 GEDI 足迹
    all_gedi_lats = []
    all_gedi_lons = []
    with h5py.File(gedi_file, 'r') as f:
        for beam_name in f.keys():
            if beam_name.startswith('BEAM'):
                beam = f[beam_name]
                if 'fpdata' in beam and 'lat_lowestmode' in beam['fpdata']:
                    lats = beam['fpdata']['lat_lowestmode'][:]
                    lons = beam['fpdata']['lon_lowestmode'][:]
                    all_gedi_lats.extend(lats)
                    all_gedi_lons.extend(lons)
    # 读取 ICESat-2 光子（采样）
    with h5py.File(icesat2_file, 'r') as f:
        ph_lats = f['lat'][::sample]
        ph_lons = f['lon'][::sample]
    # 绘图
    plt.figure(figsize=(12, 8))
    plt.scatter(ph_lons, ph_lats, s=1, c='blue', alpha=0.5, label='ICESat-2 photons')
    plt.scatter(all_gedi_lons, all_gedi_lats, s=20, c='red', marker='x', label='GEDI footprints')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('All GEDI footprints vs ICESat-2 photons')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

# 使用
gedi_file = r"D:\研究生\SanFrancisco\GEDIdata\merged_gedi_data_20260411.h5"
icesat2_file = r"D:\研究生\SanFrancisco\ICESatdata\processed\icesat2_photons_20260422.h5"
plot_all_gedi_vs_icesat2(gedi_file, icesat2_file)