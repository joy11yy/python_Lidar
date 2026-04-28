import h5py
import numpy as np

# 加载 GEDI 所有足迹的经纬度
with h5py.File(r"D:\研究生\SanFrancisco\GEDIdata\merged_gedi_data_20260411.h5", 'r') as f:
    all_gedi_lats = []
    all_gedi_lons = []
    for beam_name in f.keys():
        if beam_name.startswith('BEAM') and 'fpdata' in f[beam_name]:
            fpdata = f[beam_name]['fpdata']
            if 'lat_lowestmode' in fpdata and 'lon_lowestmode' in fpdata:
                all_gedi_lats.extend(fpdata['lat_lowestmode'][:])
                all_gedi_lons.extend(fpdata['lon_lowestmode'][:])

# 加载 ICESat-2 光子
with h5py.File(r"D:\研究生\SanFrancisco\ICESatdata\processed\icesat2_photons_20260422.h5", 'r') as f:
    ph_lats = f['lat'][:]
    ph_lons = f['lon'][:]
    ph_is_signal = f['is_signal'][:]

print(f"GEDI 足迹总数: {len(all_gedi_lats)}")
print(f"ICESat-2 光子总数: {len(ph_lats):,}")
print(f"ICESat-2 信号光子数: {ph_is_signal.sum():,}")

# 计算每个 GEDI 到最近信号光子的距离
from geopy.distance import distance

min_distances = []
for i in range(min(1000, len(all_gedi_lats))):  # 只检查前1000个
    gedi_lat, gedi_lon = all_gedi_lats[i], all_gedi_lons[i]
    # 只取信号光子
    signal_mask = ph_is_signal
    min_dist = 1e10
    for j in range(len(ph_lats)):
        if not signal_mask[j]:
            continue
        try:
            dist = distance((gedi_lat, gedi_lon), (ph_lats[j], ph_lons[j])).meters
            if dist < min_dist:
                min_dist = dist
        except:
            continue
    min_distances.append(min_dist)

print(f"\n前1000个 GEDI 到最近信号光子的距离:")
print(f"  最小距离: {np.min(min_distances):.2f} 米")
print(f"  最大距离: {np.max(min_distances):.2f} 米")
print(f"  平均距离: {np.mean(min_distances):.2f} 米")
print(f"  距离 < 28 米的: {(np.array(min_distances) < 28).sum()} 个")